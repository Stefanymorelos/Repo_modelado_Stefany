"""
Fase 2 (Entregable final): compara Python local vs. Vertex AI Pipelines vs. AutoML.

Todas las metricas se calculan aqui, con la misma funcion y sobre las mismas
filas de prueba (split TEST), a partir de las probabilidades de mora de cada
modelo. Asi la comparacion es justa entre las tres formas de entrenar.

Fuentes:
  - Python:            reportes_fase2/python_predicciones_test.csv  (preparar_datos_vertex.py)
  - Vertex Pipelines:  gs://<bucket>/fase2/vertex_pipeline/predicciones_test.csv
  - AutoML:            tabla de BigQuery que exporta AutoML con las predicciones de TEST

Uso:
    python vertex/comparar_resultados.py --proyecto nb-rie-vertex-dev-sm \
        --bucket nb-rie-vertex-dev-sm-vertex --tabla-automl <proyecto.dataset.tabla>

Salidas en reportes_fase2/: comparacion_final.csv y graficas fig_*.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, f1_score, log_loss, precision_score,
    recall_score, roc_auc_score, roc_curve,
)

RAIZ = Path(__file__).resolve().parent.parent
CARPETA = RAIZ / "reportes_fase2"
COLORES = {"Python local": "#1E2761", "Vertex Pipelines": "#0F7A73", "AutoML": "#B85450"}
NOMBRES = {
    "modelo_heuristico": "Heurístico", "regresion_logistica": "Regresión logística",
    "random_forest": "Random Forest", "gradient_boosting": "Gradient Boosting",
    "svm_rbf": "SVM (RBF)", "automl": "AutoML Tabular",
}


# ---------------------------------------------------------------------------
# Carga de predicciones
# ---------------------------------------------------------------------------

def a_formato_largo(df: pd.DataFrame, fuente: str) -> pd.DataFrame:
    cols = [c for c in df.columns if c.startswith("proba_mora__")]
    largo = df.melt(id_vars=["id_credito", "y_real"], value_vars=cols, var_name="modelo", value_name="proba_mora")
    largo["modelo"] = largo["modelo"].str.replace("proba_mora__", "", regex=False)
    largo["fuente"] = fuente
    return largo


def leer_vertex(bucket: str | None, ruta_local: str | None) -> pd.DataFrame | None:
    if ruta_local:
        return pd.read_csv(ruta_local)
    if not bucket:
        return None
    from google.cloud import storage

    blob = storage.Client().bucket(bucket).blob("fase2/vertex_pipeline/predicciones_test.csv")
    destino = CARPETA / "vertex_predicciones_test.csv"
    blob.download_to_filename(str(destino))
    return pd.read_csv(destino)


def _proba_clase(prediccion, clase: str) -> float:
    """Extrae la probabilidad de `clase` del campo de prediccion que exporta AutoML."""
    if isinstance(prediccion, str):
        prediccion = json.loads(prediccion)
    clases = [str(c) for c in prediccion["classes"]]
    return float(prediccion["scores"][clases.index(clase)])


def recuperar_id_credito(automl: pd.DataFrame, prueba: pd.DataFrame) -> pd.DataFrame:
    """
    AutoML no exporta las columnas excluidas del entrenamiento (id_credito).
    Se recupera el id cruzando por todas las columnas de entrada contra las
    filas TEST de la tabla de Vertex. Filas con atributos identicos reciben la
    misma prediccion, asi que basta con una fila por combinacion.
    """
    from pandas.api.types import is_numeric_dtype

    claves = [c for c in automl.columns if c in prueba.columns and c not in ("id_credito", "Pago_atiempo")]
    automl = automl.copy()
    prueba = prueba.copy()

    def como_texto(serie: pd.Series) -> pd.Series:
        if is_numeric_dtype(serie) and (serie.dropna() % 1 == 0).all():
            return serie.astype("Int64").astype(str)
        return serie.astype(str)

    for c in claves:
        if is_numeric_dtype(automl[c]) and is_numeric_dtype(prueba[c]):
            automl[c] = automl[c].astype(float).round(9)
            prueba[c] = prueba[c].astype(float).round(9)
        else:
            automl[c] = como_texto(automl[c])
            prueba[c] = como_texto(prueba[c])
    automl = automl.drop(columns=["Pago_atiempo"])
    unicos = automl.drop_duplicates(subset=claves)
    cruce = prueba[claves + ["id_credito", "Pago_atiempo"]].merge(unicos, on=claves, how="left")
    sin_cruce = cruce["proba_mora__automl"].isna().sum()
    if sin_cruce:
        raise ValueError(f"{sin_cruce} filas de TEST no encontraron su prediccion de AutoML")
    return cruce


def leer_automl(proyecto: str | None, tabla: str | None, ruta_local: str | None,
                tabla_vertex: str | None = None, ruta_prueba: str | None = None) -> pd.DataFrame | None:
    if ruta_local:
        df = pd.read_csv(ruta_local)
    elif tabla:
        from google.cloud import bigquery

        df = bigquery.Client(project=proyecto).query(f"SELECT * FROM `{tabla}`").to_dataframe()
    else:
        return None
    col_pred = next(c for c in df.columns if c.startswith("predicted_"))
    df["proba_mora__automl"] = df[col_pred].apply(lambda p: _proba_clase(p, "0"))
    df = df.drop(columns=[col_pred])

    if "id_credito" not in df.columns:
        if ruta_prueba:
            prueba = pd.read_csv(ruta_prueba)
        else:
            from google.cloud import bigquery

            prueba = bigquery.Client(project=proyecto).query(
                f"SELECT * FROM `{tabla_vertex}` WHERE split = 'TEST'").to_dataframe()
        prueba = prueba[prueba["split"] == "TEST"].drop(columns=["split"])
        df = recuperar_id_credito(df, prueba)
        print(f"AutoML: {len(df)} predicciones de TEST emparejadas con su id_credito")

    salida = pd.DataFrame({
        "id_credito": df["id_credito"].astype(int),
        "y_real": df["Pago_atiempo"].astype(int),
        "proba_mora__automl": df["proba_mora__automl"].astype(float),
    })
    salida.to_csv(CARPETA / "automl_predicciones_test.csv", index=False)
    return salida


# ---------------------------------------------------------------------------
# Metricas (clase positiva = mora, es decir Pago_atiempo = 0)
# ---------------------------------------------------------------------------

def umbral_mejor_f1(es_mora: np.ndarray, proba: np.ndarray) -> float:
    candidatos = np.linspace(0.02, 0.9, 89)
    f1s = [f1_score(es_mora, (proba >= u).astype(int), zero_division=0) for u in candidatos]
    return float(candidatos[int(np.argmax(f1s))])


def metricas(y_real: np.ndarray, proba: np.ndarray, n_boot: int = 500, semilla: int = 42) -> dict:
    es_mora = (y_real == 0).astype(int)
    pred_05 = (proba >= 0.5).astype(int)
    umbral = umbral_mejor_f1(es_mora, proba)
    pred_u = (proba >= umbral).astype(int)

    rng = np.random.default_rng(semilla)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(es_mora), len(es_mora))
        if es_mora[idx].min() != es_mora[idx].max():
            aucs.append(roc_auc_score(es_mora[idx], proba[idx]))
    aucs = np.array(aucs)

    return {
        "roc_auc": roc_auc_score(es_mora, proba),
        "pr_auc": average_precision_score(es_mora, proba),
        "log_loss": log_loss(es_mora, np.clip(proba, 1e-6, 1 - 1e-6)),
        "rmse_probabilidades": float(np.sqrt(np.mean((proba - es_mora) ** 2))),
        "accuracy_05": accuracy_score(es_mora, pred_05),
        "f1_mora_05": f1_score(es_mora, pred_05, zero_division=0),
        "recall_mora_05": recall_score(es_mora, pred_05, zero_division=0),
        "umbral_mejor_f1": umbral,
        "accuracy_umbral": accuracy_score(es_mora, pred_u),
        "precision_mora_umbral": precision_score(es_mora, pred_u, zero_division=0),
        "recall_mora_umbral": recall_score(es_mora, pred_u, zero_division=0),
        "f1_mora_umbral": f1_score(es_mora, pred_u, zero_division=0),
        "roc_auc_boot_std": float(aucs.std()),
        "roc_auc_ic95_inf": float(np.percentile(aucs, 2.5)),
        "roc_auc_ic95_sup": float(np.percentile(aucs, 97.5)),
        "_boot": aucs,
    }


# ---------------------------------------------------------------------------
# Graficas
# ---------------------------------------------------------------------------

def etiqueta(fila) -> str:
    return f"{NOMBRES.get(fila['modelo'], fila['modelo'])}\n({fila['fuente']})"


def grafica_barras(destacados: pd.DataFrame):
    columnas = [("roc_auc", "ROC-AUC"), ("pr_auc", "PR-AUC"), ("f1_mora_umbral", "F1 mora*"),
                ("recall_mora_umbral", "Recall mora*")]
    fig, ax = plt.subplots(figsize=(10, 5))
    ancho = 0.8 / len(destacados)
    x = np.arange(len(columnas))
    for i, (_, fila) in enumerate(destacados.iterrows()):
        valores = [fila[c] for c, _ in columnas]
        barras = ax.bar(x + i * ancho, valores, ancho, label=etiqueta(fila).replace("\n", " "),
                        color=COLORES[fila["fuente"]])
        ax.bar_label(barras, fmt="%.3f", fontsize=8)
    ax.set_xticks(x + ancho * (len(destacados) - 1) / 2, [n for _, n in columnas])
    ax.set_ylim(0, 1)
    ax.set_title("Mejor modelo de cada enfoque sobre la misma prueba (TEST)")
    ax.legend(fontsize=8, loc="upper right")
    ax.text(0, -0.12, "* con el umbral que maximiza F1 de cada modelo", transform=ax.transAxes, fontsize=8)
    fig.tight_layout()
    fig.savefig(CARPETA / "fig_01_metricas_mejores.png", dpi=150)
    plt.close(fig)


def grafica_roc(largo: pd.DataFrame, destacados: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 6))
    for _, fila in destacados.iterrows():
        sub = largo[(largo["fuente"] == fila["fuente"]) & (largo["modelo"] == fila["modelo"])]
        fpr, tpr, _ = roc_curve((sub["y_real"] == 0).astype(int), sub["proba_mora"])
        ax.plot(fpr, tpr, color=COLORES[fila["fuente"]], lw=2,
                label=f"{etiqueta(fila).replace(chr(10), ' ')} AUC={fila['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos (mora detectada)")
    ax.set_title("Curvas ROC sobre TEST")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(CARPETA / "fig_02_curvas_roc.png", dpi=150)
    plt.close(fig)


def grafica_estabilidad(resultados: pd.DataFrame, boots: dict):
    orden = resultados.sort_values("roc_auc", ascending=False)
    fig, ax = plt.subplots(figsize=(11, 5))
    datos = [boots[(f, m)] for f, m in zip(orden["fuente"], orden["modelo"])]
    caja = ax.boxplot(datos, patch_artist=True, showfliers=False)
    for parche, fuente in zip(caja["boxes"], orden["fuente"]):
        parche.set_facecolor(COLORES[fuente])
        parche.set_alpha(0.6)
    ax.set_xticks(range(1, len(orden) + 1), [etiqueta(f) for _, f in orden.iterrows()], fontsize=7, rotation=0)
    ax.set_ylabel("ROC-AUC (500 remuestreos bootstrap de TEST)")
    ax.set_title("Estabilidad: dispersión del ROC-AUC de cada modelo")
    fig.tight_layout()
    fig.savefig(CARPETA / "fig_03_estabilidad_bootstrap.png", dpi=150)
    plt.close(fig)


def grafica_local_vs_nube(resultados: pd.DataFrame):
    py = resultados[resultados["fuente"] == "Python local"].set_index("modelo")["roc_auc"]
    vx = resultados[resultados["fuente"] == "Vertex Pipelines"].set_index("modelo")["roc_auc"]
    comunes = [m for m in py.index if m in vx.index]
    if not comunes:
        return
    x = np.arange(len(comunes))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    b1 = ax.bar(x - 0.2, py[comunes], 0.4, label="Python local", color=COLORES["Python local"])
    b2 = ax.bar(x + 0.2, vx[comunes], 0.4, label="Vertex Pipelines", color=COLORES["Vertex Pipelines"])
    ax.bar_label(b1, fmt="%.3f", fontsize=8)
    ax.bar_label(b2, fmt="%.3f", fontsize=8)
    ax.set_xticks(x, [NOMBRES[m] for m in comunes])
    ax.set_ylim(0.5, max(py.max(), vx.max()) + 0.08)
    ax.set_ylabel("ROC-AUC en TEST")
    ax.set_title("Consistencia: mismo modelo entrenado en local y en Vertex Pipelines")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CARPETA / "fig_04_local_vs_vertex.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proyecto")
    p.add_argument("--bucket")
    p.add_argument("--tabla-automl", help="Tabla de BigQuery con las predicciones de TEST que exporta AutoML")
    p.add_argument("--vertex-csv", help="(alternativa) CSV local de predicciones de Vertex Pipelines")
    p.add_argument("--automl-csv", help="(alternativa) CSV local exportado por AutoML")
    p.add_argument("--tabla-vertex", help="Tabla con la particion (por defecto <proyecto>.riesgo_ml.creditos_vertex)")
    p.add_argument("--prueba-csv", help="(alternativa) CSV local de la tabla de Vertex, para emparejar ids")
    args = p.parse_args()

    fuentes = [a_formato_largo(pd.read_csv(CARPETA / "python_predicciones_test.csv"), "Python local")]
    vertex = leer_vertex(args.bucket, args.vertex_csv)
    if vertex is not None:
        fuentes.append(a_formato_largo(vertex, "Vertex Pipelines"))
    tabla_vertex = args.tabla_vertex or (f"{args.proyecto}.riesgo_ml.creditos_vertex" if args.proyecto else None)
    automl = leer_automl(args.proyecto, args.tabla_automl, args.automl_csv, tabla_vertex, args.prueba_csv)
    if automl is not None:
        fuentes.append(a_formato_largo(automl, "AutoML"))
    largo = pd.concat(fuentes, ignore_index=True)

    filas, boots = [], {}
    for (fuente, modelo), sub in largo.groupby(["fuente", "modelo"]):
        m = metricas(sub["y_real"].to_numpy(), sub["proba_mora"].to_numpy())
        boots[(fuente, modelo)] = m.pop("_boot")
        filas.append({"fuente": fuente, "modelo": modelo, "filas_test": len(sub), **m})
    resultados = pd.DataFrame(filas).sort_values(["fuente", "roc_auc"], ascending=[True, False])
    resultados.round(4).to_csv(CARPETA / "comparacion_final.csv", index=False)

    destacados = resultados.loc[resultados.groupby("fuente")["roc_auc"].idxmax()]
    grafica_barras(destacados)
    grafica_roc(largo, destacados)
    grafica_estabilidad(resultados, boots)
    grafica_local_vs_nube(resultados)

    cols = ["fuente", "modelo", "roc_auc", "pr_auc", "log_loss", "rmse_probabilidades",
            "f1_mora_umbral", "recall_mora_umbral", "roc_auc_boot_std"]
    print(resultados[cols].round(3).to_string(index=False))
    print(f"\nArchivos en {CARPETA}: comparacion_final.csv y fig_01..fig_04")


if __name__ == "__main__":
    main()
