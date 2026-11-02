"""
Fase 2 (Entregable final): prepara los datos para entrenar en Vertex AI y
genera la linea base de Python con exactamente la misma particion.

Que hace:
  1. Lee la base completa desde BigQuery (cargar_datos.leer_bigquery).
  2. La ordena de forma determinista (BigQuery no garantiza el orden de las
     filas) y aplica la misma limpieza del repositorio (ft_engineering).
  3. Reproduce la particion del repo: 75 % entrenamiento / 25 % prueba,
     estratificada, random_state=42. Del entrenamiento separa un 15 % como
     VALIDATE para AutoML (lo usa para ajustar hiperparametros).
  4. Sube la tabla lista a BigQuery con una columna `split`
     (TRAIN / VALIDATE / TEST) y un identificador `id_credito`.
     -> AutoML, Vertex Pipelines y Python se evaluan sobre las MISMAS filas de prueba.
  5. Entrena los modelos del repo (entrenar_y_comparar_modelos) y guarda las
     probabilidades sobre la prueba, para compararlas despues.

Uso (desde la raiz del repositorio, con `gcloud auth application-default login` hecho):
    python vertex/preparar_datos_vertex.py --proyecto nb-rie-vertex-dev-sm

Salidas:
    BigQuery: <proyecto>.riesgo_ml.creditos_vertex
    reportes_fase2/python_predicciones_test.csv
    reportes_fase2/python_resumen.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "mlops_pipeline" / "src"))

from cargar_datos import leer_bigquery, leer_csv  # noqa: E402
from ft_engineering import CATEGORICAL_FEATURES, TARGET_COL, build_features  # noqa: E402
from model_training_evaluation import (  # noqa: E402
    NUMERIC_FEATURES_MODELO,
    RANDOM_STATE,
    entrenar_y_comparar_modelos,
)

CARPETA_SALIDA = RAIZ / "reportes_fase2"
COLUMNAS_MODELO = NUMERIC_FEATURES_MODELO + CATEGORICAL_FEATURES


def ordenar_determinista(df: pd.DataFrame) -> pd.DataFrame:
    """Mismo orden de filas sin importar como las entregue BigQuery."""
    return df.sort_values(by=list(df.columns), kind="mergesort", na_position="last").reset_index(drop=True)


def construir_tabla_vertex(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla limpia con las columnas del modelo + objetivo + split + id."""
    X_train, X_test, y_train, y_test = build_features(df)

    _, idx_validate = train_test_split(
        X_train.index, test_size=0.15, stratify=y_train, random_state=RANDOM_STATE
    )
    entrenamiento = X_train[COLUMNAS_MODELO].copy()
    entrenamiento[TARGET_COL] = y_train
    entrenamiento["split"] = np.where(entrenamiento.index.isin(idx_validate), "VALIDATE", "TRAIN")

    prueba = X_test[COLUMNAS_MODELO].copy()
    prueba[TARGET_COL] = y_test
    prueba["split"] = "TEST"

    tabla = pd.concat([entrenamiento, prueba])
    tabla.insert(0, "id_credito", tabla.index.astype(int))
    for col in CATEGORICAL_FEATURES:
        tabla[col] = tabla[col].astype(str)
    # Objetivo como texto: AutoML lo trata como clasificacion sin ambiguedad.
    tabla[TARGET_COL] = tabla[TARGET_COL].astype(int).astype(str)
    return tabla.reset_index(drop=True)


def subir_a_bigquery(tabla: pd.DataFrame, proyecto: str, dataset: str, nombre: str, ubicacion: str) -> str:
    from google.cloud import bigquery

    cliente = bigquery.Client(project=proyecto)
    ds = bigquery.Dataset(f"{proyecto}.{dataset}")
    ds.location = ubicacion
    cliente.create_dataset(ds, exists_ok=True)
    destino = f"{proyecto}.{dataset}.{nombre}"
    config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    cliente.load_table_from_dataframe(tabla, destino, job_config=config).result()
    return destino


def linea_base_python(df: pd.DataFrame, tabla: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Entrena los modelos del repo y devuelve sus probabilidades de mora en TEST."""
    inicio = time.perf_counter()
    resultado = entrenar_y_comparar_modelos(df, fracciones_escalabilidad=(1.0,))
    duracion = time.perf_counter() - inicio

    X_test, y_test = resultado["X_test"], resultado["y_test"]
    ids_test = set(tabla.loc[tabla["split"] == "TEST", "id_credito"])
    assert set(X_test.index.astype(int)) == ids_test, "La particion de prueba no coincide con la tabla de Vertex"

    predicciones = pd.DataFrame({"id_credito": X_test.index.astype(int), "y_real": y_test.astype(int).values})
    for nombre, res in resultado["resultados_individuales"].items():
        predicciones[f"proba_mora__{nombre}"] = 1 - res.pipeline.predict_proba(X_test)[:, 1]

    tabla_cmp = resultado["tabla_comparativa"]
    resumen = {
        "duracion_total_s": round(duracion, 1),
        "mejor_modelo": resultado["mejor_modelo"],
        "modelos": {
            fila["modelo"]: {
                "roc_auc_cv_media": round(float(fila["roc_auc_cv_media"]), 4),
                "roc_auc_cv_std": round(float(fila["roc_auc_cv_std"]), 4),
                "tiempo_ajuste_s": round(float(fila["tiempo_ajuste_s"]), 2),
            }
            for _, fila in tabla_cmp.iterrows()
        },
    }
    return predicciones, resumen


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proyecto", required=True, help="Proyecto de GCP con Vertex AI (destino)")
    p.add_argument("--tabla-origen", default="nb-ret-app-dev-sm.riesgo_credito.creditos")
    p.add_argument("--dataset", default="riesgo_ml")
    p.add_argument("--tabla-destino", default="creditos_vertex")
    p.add_argument("--ubicacion", default="US")
    p.add_argument("--csv", help="(solo pruebas) leer un CSV local en vez de BigQuery y no subir nada")
    args = p.parse_args()

    CARPETA_SALIDA.mkdir(exist_ok=True)
    df = leer_csv(args.csv) if args.csv else leer_bigquery(tabla=args.tabla_origen, proyecto=args.proyecto)
    df = ordenar_determinista(df)
    print(f"Datos leidos: {df.shape[0]} filas x {df.shape[1]} columnas")

    tabla = construir_tabla_vertex(df)
    print("Filas por particion:", tabla["split"].value_counts().to_dict())
    print("Tasa de mora por particion:",
          tabla.assign(mora=tabla[TARGET_COL].eq("0")).groupby("split")["mora"].mean().round(4).to_dict())
    tabla.to_csv(CARPETA_SALIDA / "tabla_vertex.csv", index=False)

    if not args.csv:
        destino = subir_a_bigquery(tabla, args.proyecto, args.dataset, args.tabla_destino, args.ubicacion)
        print(f"Tabla subida a BigQuery: {destino}")

    print("Entrenando la linea base de Python (los 5 modelos del repositorio)...")
    predicciones, resumen = linea_base_python(df, tabla)
    predicciones.to_csv(CARPETA_SALIDA / "python_predicciones_test.csv", index=False)
    (CARPETA_SALIDA / "python_resumen.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")
    print(f"Linea base lista en {resumen['duracion_total_s']} s. Mejor modelo: {resumen['mejor_modelo']}")
    print(f"Archivos en: {CARPETA_SALIDA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
