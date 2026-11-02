"""
Fase 2 (Entregable final): pipeline custom en Vertex AI Pipelines (Kubeflow v2).

Reproduce en la nube el entrenamiento del repositorio y lo deja como un flujo
reproducible y versionado:

    preparar-datos  ->  entrenar-evaluar (x4, en paralelo)  ->  comparar-modelos

  - preparar-datos: lee la tabla de BigQuery creada por preparar_datos_vertex.py
    (misma limpieza y misma particion TRAIN/VALIDATE/TEST que Python y AutoML).
  - entrenar-evaluar: un paso por modelo (regresion logistica, random forest,
    gradient boosting y SVM), con el mismo preprocesador e hiperparametros del
    repositorio. Mide consistencia con validacion cruzada estratificada de 5
    folds y desempeno en TEST. Registra metricas, curva ROC y matriz de
    confusion en la interfaz de Vertex.
  - comparar-modelos: elige el mejor por ROC-AUC y guarda en Cloud Storage la
    tabla de metricas y las probabilidades sobre TEST.

Uso (desde la raiz del repositorio):
    python vertex/pipeline_vertex.py --proyecto nb-rie-vertex-dev-sm --bucket nb-rie-vertex-dev-sm-vertex

Resultados en: gs://<bucket>/fase2/vertex_pipeline/
"""


import argparse
from pathlib import Path

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Metrics, ClassificationMetrics, Model, Output

IMAGEN = "python:3.11-slim"
PAQUETES = ["numpy==2.1.3", "pandas==2.2.3", "scikit-learn==1.5.2", "joblib==1.4.2",
            "pyarrow==18.1.0", "db-dtypes==1.3.1", "google-cloud-bigquery==3.27.0",
            "google-cloud-storage==2.19.0"]
MODELOS = ["regresion_logistica", "random_forest", "gradient_boosting", "svm_rbf"]


@dsl.component(base_image=IMAGEN, packages_to_install=PAQUETES)
def preparar_datos(proyecto: str, tabla_bq: str, entrenamiento: Output[Dataset], prueba: Output[Dataset]):
    from google.cloud import bigquery

    df = bigquery.Client(project=proyecto).query(f"SELECT * FROM `{tabla_bq}`").to_dataframe()
    df = df.sort_values("id_credito").reset_index(drop=True)
    df[df["split"] != "TEST"].to_csv(entrenamiento.path, index=False)
    df[df["split"] == "TEST"].to_csv(prueba.path, index=False)
    entrenamiento.metadata["filas"] = int((df["split"] != "TEST").sum())
    prueba.metadata["filas"] = int((df["split"] == "TEST").sum())


@dsl.component(base_image=IMAGEN, packages_to_install=PAQUETES)
def entrenar_evaluar(
    nombre_modelo: str,
    entrenamiento: Input[Dataset],
    prueba: Input[Dataset],
    modelo: Output[Model],
    metricas: Output[Metrics],
    metricas_clasificacion: Output[ClassificationMetrics],
    predicciones: Output[Dataset],
    resumen: Output[Dataset],
):
    import json
    import time

    import joblib
    import numpy as np
    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                                 log_loss, precision_score, recall_score, roc_auc_score, roc_curve)
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.svm import SVC

    semilla = 42
    objetivo = "Pago_atiempo"
    categoricas = ["tipo_laboral", "tipo_credito_agrupado", "tendencia_ingresos"]

    train = pd.read_csv(entrenamiento.path, dtype={c: str for c in categoricas})
    test = pd.read_csv(prueba.path, dtype={c: str for c in categoricas})
    numericas = [c for c in train.columns if c not in categoricas + [objetivo, "split", "id_credito"]]
    X_train, y_train = train[numericas + categoricas], train[objetivo].astype(int)
    X_test, y_test = test[numericas + categoricas], test[objetivo].astype(int)

    estimadores = {
        "regresion_logistica": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=semilla),
        "random_forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                random_state=semilla, n_jobs=-1),
        "gradient_boosting": HistGradientBoostingClassifier(class_weight="balanced", random_state=semilla),
        "svm_rbf": SVC(kernel="rbf", class_weight="balanced", probability=True, random_state=semilla),
    }
    preprocesador = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numericas),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categoricas),
    ])
    pipe = Pipeline([("preprocesador", preprocesador), ("modelo", estimadores[nombre_modelo])])

    # Consistencia: validacion cruzada estratificada de 5 folds sobre entrenamiento.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=semilla)
    auc_cv = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

    inicio = time.perf_counter()
    pipe.fit(X_train, y_train)
    tiempo_ajuste = time.perf_counter() - inicio

    # La clase de interes es la mora (Pago_atiempo = 0).
    proba_mora = 1 - pipe.predict_proba(X_test)[:, 1]
    es_mora = (y_test == 0).astype(int).values
    pred_mora = (pipe.predict(X_test) == 0).astype(int)

    valores = {
        "accuracy": accuracy_score(es_mora, pred_mora),
        "precision_mora": precision_score(es_mora, pred_mora, zero_division=0),
        "recall_mora": recall_score(es_mora, pred_mora, zero_division=0),
        "f1_mora": f1_score(es_mora, pred_mora, zero_division=0),
        "roc_auc": roc_auc_score(es_mora, proba_mora),
        "pr_auc": average_precision_score(es_mora, proba_mora),
        "log_loss": log_loss(es_mora, np.clip(proba_mora, 1e-6, 1 - 1e-6)),
        "rmse_probabilidades": float(np.sqrt(np.mean((proba_mora - es_mora) ** 2))),
        "roc_auc_cv_media": float(auc_cv.mean()),
        "roc_auc_cv_std": float(auc_cv.std()),
        "tiempo_ajuste_s": tiempo_ajuste,
    }
    for clave, valor in valores.items():
        metricas.log_metric(clave, round(float(valor), 4))

    fpr, tpr, umbrales = roc_curve(es_mora, proba_mora)
    paso = max(1, len(fpr) // 150)
    metricas_clasificacion.log_roc_curve(
        fpr[::paso].tolist(), tpr[::paso].tolist(), np.clip(umbrales[::paso], 0, 1).tolist())
    metricas_clasificacion.log_confusion_matrix(
        ["al_dia", "mora"], confusion_matrix(es_mora, pred_mora, labels=[0, 1]).tolist())

    joblib.dump(pipe, modelo.path)
    modelo.metadata["framework"] = "scikit-learn"
    pd.DataFrame({"id_credito": test["id_credito"], "y_real": y_test,
                  f"proba_mora__{nombre_modelo}": proba_mora}).to_csv(predicciones.path, index=False)
    with open(resumen.path, "w") as f:
        json.dump({"modelo": nombre_modelo, **{k: round(float(v), 4) for k, v in valores.items()}}, f)


@dsl.component(base_image=IMAGEN, packages_to_install=PAQUETES)
def comparar_modelos(
    bucket: str,
    res_1: Input[Dataset], res_2: Input[Dataset], res_3: Input[Dataset], res_4: Input[Dataset],
    pred_1: Input[Dataset], pred_2: Input[Dataset], pred_3: Input[Dataset], pred_4: Input[Dataset],
    tabla: Output[Dataset],
    metricas: Output[Metrics],
) -> str:
    import json

    import pandas as pd
    from google.cloud import storage

    filas = [json.load(open(r.path)) for r in (res_1, res_2, res_3, res_4)]
    comparacion = pd.DataFrame(filas).sort_values("roc_auc", ascending=False)
    comparacion.to_csv(tabla.path, index=False)
    mejor = comparacion.iloc[0]
    metricas.log_metric("mejor_roc_auc", float(mejor["roc_auc"]))
    metricas.metadata["mejor_modelo"] = str(mejor["modelo"])

    preds = None
    for p in (pred_1, pred_2, pred_3, pred_4):
        df = pd.read_csv(p.path)
        preds = df if preds is None else preds.merge(df, on=["id_credito", "y_real"])

    cliente = storage.Client()
    destino = cliente.bucket(bucket)
    destino.blob("fase2/vertex_pipeline/comparacion_modelos.csv").upload_from_string(
        comparacion.to_csv(index=False), "text/csv")
    destino.blob("fase2/vertex_pipeline/predicciones_test.csv").upload_from_string(
        preds.to_csv(index=False), "text/csv")
    return str(mejor["modelo"])


@dsl.pipeline(name="riesgo-credito-pipeline-custom",
              description="Entrenamiento y comparacion de modelos de riesgo de credito (scikit-learn)")
def pipeline_riesgo(proyecto: str, tabla_bq: str, bucket: str):
    datos = preparar_datos(proyecto=proyecto, tabla_bq=tabla_bq).set_display_name("preparar-datos")
    tareas = []
    for nombre in MODELOS:
        t = entrenar_evaluar(
            nombre_modelo=nombre,
            entrenamiento=datos.outputs["entrenamiento"],
            prueba=datos.outputs["prueba"],
        ).set_display_name(f"entrenar-{nombre}").set_cpu_limit("4").set_memory_limit("16G")
        tareas.append(t)
    comparar_modelos(
        bucket=bucket,
        res_1=tareas[0].outputs["resumen"], res_2=tareas[1].outputs["resumen"],
        res_3=tareas[2].outputs["resumen"], res_4=tareas[3].outputs["resumen"],
        pred_1=tareas[0].outputs["predicciones"], pred_2=tareas[1].outputs["predicciones"],
        pred_3=tareas[2].outputs["predicciones"], pred_4=tareas[3].outputs["predicciones"],
    ).set_display_name("comparar-modelos")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--proyecto", required=True)
    p.add_argument("--bucket", required=True, help="Nombre del bucket (sin gs://)")
    p.add_argument("--region", default="us-central1")
    p.add_argument("--tabla-bq", help="Por defecto <proyecto>.riesgo_ml.creditos_vertex")
    p.add_argument("--solo-compilar", action="store_true")
    p.add_argument("--con-cache", action="store_true", help="Reutilizar pasos ya ejecutados")
    args = p.parse_args()

    plantilla = Path(__file__).resolve().parent / "pipeline_riesgo_credito.json"
    compiler.Compiler().compile(pipeline_riesgo, str(plantilla))
    print(f"Pipeline compilado: {plantilla}")
    if args.solo_compilar:
        return

    from google.cloud import aiplatform

    aiplatform.init(project=args.proyecto, location=args.region, staging_bucket=f"gs://{args.bucket}")
    trabajo = aiplatform.PipelineJob(
        display_name="riesgo-credito-pipeline-custom",
        template_path=str(plantilla),
        pipeline_root=f"gs://{args.bucket}/pipeline_root",
        parameter_values={
            "proyecto": args.proyecto,
            "tabla_bq": args.tabla_bq or f"{args.proyecto}.riesgo_ml.creditos_vertex",
            "bucket": args.bucket,
        },
        enable_caching=args.con_cache,
    )
    trabajo.submit()
    print("Pipeline enviado. Siguelo en la consola: Vertex AI > Pipelines.")
    print(f"Resultados al terminar: gs://{args.bucket}/fase2/vertex_pipeline/")


if __name__ == "__main__":
    main()
