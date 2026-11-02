"""
Entrena los modelos del pipeline y valida que sus metricas de evaluacion
cumplan los minimos acordados. Lo ejecuta Jenkins (etapa "Entrenar y validar
metricas"): si el mejor modelo no llega a los umbrales, el build falla y no
pasa a la aprobacion.

Datos: usa cargar_datos(), asi que respeta FUENTE_DATOS. En Jenkins se usa la
muestra de 3.000 creditos (FUENTE_DATOS=csv) porque el contenedor no tiene
credenciales de Google Cloud; con FUENTE_DATOS=bigquery lee la tabla completa.

Umbrales (variables de entorno, con su valor por defecto):
    UMBRAL_ROC_AUC     0.60  ROC-AUC minimo del mejor modelo en el holdout
    UMBRAL_RECALL      0.30  recall minimo de la clase mora (la que importa al negocio)
    UMBRAL_STD_CV      0.06  desviacion maxima del ROC-AUC en validacion cruzada (estabilidad)

Salidas (carpeta reportes/):
    metricas.json            metricas del mejor modelo + resultado de cada validacion
    comparacion_modelos.csv  tabla comparativa de todos los candidatos
    mejor_modelo.joblib      modelo ganador, listo para model_deploy.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "mlops_pipeline" / "src"))

from cargar_datos import cargar_datos  # noqa: E402
from model_training_evaluation import entrenar_y_comparar_modelos  # noqa: E402

CARPETA_SALIDA = RAIZ / "reportes"


def umbral(nombre: str, por_defecto: float) -> float:
    return float(os.environ.get(nombre, por_defecto))


def validar_metricas(fila: dict) -> list[dict]:
    """Compara las metricas del mejor modelo contra los umbrales."""
    reglas = [
        ("ROC-AUC holdout", fila["roc_auc_holdout"], ">=", umbral("UMBRAL_ROC_AUC", 0.60)),
        ("Recall mora holdout", fila["recall_mora_holdout"], ">=", umbral("UMBRAL_RECALL", 0.30)),
        ("Desviacion ROC-AUC en CV", fila["roc_auc_cv_std"], "<=", umbral("UMBRAL_STD_CV", 0.06)),
    ]
    resultados = []
    for nombre, valor, op, limite in reglas:
        ok = valor >= limite if op == ">=" else valor <= limite
        resultados.append({"metrica": nombre, "valor": round(float(valor), 4),
                           "condicion": f"{op} {limite}", "cumple": bool(ok)})
    return resultados


def main() -> int:
    CARPETA_SALIDA.mkdir(exist_ok=True)
    df = cargar_datos()
    print(f"Datos cargados: {df.shape[0]} filas x {df.shape[1]} columnas "
          f"(fuente: {os.environ.get('FUENTE_DATOS', 'bigquery')})")

    resultado = entrenar_y_comparar_modelos(df, ruta_modelo_final=CARPETA_SALIDA / "mejor_modelo.joblib")
    tabla = resultado["tabla_comparativa"]
    tabla.to_csv(CARPETA_SALIDA / "comparacion_modelos.csv", index=False)
    print("\nComparacion de modelos (ordenados por ROC-AUC en holdout):")
    print(tabla[["modelo", "roc_auc_holdout", "roc_auc_cv_media", "roc_auc_cv_std",
                 "recall_mora_holdout", "f1_mora_holdout"]].round(3).to_string(index=False))

    mejor = tabla.iloc[0].to_dict()
    resumen = resultado["resumenes_holdout"][mejor["modelo"]]
    validaciones = validar_metricas(mejor)
    aprobado = all(v["cumple"] for v in validaciones)

    reporte = {
        "mejor_modelo": mejor["modelo"],
        "filas_usadas": int(df.shape[0]),
        "metricas_holdout": {
            "accuracy": round(float(resumen["accuracy"]), 4),
            "precision_mora": round(float(resumen["precision_mora"]), 4),
            "recall_mora": round(float(resumen["recall_mora"]), 4),
            "f1_mora": round(float(resumen["f1_mora"]), 4),
            "roc_auc": round(float(resumen["roc_auc"]), 4),
        },
        "roc_auc_cv": {"media": round(float(mejor["roc_auc_cv_media"]), 4),
                       "std": round(float(mejor["roc_auc_cv_std"]), 4)},
        "validaciones": validaciones,
        "aprobado": aprobado,
    }
    (CARPETA_SALIDA / "metricas.json").write_text(json.dumps(reporte, indent=2, ensure_ascii=False), encoding="utf-8")

    lineas = [f"Mejor modelo: {mejor['modelo']} ({df.shape[0]} filas)"]
    lineas += [f"  {k}: {v}" for k, v in reporte["metricas_holdout"].items()]
    lineas.append("Validaciones:")
    for v in validaciones:
        estado = "OK   " if v["cumple"] else "FALLA"
        lineas.append(f"  [{estado}] {v['metrica']}: {v['valor']} (debe ser {v['condicion']})")
    lineas.append("Metricas APROBADAS." if aprobado else "Metricas NO cumplen los minimos.")
    texto = "\n".join(lineas)
    (CARPETA_SALIDA / "resumen.txt").write_text(texto + "\n", encoding="utf-8")
    print("\n" + texto)
    return 0 if aprobado else 1


if __name__ == "__main__":
    sys.exit(main())
