"""
Ingesta de datos del pipeline de riesgo de credito.

Desde el Entregable 4 (Landing Zone de NovaBank en Google Cloud) la fuente de
los datos ya no es el archivo Base_de_datos.csv sino la tabla de BigQuery del
proyecto de Retail Banking:

    nb-ret-app-dev-sm.riesgo_credito.creditos

Este modulo es el unico lugar que sabe de donde vienen los datos. El resto del
pipeline (EDA, ft_engineering, entrenamiento, despliegue y monitoreo) sigue
recibiendo un DataFrame con las mismas columnas y los mismos tipos que antes.

Variables de entorno opcionales:
    FUENTE_DATOS   "bigquery" (por defecto) o "csv" (respaldo local y pruebas)
    BQ_PROYECTO    proyecto de Google Cloud (por defecto nb-ret-app-dev-sm)
    BQ_TABLA       tabla completa proyecto.dataset.tabla
    RUTA_CSV       CSV a usar si FUENTE_DATOS=csv (las pruebas usan una muestra)

Autenticacion: gcloud auth application-default login (sin llaves JSON).
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

BQ_PROYECTO = "nb-ret-app-dev-sm"
BQ_TABLA = "nb-ret-app-dev-sm.riesgo_credito.creditos"
RUTA_CSV = "Base_de_datos.csv"


def leer_bigquery(tabla: str | None = None, proyecto: str | None = None, cliente=None) -> pd.DataFrame:
    """
    Lee la tabla completa de BigQuery.

    int_dtype=None hace que los enteros lleguen como int64 de numpy, igual que
    con pd.read_csv (sin esto llegarian como Int64 de pandas). El parametro
    `cliente` permite usar un cliente falso en las pruebas unitarias.
    """
    tabla = tabla or os.environ.get("BQ_TABLA", BQ_TABLA)
    proyecto = proyecto or os.environ.get("BQ_PROYECTO", BQ_PROYECTO)
    if cliente is None:
        from google.cloud import bigquery  # solo se importa si se usa BigQuery

        cliente = bigquery.Client(project=proyecto)
    return cliente.query(f"SELECT * FROM `{tabla}`").to_dataframe(int_dtype=None)


def leer_csv(ruta_csv: str | Path | None = None) -> pd.DataFrame:
    """Lectura de un CSV con el formato original (separador ';'). Lo usan las pruebas."""
    ruta = ruta_csv or os.environ.get("RUTA_CSV", RUTA_CSV)
    return pd.read_csv(ruta, sep=";", encoding="utf-8-sig")


def cargar_datos(fuente: str | None = None, ruta_csv: str | Path | None = None) -> pd.DataFrame:
    """Punto unico de ingesta: BigQuery por defecto, CSV si FUENTE_DATOS=csv."""
    fuente = (fuente or os.environ.get("FUENTE_DATOS", "bigquery")).strip().lower()
    if fuente == "bigquery":
        return leer_bigquery()
    if fuente == "csv":
        return leer_csv(ruta_csv)
    raise ValueError(f"FUENTE_DATOS no valida: {fuente!r}. Usa 'bigquery' o 'csv'.")
