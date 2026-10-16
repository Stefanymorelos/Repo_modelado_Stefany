"""Pruebas del modulo de ingesta (cargar_datos.py), sin conectarse a Google Cloud."""
from pathlib import Path

import pandas as pd
import pytest

import cargar_datos as cd

RUTA_CSV = Path(__file__).resolve().parent / "datos" / "muestra_creditos.csv"


class ClienteFalso:
    """Imita bigquery.Client: guarda la consulta y devuelve un DataFrame fijo."""

    def __init__(self, df):
        self.df = df
        self.consulta = None
        self.kwargs = None

    def query(self, consulta):
        self.consulta = consulta
        return self

    def to_dataframe(self, **kwargs):
        self.kwargs = kwargs
        return self.df


def test_leer_csv_lee_la_muestra_de_pruebas():
    df = cd.leer_csv(RUTA_CSV)
    assert df.shape == (3000, 23)
    assert "Pago_atiempo" in df.columns


def test_leer_bigquery_consulta_la_tabla_configurada():
    esperado = pd.DataFrame({"a": [1, 2]})
    cliente = ClienteFalso(esperado)
    df = cd.leer_bigquery(tabla="p.d.t", cliente=cliente)
    assert df is esperado
    assert cliente.consulta == "SELECT * FROM `p.d.t`"
    assert cliente.kwargs == {"int_dtype": None}


def test_leer_bigquery_usa_variable_de_entorno(monkeypatch):
    monkeypatch.setenv("BQ_TABLA", "otro.dataset.tabla")
    cliente = ClienteFalso(pd.DataFrame())
    cd.leer_bigquery(cliente=cliente)
    assert "otro.dataset.tabla" in cliente.consulta


def test_cargar_datos_csv():
    df = cd.cargar_datos(fuente="csv", ruta_csv=RUTA_CSV)
    assert len(df) == 3000


def test_cargar_datos_bigquery_por_defecto(monkeypatch):
    monkeypatch.delenv("FUENTE_DATOS", raising=False)
    llamado = {}
    monkeypatch.setattr(cd, "leer_bigquery", lambda: llamado.setdefault("ok", pd.DataFrame({"x": [1]})))
    df = cd.cargar_datos()
    assert "ok" in llamado
    assert list(df.columns) == ["x"]


def test_cargar_datos_fuente_invalida():
    with pytest.raises(ValueError):
        cd.cargar_datos(fuente="excel")
