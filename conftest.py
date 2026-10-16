"""
Configuracion compartida de pytest.

Agrega mlops_pipeline/src al path de Python para poder hacer
`from ft_engineering import ...` en los tests, sin necesidad de
convertir mlops_pipeline en un paquete (evita modificar la estructura
de carpetas fija del proyecto).
"""
import os
import sys
from pathlib import Path

SRC_PATH = Path(__file__).resolve().parent / "mlops_pipeline" / "src"
sys.path.insert(0, str(SRC_PATH))

# Las pruebas nunca se conectan a Google Cloud (GitHub Actions no tiene
# credenciales): cargar_datos() lee una muestra local de 3.000 creditos
# (tests/datos/muestra_creditos.csv) en vez de BigQuery. La base completa
# ya no esta en el repositorio: su unica fuente es BigQuery.
os.environ["FUENTE_DATOS"] = "csv"
os.environ["RUTA_CSV"] = str(Path(__file__).resolve().parent / "tests" / "datos" / "muestra_creditos.csv")
