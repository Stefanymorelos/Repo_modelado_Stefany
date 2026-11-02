"""
Valida que el repositorio tenga la estructura de carpetas y archivos que el
pipeline necesita. Lo ejecuta Jenkins (etapa "Validar estructura de carpetas")
antes de instalar dependencias: si falta algo, el build falla de inmediato.

Uso:
    python ci/validar_estructura.py            # desde la raiz del repositorio
    python ci/validar_estructura.py <ruta>     # validando otra carpeta
"""

from __future__ import annotations

import sys
from pathlib import Path

CARPETAS_OBLIGATORIAS = [
    "mlops_pipeline/src",
    "tests",
    "tests/datos",
    "graficas",
    "ci",
]

ARCHIVOS_OBLIGATORIOS = [
    "mlops_pipeline/src/cargar_datos.py",
    "mlops_pipeline/src/ft_engineering.py",
    "mlops_pipeline/src/heuristic_model.py",
    "mlops_pipeline/src/model_training_evaluation.py",
    "mlops_pipeline/src/model_deploy.py",
    "mlops_pipeline/src/model_evaluation.py",
    "mlops_pipeline/src/model_monitoring.py",
    "tests/datos/muestra_creditos.csv",
    "conftest.py",
    "requirements.txt",
    "Dockerfile",
    "Jenkinsfile",
    "readme.md",
]

# Archivos que NO deben estar: la base completa vive solo en BigQuery
# (Entregable 4) y las credenciales nunca se suben al repositorio.
ARCHIVOS_PROHIBIDOS = [
    "Base_de_datos.csv",
    "credenciales.json",
    "service-account.json",
]


def validar(raiz: Path) -> list[str]:
    """Devuelve la lista de errores encontrados (vacia si todo esta bien)."""
    errores = []
    for carpeta in CARPETAS_OBLIGATORIAS:
        if not (raiz / carpeta).is_dir():
            errores.append(f"Falta la carpeta: {carpeta}")
    for archivo in ARCHIVOS_OBLIGATORIOS:
        ruta = raiz / archivo
        if not ruta.is_file():
            errores.append(f"Falta el archivo: {archivo}")
        elif ruta.stat().st_size == 0:
            errores.append(f"El archivo esta vacio: {archivo}")
    for archivo in ARCHIVOS_PROHIBIDOS:
        if list(raiz.rglob(archivo)):
            errores.append(f"Archivo que no debe estar en el repositorio: {archivo}")
    pruebas = list((raiz / "tests").glob("test_*.py")) if (raiz / "tests").is_dir() else []
    if not pruebas:
        errores.append("No hay pruebas unitarias (tests/test_*.py)")
    return errores


def main() -> int:
    raiz = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    print(f"Validando estructura de: {raiz.resolve()}")
    errores = validar(raiz)
    total = len(CARPETAS_OBLIGATORIAS) + len(ARCHIVOS_OBLIGATORIOS) + len(ARCHIVOS_PROHIBIDOS) + 1
    if errores:
        for e in errores:
            print(f"  [ERROR] {e}")
        print(f"Estructura INVALIDA: {len(errores)} problema(s) de {total} revisiones.")
        return 1
    print(f"Estructura OK: {total} revisiones superadas.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
