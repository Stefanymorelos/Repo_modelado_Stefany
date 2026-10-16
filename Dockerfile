# Imagen para servir el modelo de riesgo de credito via FastAPI.
#
# Construir:
#   docker build -t riesgo-credito-api .
#
# Correr (monta el modelo entrenado y las credenciales de gcloud, para que
# la API lea los datos historicos de BigQuery -- ver cargar_datos.py):
#   docker run -p 8000:8000 \
#       -v "$(pwd)/mejor_modelo_final.joblib:/app/mejor_modelo_final.joblib" \
#       -v "$HOME/.config/gcloud:/root/.config/gcloud" \
#       riesgo-credito-api

FROM python:3.11-slim

WORKDIR /app

# Solo las dependencias necesarias para SERVIR el modelo (no las de
# entrenamiento/EDA como jupyter, seaborn, torch, etc.) -- la imagen de
# despliegue debe ser lo mas liviana posible.
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY mlops_pipeline/src/cargar_datos.py .
COPY mlops_pipeline/src/ft_engineering.py .
COPY mlops_pipeline/src/model_deploy.py .

EXPOSE 8000

CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
