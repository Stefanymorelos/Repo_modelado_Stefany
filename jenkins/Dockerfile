# Imagen de Jenkins con Python 3 para correr el pipeline del repositorio.
FROM jenkins/jenkins:lts-jdk17

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*
USER jenkins
