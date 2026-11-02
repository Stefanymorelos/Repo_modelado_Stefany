// Pipeline de aprobacion del modelo de riesgo de credito (Entregable final, Fase 1).
//
// Flujo: clonar -> validar estructura -> preparar entorno -> pruebas unitarias
//        -> entrenar y validar metricas -> aprobacion manual -> publicar artefactos
//        -> notificacion por correo (siempre, pase o falle).
//
// Disparador: Jenkins revisa la rama master cada 2 minutos (pollSCM); cuando
// entra un merge nuevo a master, el pipeline arranca solo.

pipeline {
    agent any

    options {
        skipDefaultCheckout(true)        // el clonado se hace explicito en la primera etapa
        timestamps()
        buildDiscarder(logRotator(numToKeepStr: '15'))
        timeout(time: 90, unit: 'MINUTES')
    }

    triggers {
        pollSCM('H/2 * * * *')
    }

    environment {
        REPO_URL       = 'https://github.com/Stefanymorelos/repo_modelado_Stefany.git'
        RAMA           = 'master'
        CORREO         = 'stefanymorelos29@gmail.com'          // 
        VENV           = "${JENKINS_HOME}/venvs/riesgo-credito"
        FUENTE_DATOS   = 'csv'                           // el contenedor no tiene credenciales de GCP
        MPLBACKEND     = 'Agg'                           // graficas sin pantalla
        UMBRAL_ROC_AUC = '0.60'
        UMBRAL_RECALL  = '0.30'
        UMBRAL_STD_CV  = '0.06'
    }

    stages {
        stage('Clonar repositorio') {
            steps {
                deleteDir()
                git branch: "${RAMA}", url: "${REPO_URL}"
                sh 'git log -1 --pretty="Commit: %h  %an  %s"'
            }
        }

        stage('Validar estructura de carpetas') {
            steps {
                sh 'python3 ci/validar_estructura.py'
            }
        }

        stage('Preparar entorno') {
            steps {
                sh '''
                    [ -d "$VENV" ] || python3 -m venv "$VENV"
                    . "$VENV/bin/activate"
                    pip install --quiet --upgrade pip
                    pip install --quiet --timeout 120 --retries 5 -r requirements.txt pytest pytest-cov
                    python --version
                '''
            }
        }

        stage('Pruebas unitarias') {
            steps {
                sh '''
                    . "$VENV/bin/activate"
                    mkdir -p reportes
                    pytest tests -q --junitxml=reportes/pytest.xml --cov=mlops_pipeline --cov-report=term
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'reportes/pytest.xml'
                }
            }
        }

        stage('Entrenar y validar metricas') {
            steps {
                sh '''
                    . "$VENV/bin/activate"
                    export RUTA_CSV="$WORKSPACE/tests/datos/muestra_creditos.csv"
                    python ci/entrenar_y_validar.py
                '''
            }
        }

        stage('Aprobacion') {
            steps {
                timeout(time: 30, unit: 'MINUTES') {
                    input message: 'Las pruebas y las metricas pasaron. ¿Aprobar el modelo para despliegue?',
                          ok: 'Aprobar'
                }
            }
        }

        stage('Publicar artefactos') {
            steps {
                archiveArtifacts artifacts: 'reportes/**', fingerprint: true
                echo 'Modelo aprobado: reportes/mejor_modelo.joblib queda archivado en este build.'
            }
        }
    }

    post {
        always {
            emailext(
                to: "${CORREO}",
                subject: "Jenkins ${currentBuild.currentResult}: ${env.JOB_NAME} #${env.BUILD_NUMBER}",
                mimeType: 'text/plain',
                body: """Resultado del pipeline: ${currentBuild.currentResult}
Proyecto: ${env.JOB_NAME}  (build #${env.BUILD_NUMBER})
Duracion: ${currentBuild.durationString}
Detalle: ${env.BUILD_URL}

\${FILE,path="reportes/resumen.txt"}
""",
                attachmentsPattern: 'reportes/metricas.json, reportes/comparacion_modelos.csv'
            )
        }
    }
}
