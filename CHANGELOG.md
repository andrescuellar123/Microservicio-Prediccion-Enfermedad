# CHANGELOG

Todos los cambios relevantes entre versiones del proyecto se documentan en este archivo.

El formato sigue el estándar [Keep a Changelog](https://keepachangelog.com/es/1.0.0/).

---

## [2.0.0] — 2026-05-31

### Resumen del cambio

Esta versión representa una reescritura completa de la documentación del proyecto. La propuesta original describía un microservicio de predicción simulada sin modelo real, sin pipeline de MLOps y con documentación mínima. La versión actual documenta un pipeline end-to-end completo con entrenamiento, evaluación, despliegue, monitoreo y criterios de aprobación clínicamente justificados.

---

### AÑADIDO

#### Definición del problema
- Se agregaron las 12 variables de entrada clínicas completas: `edad`, `sexo`, `temperatura`, `frecuencia_cardiaca`, `presion_sistolica`, `duracion_sintomas_dias`, `nivel_dolor`, `fiebre`, `tos`, `fatiga`, `perdida_peso`, `antecedentes_cronicos`. La propuesta original solo contemplaba 5 variables (`edad`, `fiebre`, `dolor`, `duracion_dias`, `condicion_cronica`).
- Se agregó la distinción entre dos escenarios: enfermedades comunes y enfermedades huérfanas.
- Se declaró explícitamente que el escenario de enfermedades huérfanas es trabajo futuro y no está implementado en esta versión.

#### Arquitectura general
- Se agregó el diagrama Mermaid completo del pipeline MLOps end-to-end con todas las etapas: ingesta, validación, preprocesamiento, entrenamiento, evaluación, aprobación humana, despliegue, predicción, logging y monitoreo.
- Se agregó el nodo de **aprobación humana** del equipo ML entre la evaluación del modelo y el despliegue, dejando explícito que la promoción a producción no es automática.

#### Etapa: GitHub y GitHub Actions
- Se agregó la justificación de GitHub frente a alternativas (GitLab CI, Jenkins, CircleCI).
- Se agregó la estrategia de ramas: `main`, `develop`, `feature/*` con reglas de protección.
- Se documentó el comportamiento ante fallo del CI: el merge queda bloqueado y se genera notificación en el pull request.
- Se detalló el flujo completo de CI con 6 pasos explícitos incluyendo Pytest, flake8 y verificación de inicio de FastAPI.
- Se documentó qué archivos se versionan en el repositorio y cuáles no (datos clínicos reales).

#### Etapa: Data Input
- Se agregó la justificación del formato CSV frente a alternativas (bases de datos relacionales, Parquet).
- Se documentó la estructura completa del dataset con todas las columnas esperadas.
- Se agregaron 7 suposiciones explícitas sobre los datos de entrada, incluyendo codificación UTF-8, anonimización y tamaño en memoria.

#### Etapa: Data Validation
- Se agregó la justificación de scripts propios en Python frente a Great Expectations y Pandera.
- Se documentó el comportamiento diferenciado entre errores críticos (detienen el pipeline) y advertencias (umbral del 5% configurable via variable de entorno `VALIDATION_ERROR_THRESHOLD`).
- Se especificó la ruta del reporte de errores: `logs/validation_report.txt`.
- Se documentó el contenido del reporte: regla fallida, columna afectada, número de registros problemáticos y descripción.

#### Etapa: Data Preprocessing
- Se agregó la justificación de cada transformación con la alternativa descartada:
  - Mediana vs media para imputación numérica.
  - Moda para variables categóricas binarias.
  - StandardScaler vs MinMaxScaler vs RobustScaler.
- Se especificó la proporción del split train/test: **80/20 con estratificación**.
- Se declaró explícitamente que no hay conjunto de validación separado en esta versión.
- Se documentó que SMOTE se aplica **después del split** para evitar data leakage.
- Se agregaron 4 suposiciones explícitas incluyendo la serialización del preprocesador junto al modelo.

#### Etapa: Model Iterations
- Se agregó la justificación de Random Forest frente a XGBoost.
- Se agregó la justificación de SMOTE frente a undersampling y uso exclusivo de `class_weight`.
- Se agregó la justificación de MLflow frente a Weights & Biases, DVC y Neptune.ai.
- Se documentó que MLflow corre localmente en Docker disponible en `http://localhost:5000`.
- Se especificaron los hiperparámetros iniciales del modelo: `n_estimators=100`, `max_depth=None`, `min_samples_split=2`, `class_weight='balanced'`, `random_state=42`.
- Se documentó qué registra MLflow por experimento: 8 elementos incluyendo hash del commit y artefacto del modelo.
- Se declaró que no se realiza búsqueda automática de hiperparámetros en esta versión.

#### Etapa: Model Selection y Evaluation
- Se documentó la justificación clínica del umbral recall ≥ 0.80 (costo asimétrico del falso negativo).
- Se especificaron 6 criterios de aprobación del modelo.
- Se documentó el proceso de aprobación manual: revisión en MLflow UI, tag `status=approved` y `stage=Production`.

#### Etapa: Model Deployment
- Se agregó la justificación de FastAPI frente a Flask, Django REST Framework y Streamlit.
- Se agregó la justificación de Docker frente a entornos virtuales y ejecutables empaquetados.
- Se documentó **cómo se carga el modelo en el contenedor**: mediante `COPY` en el Dockerfile con ruta configurada por variable de entorno `MODEL_PATH`.
- Se documentó el comportamiento si el modelo no existe al iniciar: la API lanza error explícito y no levanta el servicio.
- Se agregó la **persistencia automática de predicciones** en SQLite (`data/predictions_log.db`) con timestamp, variables de entrada, clase predicha, probabilidad y versión del modelo.
- Se documentó la configuración del servidor: Uvicorn con 1 worker, puerto 8000, sin autenticación.
- Se declaró explícitamente la suposición de endpoints sin autenticación y sus implicaciones.
- Se documentó el flujo completo de una predicción remota en 6 pasos.

#### Etapa: Model Monitoring — Técnico
- Se agregó la justificación de Prometheus frente a Grafana Cloud, Datadog y logs en archivos planos.
- Se documentaron 5 métricas expuestas con nombre exacto y descripción.
- Se definieron 4 reglas de alerta con umbrales numéricos explícitos.
- Se declaró que las alertas son visibles en la UI de Prometheus y que no hay notificaciones externas en esta versión.

#### Etapa: Model Monitoring — Drift
- Se agregó la justificación de scripts propios con scipy frente a Evidently AI y WhyLogs.
- Se documentó la **fuente de datos para el monitoreo**: dataset de referencia (CSV del entrenamiento) vs dataset actual (leído desde `predictions_log.db`).
- Se definió el **mínimo de 30 registros** requeridos para que las pruebas estadísticas sean válidas.
- Se documentó la **frecuencia de ejecución**: semanal el primer mes, quincenal después, inmediata ante concentración anómala de predicciones.
- Se definió la **acción concreta** ante drift detectado: reporte en `logs/drift_report.txt`, revisión por el equipo, reentrenamiento si afecta variables críticas.

#### Estructura del repositorio
- Se agregaron las carpetas y archivos nuevos: `models/`, `logs/`, `.env.example`.
- Se documentó qué archivos se generan automáticamente en ejecución: `predictions_log.db`, `validation_report.txt`, `drift_report.txt`.

#### Cómo ejecutar el sistema
- Se agregaron los requisitos previos: Docker Desktop y Docker Compose v2.
- Se documentó el paso de copiar variables de entorno (`cp .env.example .env`).
- Se agregaron los 4 servicios disponibles con sus URLs: API, Swagger UI, MLflow y Prometheus.
- Se agregaron comandos para ejecutar pruebas, validación de datos y drift manualmente.

---

### MODIFICADO

#### Variables de entrada
- **Antes:** 5 variables (`edad`, `fiebre`, `dolor`, `duracion_dias`, `condicion_cronica`)
- **Después:** 12 variables clínicas completas con nombres estandarizados en inglés técnico (`temperatura`, `frecuencia_cardiaca`, `presion_sistolica`, `nivel_dolor`, etc.)
- **Razón:** El modelo original usaba reglas simuladas; el modelo nuevo usa un clasificador entrenado con variables clínicas reales.

#### Puerto de la API
- **Antes:** puerto `5000`
- **Después:** puerto `8000` para la API de predicción; puerto `5000` para la UI de MLflow
- **Razón:** El puerto 5000 se reserva para MLflow, que es el estándar de esa herramienta. FastAPI usa 8000 por convención.

#### Endpoint de predicción
- **Antes:** `POST /predecir`
- **Después:** `POST /predict`
- **Razón:** Estandarización al inglés técnico, consistente con el resto de los endpoints (`/health`, `/metrics`).

#### Respuesta del endpoint de predicción
- **Antes:** devolvía `estado` y `entrada` (echo de los datos recibidos)
- **Después:** devuelve `predicted_class`, `probability`, `model_version` y `message`
- **Razón:** La respuesta ahora incluye la probabilidad de la predicción y la versión del modelo, información necesaria para la trazabilidad clínica y el monitoreo.

#### Construcción y ejecución Docker
- **Antes:** `docker build` + `docker run` como comandos separados
- **Después:** `docker-compose up --build` como comando único que levanta API, MLflow y Prometheus simultáneamente
- **Razón:** El sistema ahora requiere múltiples contenedores orquestados.

---

### ELIMINADO

#### Predicción simulada con reglas fijas
- **Antes:** el sistema usaba reglas condicionales programadas manualmente para asignar una clase (`if fiebre > 38 and dolor > 5 → ENFERMEDAD_AGUDA`).
- **Después:** el sistema usa un modelo de machine learning entrenado (Random Forest Classifier) con datos clínicos reales o simulados, registrado y versionado en MLflow.
- **Razón:** El objetivo del proyecto es un pipeline de MLOps real, no una simulación de reglas.

---

## [1.0.0] — Versión original

Microservicio de predicción simulada con Flask, reglas condicionales fijas, interfaz web y un único contenedor Docker en puerto 5000. Sin modelo de machine learning, sin pipeline de entrenamiento, sin monitoreo y sin versionamiento de modelos.
