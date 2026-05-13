# Servicio Docker para predicción simulada de enfermedad

## Descripción

Este proyecto contiene una solución donde el servicio permite que un médico ingrese datos básicos de un paciente y obtenga uno de los siguientes estados:

- NO ENFERMO
- ENFERMEDAD LEVE
- ENFERMEDAD AGUDA
- ENFERMEDAD CRÓNICA




##  Variables de entrada

La función recibe como mínimo cuatro variables:

| Variable | Descripción | Ejemplo |
|---|---|---|
| edad | Edad del paciente | 45 |
| fiebre | Temperatura corporal | 38.5 |
| dolor | Nivel de dolor de 0 a 10 | 7 |
| duracion_dias | Días con síntomas | 3 |
| condicion_cronica | Indica si el paciente tiene antecedente crónico | true / false |

## Estados retornados

El servicio retorna uno de estos estados:

- `NO ENFERMO`
- `ENFERMEDAD LEVE`
- `ENFERMEDAD AGUDA`
- `ENFERMEDAD CRÓNICA`

##  Construcción de la imagen Docker

Desde la carpeta del proyecto, ejecutar:

```bash
docker build -t enfermedad-mlops .
```

##  Ejecución del contenedor

Ejecutar:

```bash
docker run -p 5000:5000 enfermedad-mlops
```

Luego abrir en el navegador:

```text
http://localhost:5000
```

## Uso desde la página web

La aplicación incluye una página sencilla donde el médico puede ingresar:

- Edad.
- Temperatura corporal.
- Nivel de dolor.
- Duración de síntomas.
- Si el paciente tiene o no condición crónica.

Después de presionar el botón, el sistema muestra la predicción simulada.

##  Uso desde API

También se puede consumir mediante una petición POST al endpoint:

```text
http://localhost:5000/predecir
```

Ejemplo usando `curl`:

```bash
curl -X POST http://localhost:5000/predecir \
  -H "Content-Type: application/json" \
  -d "{\"edad\":45,\"fiebre\":38.5,\"dolor\":7,\"duracion_dias\":3,\"condicion_cronica\":false}"
```

Respuesta esperada:

```json
{
  "estado": "ENFERMEDAD AGUDA",
  "entrada": {
    "edad": 45,
    "fiebre": 38.5,
    "dolor": 7,
    "duracion_dias": 3,
    "condicion_cronica": false
  },
  "nota": "Esta es una predicción simulada para fines académicos. No reemplaza el criterio médico."
}
```

## Endpoint de verificación

Para validar que el servicio está funcionando:

```text
http://localhost:5000/health
```

Respuesta:

```json
{
  "status": "ok"
}
```

