from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

ESTADOS = [
    "NO ENFERMO",
    "ENFERMEDAD LEVE",
    "ENFERMEDAD AGUDA",
    "ENFERMEDAD CRÓNICA"
]


def simular_prediccion(edad, fiebre, dolor, duracion_dias, condicion_cronica=False):
    """
    Función simulada para representar el comportamiento de un modelo de ML.

    Parámetros:
    - edad: edad del paciente en años.
    - fiebre: temperatura corporal en grados Celsius.
    - dolor: escala de dolor de 0 a 10.
    - duracion_dias: número de días con síntomas.
    - condicion_cronica: indica si el paciente tiene antecedente crónico.

    Retorna uno de los estados solicitados:
    - NO ENFERMO
    - ENFERMEDAD LEVE
    - ENFERMEDAD AGUDA
    - ENFERMEDAD CRÓNICA
    """

    score = 0

    if fiebre >= 38:
        score += 2
    elif fiebre >= 37.5:
        score += 1

    if dolor >= 8:
        score += 3
    elif dolor >= 5:
        score += 2
    elif dolor >= 2:
        score += 1

    if duracion_dias >= 30:
        score += 3
    elif duracion_dias >= 7:
        score += 2
    elif duracion_dias >= 2:
        score += 1

    if edad >= 65:
        score += 1

    if condicion_cronica:
        score += 3

    if condicion_cronica and duracion_dias >= 14:
        return "ENFERMEDAD CRÓNICA"

    if score <= 1:
        return "NO ENFERMO"
    elif score <= 4:
        return "ENFERMEDAD LEVE"
    elif score <= 7:
        return "ENFERMEDAD AGUDA"
    else:
        return "ENFERMEDAD CRÓNICA"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", resultado=None)


@app.route("/predecir", methods=["POST"])
def predecir():
    """
    Punto final que permite obtener una predicción.

    Puede recibir datos de dos formas:
    1. JSON, por ejemplo:
       {
         "edad": 45,
         "fiebre": 38.5,
         "dolor": 7,
         "duracion_dias": 3,
         "condicion_cronica": false
       }

    2. Formulario HTML desde la página principal.
    """

    try:
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        edad = int(data.get("edad"))
        fiebre = float(data.get("fiebre"))
        dolor = int(data.get("dolor"))
        duracion_dias = int(data.get("duracion_dias"))
        condicion_cronica = str(data.get("condicion_cronica", "false")).lower() in ["true", "1", "si", "sí", "on"]

        resultado = simular_prediccion(
            edad=edad,
            fiebre=fiebre,
            dolor=dolor,
            duracion_dias=duracion_dias,
            condicion_cronica=condicion_cronica
        )

        response = {
            "estado": resultado,
            "entrada": {
                "edad": edad,
                "fiebre": fiebre,
                "dolor": dolor,
                "duracion_dias": duracion_dias,
                "condicion_cronica": condicion_cronica
            },
            "nota": "Esta es una predicción simulada para fines académicos. No reemplaza el criterio médico."
        }

        if request.is_json:
            return jsonify(response)

        return render_template("index.html", resultado=response)

    except Exception as e:
        error = {
            "error": "Datos de entrada inválidos",
            "detalle": str(e),
            "campos_requeridos": ["edad", "fiebre", "dolor", "duracion_dias"]
        }

        if request.is_json:
            return jsonify(error), 400

        return render_template("index.html", resultado=error), 400


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
