import base64
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import tensorflow as tf

# Ruta del modelo
MODEL_PATH = './clasificador_canecas.h5'

# Clases del modelo
CLASES = ['caneca blanca', 'caneca negra', 'caneca verde']

# Cargar el modelo
try:
    print("-------------------------------------------------------")
    print(f"Cargando modelo Keras (.h5) desde: {MODEL_PATH}")
    MODEL = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Modelo .h5 cargado exitosamente.")
    MODEL.summary()
    print("-------------------------------------------------------")
except Exception as e:
    print("-------------------------------------------------------")
    print(f"❌ Error CRÍTICO al cargar el modelo .h5: {e}")
    print("El servidor no puede iniciarse sin el modelo. Verifica el nombre del archivo.")
    print("-------------------------------------------------------")
    MODEL = None

# Crear la app Flask
app = Flask(__name__)
CORS(app)  # Permitir peticiones desde otros orígenes (ESP32 o interfaz web)

@app.route('/predict', methods=['POST'])
def predict():
    """Recibe una imagen en Base64 y devuelve la clase predicha."""
    if MODEL is None:
        return jsonify({"error": "Modelo no cargado. Revisa los logs de Docker para el error de inicio."}), 500

    try:
        # Obtener el JSON del cuerpo de la solicitud
        data = request.get_json()

        if 'imagen' not in data:
            return jsonify({"error": "Falta el campo 'imagen' en la solicitud."}), 400

        # Limpiar la cadena Base64
        imagen_b64 = data['imagen'].replace('\n', '').replace('\r', '').replace(' ', '')

        # Eliminar encabezado MIME si existe
        if ',' in imagen_b64:
            imagen_b64 = imagen_b64.split(',')[1]

        # Decodificar Base64 a bytes
        try:
            imagen_bytes = base64.b64decode(imagen_b64, validate=True)
            img = Image.open(BytesIO(imagen_bytes)).convert('RGB')
        except Exception as e:
            print("❌ Error al decodificar la imagen:", str(e))
            return jsonify({'error': 'No se pudo decodificar la imagen correctamente'}), 400

        # Preprocesamiento de la imagen
        TARGET_SIZE = (224, 224)
        img = img.resize(TARGET_SIZE)
        image_array = tf.keras.utils.img_to_array(img)
        image_array = image_array / 255.0  # Normalización
        input_tensor = np.expand_dims(image_array, axis=0)

        # Predicción
        predictions_array = MODEL.predict(input_tensor, verbose=0)[0]
        predicted_class_index = int(np.argmax(predictions_array))
        predicted_class = CLASES[predicted_class_index]
        confidence = float(predictions_array[predicted_class_index])

        # Construir respuesta
        response = {
            "status": "success",
            "prediccion": [
                {
                    "clase": predicted_class,
                    "confianza": round(confidence, 4)
                }
            ],
            "modelo_formato": "Keras .h5 Nativo"
        }

        print("✅ Predicción realizada:", response)
        return jsonify(response)

    except Exception as e:
        print(f"❌ Error inesperado durante la predicción: {e}")
        return jsonify({"error": f"Error interno del servidor durante la inferencia: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
