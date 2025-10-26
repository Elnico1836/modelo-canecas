// predict.js
const tf = require('@tensorflow/tfjs-node'); // Importa la versión Node.js
const path = require('path');

// Define las clases de tu caneca verde
const CLASES = [
    "cascara de platano", 
    "restos de frutas", 
    // ... otras clases
    "residuos biodegradables"
];

// Variable global para cargar el modelo una sola vez (para reutilización en el servidor)
let model = null;

exports.handler = async (event, context) => {
    // 1. Carga el modelo una vez
    if (!model) {
        console.log("Cargando modelo por primera vez...");
        const modelPath = path.join(process.cwd(), 'model', 'model.json');
        try {
            // El prefijo 'file://' es CRÍTICO para cargar archivos locales en Node.js
            model = await tf.loadLayersModel(`file://${modelPath}`);
            console.log("Modelo cargado exitosamente.");
        } catch (error) {
            console.error("Error al cargar el modelo:", error);
            return {
                statusCode: 500,
                body: JSON.stringify({ error: "No se pudo cargar el modelo en el servidor." }),
            };
        }
    }

    // La función necesita un método HTTP POST para recibir la imagen
    if (event.httpMethod !== 'POST') {
        return { statusCode: 405, body: "Método no permitido" };
    }

    let imagen_data;
    try {
        // 2. Obtener la imagen Base64 del cuerpo de la petición del ESP32
        const body = JSON.parse(event.body);
        imagen_data = body.imagen;
    } catch (e) {
        return { statusCode: 400, body: JSON.stringify({ error: "Faltan datos de imagen." }) };
    }

    try {
        // 3. Decodificación y Preprocesamiento
        const imageBuffer = Buffer.from(imagen_data, 'base64');
        const imageTensor = tf.node.decodeImage(imageBuffer);
        
        // **Ajusta estas líneas a las dimensiones que espera tu modelo**
        const processedImage = tf.image
            .resizeBilinear(imageTensor, [224, 224]) // Redimensiona
            .toFloat()
            .div(tf.scalar(255)) // Normaliza (si aplica)
            .expandDims(0); // Agrega la dimensión del lote

        // 4. Predicción
        const predictions = model.predict(processedImage);
        const predictionArray = await predictions.data();
        
        // Encuentra el índice con la mayor confianza
        const maxIndex = predictionArray.indexOf(Math.max(...predictionArray));

        // 5. Devolver la respuesta
        return {
            statusCode: 200,
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                status: "success",
                clase_predicha: CLASES[maxIndex],
                confianza: predictionArray[maxIndex]
            }),
        };

    } catch (e) {
        console.error("Error durante la inferencia:", e);
        return { statusCode: 500, body: JSON.stringify({ error: `Error de inferencia: ${e.message}` }) };
    }
};
