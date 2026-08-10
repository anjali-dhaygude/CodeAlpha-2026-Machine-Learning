# ============================================================
# NEURAWRITE AI
# Modern Handwritten Character Recognition Backend
# ============================================================

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import numpy as np

from PIL import Image, ImageOps
import base64
import io
import os
import logging
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

APP_NAME = "NeuraWrite AI"

MODEL_PATH = os.path.join(
    "model",
    "handwritten_model.keras"
)

HOST = "127.0.0.1"
PORT = 5000


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# Allow frontend/API requests
CORS(app)


# Maximum request size: 5 MB
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD AI MODEL
# ============================================================

logger.info("Starting %s...", APP_NAME)

if not os.path.exists(MODEL_PATH):

    logger.error(
        "Model not found: %s",
        MODEL_PATH
    )

    raise FileNotFoundError(
        f"AI model not found: {MODEL_PATH}"
    )


logger.info("Loading CNN model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

logger.info("CNN model loaded successfully.")


# ============================================================
# MODEL INFORMATION
# ============================================================

MODEL_INFO = {
    "name": "Handwritten Digit CNN",
    "framework": "TensorFlow / Keras",
    "input_size": "28x28",
    "classes": 10,
    "classes_name": [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9"
    ],
    "model_file": "handwritten_model.keras"
}


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK API
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():

    return jsonify({

        "status": "online",

        "service": APP_NAME,

        "model": "loaded",

        "timestamp":
            datetime.utcnow().isoformat()

    })


# ============================================================
# MODEL INFORMATION API
# ============================================================

@app.route("/api/model-info", methods=["GET"])
def model_info():

    return jsonify({

        "status": "success",

        "model": MODEL_INFO

    })


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Convert browser image into
    MNIST-style 28x28 image.
    """

    # --------------------------------
    # Convert to grayscale
    # --------------------------------

    image = image.convert("L")


    # --------------------------------
    # Find bounding box
    # --------------------------------

    array = np.array(image)

    # Find non-black pixels
    coords = np.argwhere(array > 20)


    # If image is completely blank
    if coords.size == 0:

        return None


    # --------------------------------
    # Crop around character
    # --------------------------------

    y_min, x_min = coords.min(axis=0)

    y_max, x_max = coords.max(axis=0)


    cropped = image.crop(
        (
            x_min,
            y_min,
            x_max + 1,
            y_max + 1
        )
    )


    # --------------------------------
    # Make square
    # --------------------------------

    width, height = cropped.size

    size = max(
        width,
        height
    )


    square = Image.new(
        "L",
        (size, size),
        0
    )


    x_offset = (
        size - width
    ) // 2

    y_offset = (
        size - height
    ) // 2


    square.paste(
        cropped,
        (x_offset, y_offset)
    )


    # --------------------------------
    # Resize to 20x20
    # --------------------------------

    square = square.resize(
        (20, 20),
        Image.Resampling.LANCZOS
    )


    # --------------------------------
    # Put inside 28x28 canvas
    # --------------------------------

    final_image = Image.new(
        "L",
        (28, 28),
        0
    )


    final_image.paste(
        square,
        (4, 4)
    )


    # --------------------------------
    # Convert to numpy
    # --------------------------------

    image_array = np.array(
        final_image
    )


    # --------------------------------
    # Normalize
    # --------------------------------

    image_array = (
        image_array.astype(
            np.float32
        ) / 255.0
    )


    # --------------------------------
    # CNN input shape
    # --------------------------------

    image_array = image_array.reshape(
        1,
        28,
        28,
        1
    )


    return image_array


# ============================================================
# PREDICTION API
# ============================================================

@app.route(
    "/api/predict",
    methods=["POST"]
)
def predict():

    try:

        # --------------------------------
        # Check JSON
        # --------------------------------

        if not request.is_json:

            return jsonify({

                "status": "error",

                "message":
                    "Request must contain JSON data."

            }), 400


        data = request.get_json()


        # --------------------------------
        # Check image
        # --------------------------------

        if "image" not in data:

            return jsonify({

                "status": "error",

                "message":
                    "Image data is required."

            }), 400


        image_data = data["image"]


        # --------------------------------
        # Validate image string
        # --------------------------------

        if not isinstance(
            image_data,
            str
        ):

            return jsonify({

                "status": "error",

                "message":
                    "Invalid image format."

            }), 400


        # --------------------------------
        # Remove Base64 header
        # --------------------------------

        if "," in image_data:

            image_data = (
                image_data.split(
                    ",",
                    1
                )[1]
            )


        # --------------------------------
        # Decode image
        # --------------------------------

        image_bytes = base64.b64decode(
            image_data
        )


        # --------------------------------
        # Open image
        # --------------------------------

        image = Image.open(
            io.BytesIO(image_bytes)
        )


        # --------------------------------
        # Preprocess
        # --------------------------------

        processed = preprocess_image(
            image
        )


        # --------------------------------
        # Blank drawing
        # --------------------------------

        if processed is None:

            return jsonify({

                "status": "success",

                "prediction": None,

                "message":
                    "Please draw a digit first."

            })


        # --------------------------------
        # AI prediction
        # --------------------------------

        predictions = model.predict(
            processed,
            verbose=0
        )[0]


        # --------------------------------
        # Get predicted digit
        # --------------------------------

        digit = int(
            np.argmax(predictions)
        )


        # --------------------------------
        # Confidence
        # --------------------------------

        confidence = float(
            np.max(predictions) * 100
        )


        # --------------------------------
        # All probabilities
        # --------------------------------

        probabilities = {

            str(i):
                round(
                    float(
                        predictions[i]
                        * 100
                    ),
                    2
                )

            for i in range(10)

        }


        # --------------------------------
        # Logging
        # --------------------------------

        logger.info(
            "Prediction: %s | Confidence: %.2f%%",
            digit,
            confidence
        )


        # --------------------------------
        # Response
        # --------------------------------

        return jsonify({

            "status": "success",

            "prediction": {

                "digit": digit,

                "confidence":
                    round(
                        confidence,
                        2
                    ),

                "probabilities":
                    probabilities

            },

            "timestamp":
                datetime.utcnow().isoformat()

        })


    except Exception as error:

        logger.exception(
            "Prediction failed"
        )


        return jsonify({

            "status": "error",

            "message":
                "Unable to process the image.",

            "details":
                str(error)

        }), 500


# ============================================================
# API STATUS
# ============================================================

@app.route("/api/status", methods=["GET"])
def status():

    return jsonify({

        "application":
            APP_NAME,

        "status":
            "running",

        "ai_model":
            "CNN",

        "model_status":
            "ready",

        "api_version":
            "1.0",

        "server_time":
            datetime.utcnow().isoformat()

    })


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def not_found(error):

    return jsonify({

        "status": "error",

        "message":
            "API endpoint not found."

    }), 404


@app.errorhandler(413)
def too_large(error):

    return jsonify({

        "status": "error",

        "message":
            "Image is too large."

    }), 413


@app.errorhandler(500)
def internal_error(error):

    return jsonify({

        "status": "error",

        "message":
            "Internal server error."

    }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    logger.info(
        "=========================================="
    )

    logger.info(
        "        %s",
        APP_NAME
    )

    logger.info(
        "        AI Backend Starting"
    )

    logger.info(
        "=========================================="
    )

    logger.info(
        "Model: %s",
        MODEL_PATH
    )

    logger.info(
        "Server: http://%s:%s",
        HOST,
        PORT
    )


    app.run(

        host=HOST,

        port=PORT,

        debug=True

    )