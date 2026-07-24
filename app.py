import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from src.disease_advice import summarize_supported_classes
from src.predict import load_metadata, predict_leaf, load_model_from_path
from src.logger import setup_logger
from src.config import get_config


config = get_config()
logger = setup_logger(__name__)

app = Flask(__name__)
app.config.from_object(config)
app.config["UPLOAD_FOLDER"] = config.UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH

config.ensure_directories()

try:
    metadata = load_metadata(config.get_metadata_path())
    logger.info(f"Metadata loaded successfully: {config.get_metadata_path()}")
except Exception as e:
    logger.error(f"Failed to load metadata: {e}")
    metadata = {}

try:
    model_summary = summarize_supported_classes(metadata)
    logger.info(f"Model summary generated: {model_summary}")
except Exception as e:
    logger.error(f"Failed to generate model summary: {e}")
    model_summary = {}

model = None

MODEL_PATH = config.get_model_path()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def get_model():
    global model
    if model is None:
        try:
            logger.info("Loading model...")
            model = load_model_from_path(MODEL_PATH)
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return model


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_url = None
    error = None

    if request.method == "POST":
        image = request.files.get("image")
        if image is None or image.filename == "":
            error = "অনুগ্রহ করে একটি পাতার ছবি নির্বাচন করুন।"
            logger.warning("POST request received without image")
        elif not allowed_file(image.filename):
            error = "শুধু JPG, JPEG, PNG অথবা WEBP ছবি আপলোড করুন।"
            logger.warning(f"Invalid file type uploaded: {image.filename}")
        else:
            try:
                filename = secure_filename(image.filename)
                saved_name = f"{uuid4().hex}_{filename}"
                saved_path = app.config["UPLOAD_FOLDER"] / saved_name
                image.save(saved_path)
                logger.info(f"Image uploaded and saved: {saved_name}")

                result = predict_leaf(saved_path, get_model(), metadata)
                logger.info(f"Prediction completed for {saved_name}: {result.get('status')}")
                image_url = url_for("static", filename=f"uploads/{saved_name}")
            except Exception as e:
                error = "ছবি প্রক্রিয়া করতে একটি ত্রুটি হয়েছে। আবার চেষ্টা করুন।"
                logger.error(f"Error processing image: {e}", exc_info=True)

    return render_template(
        "index.html",
        result=result,
        image_url=image_url,
        error=error,
        model_summary=model_summary,
    )


if __name__ == "__main__":
    logger.info("Starting Leaf Disease Detection application")
    logger.info(f"Model path: {MODEL_PATH}")
    logger.info(f"Upload folder: {UPLOAD_FOLDER}")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
