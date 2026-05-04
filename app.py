from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from src.predict import load_metadata, predict_leaf, tf


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "leaf_model.h5"
METADATA_PATH = BASE_DIR / "models" / "class_names.json"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
metadata = load_metadata(METADATA_PATH)
model = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model():
    global model
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
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
        elif not allowed_file(image.filename):
            error = "শুধু JPG, JPEG, PNG অথবা WEBP ছবি আপলোড করুন।"
        else:
            filename = secure_filename(image.filename)
            saved_name = f"{uuid4().hex}_{filename}"
            saved_path = app.config["UPLOAD_FOLDER"] / saved_name
            image.save(saved_path)

            result = predict_leaf(saved_path, get_model(), metadata)
            image_url = url_for("static", filename=f"uploads/{saved_name}")

    return render_template("index.html", result=result, image_url=image_url, error=error)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
