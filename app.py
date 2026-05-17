import os
from pathlib import Path
from uuid import uuid4

from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename

from src.disease_advice import summarize_supported_classes
from src.predict import load_metadata, predict_leaf, load_model_from_path


BASE_DIR = Path(__file__).resolve().parent
METADATA_PATH = BASE_DIR / "models" / "class_names.json"
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
metadata = load_metadata(METADATA_PATH)
model_summary = summarize_supported_classes(metadata)
model = None


def resolve_model_path():
    configured_path = os.environ.get("LEAF_MODEL_PATH")
    if configured_path:
        path = Path(configured_path)
        return path if path.is_absolute() else BASE_DIR / path

    for candidate in (
        BASE_DIR / "models" / "leaf_disease_model.keras",
        BASE_DIR / "models" / "leaf_model.h5",
    ):
        if candidate.exists():
            return candidate

    return BASE_DIR / "models" / "leaf_disease_model.keras"


MODEL_PATH = resolve_model_path()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_model():
    global model
    if model is None:
        model = load_model_from_path(MODEL_PATH)
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

    return render_template(
        "index.html",
        result=result,
        image_url=image_url,
        error=error,
        model_summary=model_summary,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
