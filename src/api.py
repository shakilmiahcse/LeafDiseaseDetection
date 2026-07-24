"""
REST API endpoints for Leaf Disease Detection.
"""
import json
from pathlib import Path
from datetime import datetime
from functools import wraps

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from uuid import uuid4

from src.logger import setup_logger
from src.predict import predict_leaf, load_model_from_path
from src.schemas import PredictionResult, ImageUploadRequest, HealthCheckResponse

api_bp = Blueprint("api", __name__, url_prefix="/api/v1")
logger = setup_logger(__name__)


def require_model(f):
    """Decorator to ensure model is loaded."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            model = current_app.config.get("_model_instance")
            if model is None:
                return jsonify({"error": "Model not loaded"}), 503
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in require_model decorator: {e}")
            return jsonify({"error": "Internal server error"}), 500
    return decorated_function


def get_model():
    """Get or load the model."""
    config = current_app.config
    if "_model_instance" not in config or config["_model_instance"] is None:
        try:
            from src.config import Config
            config["_model_instance"] = load_model_from_path(Config.get_model_path())
            logger.info("Model loaded for API")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    return config["_model_instance"]


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    try:
        config = current_app.config
        metadata = config.get("_metadata", {})
        model_loaded = "_model_instance" in config and config["_model_instance"] is not None
        metadata_loaded = bool(metadata)
        
        response = HealthCheckResponse(
            status="healthy" if (model_loaded and metadata_loaded) else "unhealthy",
            model_loaded=model_loaded,
            metadata_loaded=metadata_loaded,
            timestamp=datetime.utcnow().isoformat()
        )
        logger.debug("Health check requested")
        return jsonify(response.dict()), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"error": "Health check failed"}), 500


@api_bp.route("/predict", methods=["POST"])
@require_model
def predict():
    """Predict disease from uploaded image."""
    try:
        if "image" not in request.files:
            logger.warning("POST /api/v1/predict without image file")
            return jsonify({"error": "No image file provided"}), 400
        
        image_file = request.files["image"]
        if image_file.filename == "":
            logger.warning("POST /api/v1/predict with empty filename")
            return jsonify({"error": "No file selected"}), 400
        
        # Validate file
        config = current_app.config
        if not self._allowed_file(image_file.filename, config):
            logger.warning(f"Invalid file type: {image_file.filename}")
            return jsonify({"error": "Invalid file type"}), 400
        
        # Check file size
        image_file.seek(0, 2)
        file_size = image_file.tell()
        image_file.seek(0)
        
        if file_size > config.get("MAX_CONTENT_LENGTH", 8 * 1024 * 1024):
            logger.warning(f"File too large: {file_size} bytes")
            return jsonify({"error": "File too large"}), 413
        
        # Save file
        filename = secure_filename(image_file.filename)
        saved_name = f"{uuid4().hex}_{filename}"
        saved_path = config.get("UPLOAD_FOLDER", Path("static/uploads")) / saved_name
        saved_path.parent.mkdir(parents=True, exist_ok=True)
        image_file.save(saved_path)
        logger.info(f"Image saved for prediction: {saved_name}")
        
        # Run prediction
        metadata = current_app.config.get("_metadata", {})
        result = predict_leaf(saved_path, get_model(), metadata)
        logger.info(f"Prediction completed: {result.get('status')}")
        
        # Validate result schema
        prediction = PredictionResult(**result)
        return jsonify(prediction.dict()), 200
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return jsonify({"error": "Prediction failed", "message": str(e)}), 500


@api_bp.route("/model-info", methods=["GET"])
@require_model
def model_info():
    """Get model information and supported classes."""
    try:
        config = current_app.config
        metadata = config.get("_metadata", {})
        class_names = metadata.get("class_names", [])
        
        # Extract crops and diseases
        crops = set()
        diseases_by_crop = {}
        
        for class_name in class_names:
            if "___" in class_name:
                crop, disease = class_name.split("___", 1)
                crops.add(crop)
                if crop not in diseases_by_crop:
                    diseases_by_crop[crop] = []
                diseases_by_crop[crop].append(disease)
        
        info = {
            "model_path": str(config.get("_model_path", "unknown")),
            "metadata_path": str(config.get("_metadata_path", "unknown")),
            "classes_count": len(class_names),
            "crops": sorted(crops),
            "supported_diseases": diseases_by_crop,
            "min_confidence": metadata.get("min_confidence", 0.7),
            "min_confidence_margin": metadata.get("min_confidence_margin", 0.1),
        }
        
        logger.info("Model info requested")
        return jsonify(info), 200
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        return jsonify({"error": "Failed to get model info"}), 500


@api_bp.route("/predict-batch", methods=["POST"])
@require_model
def predict_batch():
    """Predict diseases from multiple uploaded images."""
    try:
        if "images" not in request.files:
            return jsonify({"error": "No images provided"}), 400
        
        files = request.files.getlist("images")
        if not files:
            return jsonify({"error": "No files selected"}), 400
        
        results = []
        config = current_app.config
        metadata = config.get("_metadata", {})
        
        for image_file in files:
            try:
                if image_file.filename == "":
                    continue
                
                if not self._allowed_file(image_file.filename, config):
                    results.append({
                        "filename": image_file.filename,
                        "error": "Invalid file type"
                    })
                    continue
                
                filename = secure_filename(image_file.filename)
                saved_name = f"{uuid4().hex}_{filename}"
                saved_path = config.get("UPLOAD_FOLDER", Path("static/uploads")) / saved_name
                saved_path.parent.mkdir(parents=True, exist_ok=True)
                image_file.save(saved_path)
                
                result = predict_leaf(saved_path, get_model(), metadata)
                prediction = PredictionResult(**result)
                
                results.append({
                    "filename": image_file.filename,
                    "prediction": prediction.dict()
                })
                
            except Exception as e:
                logger.error(f"Batch prediction failed for {image_file.filename}: {e}")
                results.append({
                    "filename": image_file.filename,
                    "error": str(e)
                })
        
        logger.info(f"Batch prediction completed: {len(results)} images")
        return jsonify({"results": results}), 200
        
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}", exc_info=True)
        return jsonify({"error": "Batch prediction failed"}), 500


@staticmethod
def _allowed_file(filename, config):
    """Check if file extension is allowed."""
    allowed = config.get("ALLOWED_EXTENSIONS", {"jpg", "jpeg", "png", "webp"})
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


# Register error handlers
@api_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({"error": "Endpoint not found"}), 404


@api_bp.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({"error": "Method not allowed"}), 405


@api_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500
