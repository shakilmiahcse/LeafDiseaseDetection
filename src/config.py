"""
Configuration management for Leaf Disease Detection application.
"""
import os
from pathlib import Path
from typing import Optional


class Config:
    """Base configuration."""
    
    # Paths
    BASE_DIR = Path(__file__).resolve().parent.parent
    MODELS_DIR = BASE_DIR / "models"
    STATIC_DIR = BASE_DIR / "static"
    TEMPLATES_DIR = BASE_DIR / "templates"
    LOGS_DIR = BASE_DIR / "logs"
    
    # Model
    DEFAULT_MODEL_NAME = "leaf_disease_model.keras"
    METADATA_FILENAME = "class_names.json"
    
    # Upload
    UPLOAD_FOLDER = STATIC_DIR / "uploads"
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8MB
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    
    # Image Processing
    IMAGE_SIZE = 224
    MIN_GREEN_RATIO = 0.06
    MIN_VEGETATION_RATIO = 0.08
    
    # Prediction Thresholds
    DEFAULT_MIN_CONFIDENCE = 0.7
    DEFAULT_MIN_CONFIDENCE_MARGIN = 0.1
    
    # Flask
    DEBUG = False
    HOST = "127.0.0.1"
    PORT = 5000
    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_model_path(cls) -> Path:
        """Resolve model path from environment or default."""
        env_path = os.environ.get("LEAF_MODEL_PATH")
        if env_path:
            path = Path(env_path)
            return path if path.is_absolute() else cls.BASE_DIR / path
        
        # Check for existing models
        for candidate in (
            cls.MODELS_DIR / "leaf_disease_model.keras",
            cls.MODELS_DIR / "leaf_model.h5",
        ):
            if candidate.exists():
                return candidate
        
        return cls.MODELS_DIR / cls.DEFAULT_MODEL_NAME
    
    @classmethod
    def get_metadata_path(cls) -> Path:
        """Get metadata file path."""
        return cls.MODELS_DIR / cls.METADATA_FILENAME
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create necessary directories."""
        cls.UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.MODELS_DIR.mkdir(parents=True, exist_ok=True)


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = "INFO"
    HOST = "0.0.0.0"
    SECRET_KEY = os.environ.get("SECRET_KEY")
    if not SECRET_KEY:
        raise ValueError("SECRET_KEY environment variable is required in production")


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    TESTING = True
    LOG_LEVEL = "DEBUG"
    # Use in-memory or test database
    UPLOAD_FOLDER = Path("/tmp/test_uploads")


def get_config(config_name: Optional[str] = None) -> Config:
    """
    Get configuration object based on environment or explicit name.
    
    Args:
        config_name: Optional config name ('development', 'production', 'testing')
        
    Returns:
        Config object
    """
    if not config_name:
        config_name = os.environ.get("FLASK_ENV", "development").lower()
    
    configs = {
        "development": DevelopmentConfig,
        "production": ProductionConfig,
        "testing": TestingConfig,
    }
    
    config_class = configs.get(config_name, DevelopmentConfig)
    return config_class()
