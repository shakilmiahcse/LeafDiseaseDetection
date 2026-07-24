"""
Pydantic schemas for input validation and type hints.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class PredictionStatus(str, Enum):
    """Prediction status enum."""
    OK = "ok"
    NOT_LEAF = "not_leaf"
    UNCERTAIN = "uncertain"


class PredictionResult(BaseModel):
    """Prediction result schema."""
    status: PredictionStatus
    class_name: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0.0, le=1.0)
    margin: Optional[float] = Field(None, ge=0.0, le=1.0)
    crop_bn: Optional[str] = None
    disease_bn: Optional[str] = None
    name_bn: str
    solution_bn: str
    visual_stats: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = False


class ImageUploadRequest(BaseModel):
    """Image upload request validation."""
    filename: str = Field(..., min_length=1, max_length=255)
    file_size: int = Field(..., gt=0, le=8 * 1024 * 1024)  # Max 8MB
    
    @validator('filename')
    def validate_extension(cls, v):
        """Validate file extension."""
        allowed_extensions = {"jpg", "jpeg", "png", "webp"}
        extension = v.rsplit(".", 1)[-1].lower() if "." in v else ""
        if extension not in allowed_extensions:
            raise ValueError(f"File extension '{extension}' not allowed. Allowed: {allowed_extensions}")
        return v


class ModelInfo(BaseModel):
    """Model information schema."""
    model_path: str
    metadata_path: str
    classes_count: int
    crops: list = Field(default_factory=list)
    supported_diseases: Dict[str, list] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = False


class HealthCheckResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(..., pattern="^(healthy|unhealthy)$")
    model_loaded: bool
    metadata_loaded: bool
    timestamp: str


class APIErrorResponse(BaseModel):
    """API error response schema."""
    error: str
    message: str
    status_code: int = Field(..., ge=400, le=599)
    timestamp: Optional[str] = None
