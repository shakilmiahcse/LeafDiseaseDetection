"""
Unit tests for Leaf Disease Detection application.
"""
import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app import app, config
from src.schemas import ImageUploadRequest, PredictionStatus
from src.config import Config, get_config


class TestConfig:
    """Test configuration module."""
    
    def test_development_config(self):
        """Test development configuration."""
        dev_config = get_config("development")
        assert dev_config.DEBUG is True
        assert dev_config.LOG_LEVEL == "DEBUG"
    
    def test_production_config(self):
        """Test production configuration defaults."""
        prod_config = get_config("production")
        assert prod_config.DEBUG is False
        assert prod_config.LOG_LEVEL == "INFO"
        assert prod_config.HOST == "0.0.0.0"
    
    def test_config_directories(self):
        """Test config creates required directories."""
        config = Config()
        # This should not raise an error
        config.ensure_directories()
        assert config.UPLOAD_FOLDER.exists()
        assert config.LOGS_DIR.exists()
    
    def test_model_path_resolution(self):
        """Test model path resolution."""
        config = Config()
        model_path = config.get_model_path()
        assert isinstance(model_path, Path)
        assert "leaf" in str(model_path).lower()


class TestSchemas:
    """Test Pydantic schemas."""
    
    def test_prediction_status_enum(self):
        """Test prediction status enum values."""
        assert PredictionStatus.OK == "ok"
        assert PredictionStatus.NOT_LEAF == "not_leaf"
        assert PredictionStatus.UNCERTAIN == "uncertain"
    
    def test_image_upload_request_valid(self):
        """Test valid image upload request."""
        request = ImageUploadRequest(
            filename="leaf.jpg",
            file_size=1024 * 100  # 100KB
        )
        assert request.filename == "leaf.jpg"
        assert request.file_size == 102400
    
    def test_image_upload_request_invalid_extension(self):
        """Test image upload request with invalid extension."""
        with pytest.raises(ValueError):
            ImageUploadRequest(
                filename="leaf.txt",
                file_size=1024
            )
    
    def test_image_upload_request_file_too_large(self):
        """Test image upload request with oversized file."""
        with pytest.raises(ValueError):
            ImageUploadRequest(
                filename="leaf.jpg",
                file_size=10 * 1024 * 1024  # 10MB, exceeds 8MB limit
            )


class TestFlaskApp:
    """Test Flask application."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_app_creation(self):
        """Test Flask app is created properly."""
        assert app is not None
        assert app.config['TESTING'] is True
    
    def test_web_ui_loads(self, client):
        """Test web UI home page loads."""
        response = client.get('/')
        assert response.status_code == 200
        assert b'Leaf Disease Detection' in response.data or \
               b'leaf' in response.data.lower()
    
    def test_health_check_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get('/api/v1/health')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'status' in data
        assert data['status'] in ['healthy', 'unhealthy']
    
    def test_model_info_endpoint(self, client):
        """Test model info endpoint."""
        response = client.get('/api/v1/model-info')
        # Should return 200 or 503 depending on model state
        assert response.status_code in [200, 503]
    
    def test_predict_without_image(self, client):
        """Test predict endpoint without image."""
        response = client.post('/api/v1/predict')
        assert response.status_code == 400
        data = json.loads(response.data)
        assert 'error' in data
    
    def test_predict_with_invalid_file_type(self, client):
        """Test predict endpoint with invalid file type."""
        data = {'image': (b'dummy content', 'test.txt')}
        response = client.post(
            '/api/v1/predict',
            data=data,
            content_type='multipart/form-data'
        )
        assert response.status_code in [400, 413]
    
    def test_batch_predict_without_images(self, client):
        """Test batch predict endpoint without images."""
        response = client.post('/api/v1/predict-batch')
        assert response.status_code == 400


class TestErrorHandling:
    """Test error handling."""
    
    @pytest.fixture
    def client(self):
        """Create test client."""
        app.config['TESTING'] = True
        with app.test_client() as client:
            yield client
    
    def test_404_not_found(self, client):
        """Test 404 error handling."""
        response = client.get('/api/v1/nonexistent')
        assert response.status_code == 404
    
    def test_405_method_not_allowed(self, client):
        """Test 405 error handling."""
        response = client.put('/api/v1/health')
        assert response.status_code == 405
    
    def test_invalid_json(self, client):
        """Test invalid JSON handling."""
        response = client.post(
            '/api/v1/predict',
            data='invalid json',
            content_type='application/json'
        )
        # Should return 400 or 415 depending on handling
        assert response.status_code in [400, 415]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
