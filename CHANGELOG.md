# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-07-24

### Added

#### Core Features
- **REST API Endpoints**: Complete API with `/api/v1/predict`, `/api/v1/predict-batch`, `/api/v1/health`, `/api/v1/model-info`
- **Logging System**: Comprehensive logging with rotating file handlers and console output
- **Configuration Management**: Centralized config module with development/production/testing environments
- **Input Validation**: Pydantic schemas for strict input validation and type checking
- **Docker Support**: Multi-stage Dockerfile and docker-compose for easy deployment
- **Error Handling**: Robust error handling with detailed error responses
- **Health Checks**: Built-in health check endpoints for monitoring

#### Documentation
- **API Documentation** (`API.md`): Complete REST API documentation with examples
- **Deployment Guide** (`DEPLOYMENT.md`): Setup instructions for multiple platforms (Docker, AWS, Google Cloud, etc.)
- **Updated README**: Comprehensive documentation with features and usage examples
- **Contributing Guide** (`CONTRIBUTING.md`): Guidelines for contributors

#### Development Tools
- **Unit Tests**: Comprehensive test suite using pytest
- **Makefile**: Development commands for common tasks (test, lint, format, docker, etc.)
- **Test Configuration**: pytest.ini with test markers and coverage settings
- **CI/CD Ready**: Structure supports GitHub Actions and other CI/CD platforms

#### Production Deployment
- **Gunicorn WSGI**: Production entry point (`wsgi.py`) for gunicorn
- **Docker Compose**: Local development setup with volume mounts
- `.dockerignore`: Efficient Docker builds
- `.env.example`: Environment variable configuration template

### Changed

#### Refactoring
- Moved hardcoded values to `Config` classes
- Enhanced `train.py` with complete training pipeline
- Added logging throughout the application
- Improved error messages for better debugging
- Restructured app initialization for better modularity

#### Improvements
- Better model path resolution with fallbacks
- More informative console and file logging
- Enhanced metadata handling
- Improved image preprocessing documentation
- Better separation of concerns with API blueprint

### Fixed

- Model loading error handling
- File upload validation
- Metadata loading robustness
- Image preprocessing error messages

### Security

- Input validation with Pydantic
- File type and size restrictions
- Flask configuration for production
- Secure file upload handling

## [1.0.0] - 2024-01-01

### Initial Release

#### Features
- Multi-crop leaf disease detection using TensorFlow
- Web UI for image upload and prediction
- Bengali language support
- Dataset preparation utilities
- Model training pipeline
- Disease management advice in Bengali

#### Documentation
- Basic README with quick start
- Dataset layout documentation
- Dataset preparation instructions

---

## Upgrade Guide

### From v1.0.0 to v2.0.0

No breaking changes to the core prediction functionality. However, several improvements require attention:

1. **Environment Variables**: Create `.env` file from `.env.example`
2. **Dependencies**: Run `pip install -r requirements.txt` for new dependencies (pydantic, pytest, etc.)
3. **Docker**: Use new `docker-compose.yml` for easier local development
4. **Logging**: Logs now saved to `logs/` directory (automatically created)
5. **Configuration**: Can now use `FLASK_ENV` to switch configurations

All existing models are compatible. No retraining needed.

---

## Future Plans

### Planned for v2.1.0
- [ ] Rate limiting middleware
- [ ] Database integration for prediction history
- [ ] Advanced model analytics dashboard
- [ ] Batch training support
- [ ] Multi-GPU training support

### Planned for v3.0.0
- [ ] Model versioning and A/B testing
- [ ] Kubernetes deployment templates
- [ ] Enhanced security (JWT authentication)
- [ ] Advanced monitoring and alerting
- [ ] Real-time model performance metrics

### Under Discussion
- [ ] Mobile app integration
- [ ] Federated learning support
- [ ] Computer vision enhancements
- [ ] Multi-language support expansion
- [ ] Edge device deployment (Raspberry Pi)

---

## Migration Notes

### Configuration Changes
The application now supports environment-based configuration. 

**Before:**
- Hardcoded paths in `app.py`
- Configuration scattered throughout code

**After:**
- Centralized in `src/config.py`
- Environment variable support
- Easy switching between dev/prod

### Logging Changes
The application now provides comprehensive logging.

**Before:**
- Limited print statements
- No file logging
- No structured logging

**After:**
- File and console logging
- Rotating log files
- Structured logging format
- Configurable log levels

### API Changes
New REST API endpoints are available alongside the web UI.

**Before:**
- Web UI only
- Form-based interface

**After:**
- RESTful API with JSON responses
- Programmatic access
- Batch processing support
- Health checks and monitoring

---

## Contributors

Thanks to all contributors and users who have helped improve this project!

## Support

For issues, questions, or suggestions:
- Open GitHub Issue
- Check documentation:
  - [API.md](API.md) - API documentation
  - [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guides
  - [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- Review existing issues and discussions

---

**Last Updated**: 2024-07-24
