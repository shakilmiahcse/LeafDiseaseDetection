# Contributing to Leaf Disease Detection

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help each other learn and grow

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork locally:**
```bash
git clone https://github.com/YOUR_USERNAME/LeafDiseaseDetection.git
cd LeafDiseaseDetection
```

3. **Create a virtual environment:**
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
```

4. **Install development dependencies:**
```bash
make dev-install
# or
pip install -r requirements.txt
pip install black flake8 pytest pytest-cov pytest-mock
```

## Development Workflow

1. **Create a feature branch:**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes** and write tests

3. **Run tests locally:**
```bash
make test
```

4. **Check code quality:**
```bash
make lint
make format-check
```

5. **Format your code:**
```bash
make format
```

6. **Commit your changes:**
```bash
git commit -m "feat: Add your feature description

- Detail 1
- Detail 2"
```

7. **Push to your fork:**
```bash
git push origin feature/your-feature-name
```

8. **Open a Pull Request** on GitHub with a clear description

## Code Style

We follow these conventions:

### Python Style
- Follow PEP 8 using `black` for formatting
- Line length: 120 characters
- Use type hints where possible
- Write docstrings for modules, functions, and classes

### Naming Conventions
- `function_names`: snake_case
- `ClassName`: PascalCase  
- `CONSTANT_NAME`: UPPER_SNAKE_CASE
- `_private_method`: underscore prefix for private

### Example:
```python
def process_image(image_path: str) -> dict:
    """
    Process an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Dictionary containing processed image data
        
    Raises:
        FileNotFoundError: If image doesn't exist
    """
    pass
```

## Writing Tests

- Write tests for new features/bug fixes
- Aim for >80% code coverage
- Use descriptive test names
- Group related tests in test classes

Example:
```python
class TestImageProcessing:
    """Tests for image processing module."""
    
    def test_load_image_valid_path(self):
        """Test loading image from valid path."""
        result = load_image("test_image.jpg")
        assert result is not None
    
    def test_load_image_invalid_path(self):
        """Test loading image from invalid path."""
        with pytest.raises(FileNotFoundError):
            load_image("nonexistent.jpg")
```

## Commit Messages

Use clear, descriptive commit messages:

```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (no logic changes)
- `refactor`: Code refactoring
- `perf`: Performance improvements
- `test`: Adding/updating tests

Example:
```
feat: Add batch prediction API endpoint

- Create new POST /api/v1/predict-batch endpoint
- Support multiple image uploads
- Return results for each image
- Add error handling for individual images
```

## Pull Request Process

1. **Update documentation** if needed (README, API.md, etc.)
2. **Add tests** for new features
3. **Run full test suite:**
```bash
make test
make lint
```

4. **Update CHANGELOG** if applicable
5. **Link related issues** in PR description
6. **Keep PR focused** on a single feature/fix
7. **Squash commits** if needed before merging

## Reporting Issues

When reporting bugs:

1. **Check if issue already exists** on GitHub
2. **Provide detailed description:**
   - What you were doing
   - What happened
   - What you expected
   - Steps to reproduce

3. **Include environment details:**
   - OS (Windows/macOS/Linux)
   - Python version
   - Key dependencies versions

Example:
```
**Title:** Model not loading with custom path

**Description:** 
When I set LEAF_MODEL_PATH to a custom location, the app crashes.

**Steps to Reproduce:**
1. Set LEAF_MODEL_PATH=/custom/path/model.keras
2. Run python app.py
3. Access http://localhost:5000

**Error:**
FileNotFoundError: [Errno 2] No such file or directory: '/custom/path/model.keras'

**Expected:** 
App should load and show error message in UI, not crash.

**Environment:**
- OS: Windows 11
- Python: 3.11.0
- TensorFlow: 2.21.0
```

## Feature Requests

When suggesting features:

1. **Check existing issues** first
2. **Be specific** about the use case
3. **Explain benefits** to the project
4. **Provide examples** if possible

Example:
```
**Title:** Add model versioning support

**Description:**
Currently, only one model can be loaded at a time. 
It would be useful to support multiple model versions 
for A/B testing or gradual rollouts.

**Use Case:**
- Test new model versions before deploying to production
- Fallback to previous version if new one performs poorly
- Compare predictions between versions

**Suggested Approach:**
Add version parameter to API endpoints:
GET /api/v1/model-info?version=latest
POST /api/v1/predict?version=v1.0.0
```

## Documentation

Help improve documentation:

1. **Fix typos** and unclear explanations
2. **Add examples** for better clarity
3. **Update outdated** information
4. **Add missing** documentation

Documentation files:
- [README.md](README.md) - Overview and quick start
- [API.md](API.md) - API documentation
- [DEPLOYMENT.md](DEPLOYMENT.md) - Deployment guides
- Code comments - Implementation details

## Project Structure

```
LeafDiseaseDetection/
├── src/
│   ├── api.py           # REST API endpoints
│   ├── config.py        # Configuration management
│   ├── disease_advice.py# Disease info translations
│   ├── logger.py        # Logging setup
│   ├── predict.py       # Prediction logic
│   ├── schemas.py       # Data validation
│   ├── train.py         # Training pipeline
│   └── prepare_dataset.py # Dataset utilities
├── tests/
│   └── test_app.py      # Test suite
├── templates/
│   └── index.html       # Web UI
├── static/
│   └── styles.css       # Web styling
├── models/              # Trained models & metadata
├── app.py               # Flask app entry point
├── wsgi.py              # Production entry point
├── Dockerfile           # Container configuration
├── docker-compose.yml   # Local dev setup
├── requirements.txt     # Dependencies
└── Makefile             # Development commands
```

## Development Tools

### Available Commands
```bash
make help              # Show all available commands
make dev              # Run development server
make test             # Run tests with coverage
make lint             # Check code style
make format           # Auto-format code
make docker-run       # Run in Docker
```

### Recommended Tools
- **IDE**: VS Code, PyCharm
- **Git GUI**: GitHub Desktop, GitKraken
- **Database**: DBeaver (if adding database features)
- **API Testing**: Postman, Insomnia

## Troubleshooting

### Tests Fail Locally
- Ensure all dependencies installed: `pip install -r requirements.txt`
- Clear cache: `make clean`
- Reinstall: `pip install -e .`

### Code Formatting Issues
- Run formatter: `make format`
- Check style: `make lint`

### Git Conflicts
- Pull latest: `git pull origin main`
- Rebase your branch: `git rebase origin/main`

## Questions?

- Check existing issues/discussions
- Review documentation
- Ask in PR comments
- Reach out to maintainers

## Thank You!

We appreciate your contributions to making this project better! 🎉
