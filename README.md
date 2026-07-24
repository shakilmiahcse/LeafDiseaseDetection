# Multi-Crop Leaf Disease Detection

![Python](https://img.shields.io/badge/python-3.11+-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.21+-orange)
![Flask](https://img.shields.io/badge/flask-3.1+-green)
![Docker](https://img.shields.io/badge/docker-ready-blue)

TensorFlow + Flask image classification project for crop leaf disease detection and management advice in Bengali. The application supports multi-crop detection and provides REST API endpoints for integration.

## 🌟 Features

- **Multi-Crop Support**: Detect diseases across multiple crop types
- **Bangla Interface & Output**: Full Bengali language support for user-friendly experience
- **REST API**: JSON API endpoints for programmatic access
- **Web Interface**: Upload images and get disease predictions with management advice
- **Batch Processing**: Process multiple images in one request
- **Health Checks**: Built-in health check endpoints for monitoring
- **Comprehensive Logging**: Detailed logging for debugging and monitoring
- **Docker Ready**: Multi-stage Docker build for production deployment
- **Error Handling**: Robust error handling throughout the application
- **Input Validation**: Pydantic schemas for strict input validation

## 📋 System Architecture

```
├── app.py                 # Flask application entry point
├── src/
│   ├── api.py            # REST API blueprints
│   ├── config.py         # Configuration management
│   ├── disease_advice.py # Disease information & translations
│   ├── logger.py         # Logging configuration
│   ├── predict.py        # Prediction logic & image preprocessing
│   ├── prepare_dataset.py # Dataset preparation utilities
│   ├── schemas.py        # Pydantic validation schemas
│   └── train.py          # Training pipeline
├── templates/
│   └── index.html        # Web UI template
├── static/
│   └── styles.css        # Web UI styling
├── models/
│   ├── class_names.json  # Model metadata
│   └── leaf_disease_model.keras
├── Dockerfile            # Production Docker image
└── docker-compose.yml    # Local development setup
```

## 🚀 Quick Start

### Web Interface

1. **Create virtual environment:**
```bash
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run development server:**
```bash
python app.py
```

4. **Access application:**
   - Web UI: http://localhost:5000
   - Health Check: http://localhost:5000/api/v1/health
   - API Docs: See [API.md](API.md)

### REST API

Predict disease from an image:
```bash
curl -X POST -F "image=@leaf_sample.jpg" \
  http://localhost:5000/api/v1/predict
```

Response:
```json
{
  "status": "ok",
  "class_name": "Tomato___Leaf_Mold",
  "confidence": 0.95,
  "crop_bn": "টমেটো",
  "disease_bn": "ছাঁচ রোগ",
  "name_bn": "টমেটো - ছাঁচ রোগ",
  "solution_bn": "আক্রান্ত পাতা তুলে নষ্ট করুন..."
}
```

See [API.md](API.md) for comprehensive API documentation.

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)
```bash
docker-compose up -d
```

Access the application at: http://localhost:5000

### Build Docker Image
```bash
docker build -t leaf-disease-detection:latest .
```

### Run Docker Container
```bash
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -e FLASK_ENV=production \
  leaf-disease-detection:latest
```

## 📚 Comprehensive Documentation

- **[API.md](API.md)** - Complete REST API documentation with examples
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Deployment guides for various platforms (AWS, Google Cloud, etc.)

## 🎯 Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Key settings:
```
FLASK_ENV=production
LEAF_MODEL_PATH=models/leaf_disease_model.keras
MIN_CONFIDENCE=0.7
MIN_CONFIDENCE_MARGIN=0.1
LOG_LEVEL=INFO
```

See [.env.example](.env.example) for all available options.

## 📊 Dataset Preparation

### Dataset Structure
```
dataset/
  train/
    Apple___Apple_scab/
    Tomato___Healthy/
    Tomato___Leaf_Mold/
    Not_Leaf/
  valid/
    Apple___Apple_scab/
    Tomato___Healthy/
    Tomato___Leaf_Mold/
    Not_Leaf/
```

### Prepare Dataset from PlantVillage
```bash
# Dry run (preview changes)
python src/prepare_dataset.py \
  --source path/to/raw/images \
  --output dataset \
  --valid-ratio 0.2 \
  --dry-run

# Apply changes
python src/prepare_dataset.py \
  --source path/to/raw/images \
  --output dataset \
  --valid-ratio 0.2

# Select specific classes only
python src/prepare_dataset.py \
  --source path/to/all_classes \
  --classes Potato___Late_blight Tomato___Healthy \
  --output dataset
```

## 🏋️ Model Training

### Train New Model
```bash
python src/train.py \
  --data-dir dataset \
  --model-out models/leaf_disease_model.keras \
  --epochs 20 \
  --batch-size 32 \
  --img-size 224 \
  --base-weights imagenet
```

### Training Parameters
| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data-dir` | dataset | Dataset root directory |
| `--epochs` | 10 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--img-size` | 224 | Input image size |
| `--base-weights` | imagenet | Pretrained weights (imagenet/none) |
| `--min-confidence` | 0.7 | Confidence threshold |

## 📈 Supported Crops & Diseases

Currently trained on:
- **Tomato**: Healthy, Leaf_Mold

To add more crops:
1. Add disease images organized in class folders
2. Retrain the model with new dataset
3. Metadata updates automatically

## 🔍 Monitoring & Logging

### View Logs
```bash
# Development
tail -f logs/app_*.log

# Docker
docker-compose logs -f leaf-disease-api
```

### Health Check
```bash
curl http://localhost:5000/api/v1/health
```

### Model Information
```bash
curl http://localhost:5000/api/v1/model-info
```

## 🛠️ Troubleshooting

### Model Not Loading
```
Error: Missing metadata file: models/class_names.json
```
**Solution**: Train model or set `LEAF_MODEL_PATH` to existing model

### Port Already in Use
```
Error: Address already in use
```
**Solution**: Change port in config or kill process using port 5000

### Out of Memory
```
Error: CUDA out of memory
```
**Solution**: Reduce batch size or use CPU (`CUDA_VISIBLE_DEVICES=-1`)

### Docker Build Issues
Ensure Docker is running and try:
```bash
docker-compose down
docker system prune -a
docker-compose up --build
```

## 🔐 Production Security

1. Change `SECRET_KEY` in `.env`
2. Use HTTPS with SSL certificates
3. Implement authentication/rate limiting
4. Run with reverse proxy (Nginx)
5. Keep dependencies updated
6. Use secrets management for credentials

See [DEPLOYMENT.md](DEPLOYMENT.md) for production setup.

## 📦 Development

### Install Development Dependencies
```bash
pip install -r requirements.txt pytest pytest-cov
```

### Running Tests (Future)
```bash
pytest tests/
```

### Code Style
```bash
pip install black flake8
black src/ app.py
flake8 src/ app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- PlantVillage dataset for training data
- TensorFlow team for deep learning framework
- Flask community for web framework

## 📞 Support

For issues, questions, or suggestions:
- Open GitHub issue
- Check [API.md](API.md) for API help
- Review [DEPLOYMENT.md](DEPLOYMENT.md) for deployment issues

---

**Note**: This model can only recognize classes it was trained on. To detect many crops, add a comprehensive multi-crop leaf dataset and retrain the model.

Last Updated: 2024-07-24

- virtual environments (`.venv/`, `venv/`, `env/`)
- Python caches (`__pycache__/`, `*.pyc`)
- model binary artifacts (`models/*.keras`, `models/*.h5`)
- generated uploads (`static/uploads/`)
- large dataset folders (`dataset/`, `_plantvillage_full/`, `_plantvillage_src/`)
- output and log files (`outputs/`, `logs/`)

Keep only source code and metadata (`models/class_names.json`) in Git. Large model or dataset files should be stored outside the repository or using Git LFS if needed.

## Train

```powershell
python src\train.py --epochs 20 --batch-size 32
```

MobileNetV2 uses ImageNet weights by default. If you are offline, run:

```powershell
python src\train.py --epochs 20 --base-weights none
```

The script saves:

```text
models/leaf_disease_model.keras
models/class_names.json
outputs/training_history.png
```

The metadata also stores class count, supported crops, image size, and prediction confidence thresholds.

## Predict

```powershell
python src\predict.py path\to\leaf.jpg
```

## Web App

```powershell
python app.py
```

Open `http://127.0.0.1:5000`, upload a crop leaf image, and the app shows the predicted disease with Bangla advice.

If you want to use a different model file:

```powershell
$env:LEAF_MODEL_PATH="models\leaf_disease_model.keras"
python app.py
```

## Advice Mapping

Bangla crop/disease names and generic advice live in:

```text
src/disease_advice.py
```

When you add a new class folder, the app will still generate a readable result from the class name. For better Bangla names, add the crop or disease term to `CROP_NAMES_BN` or `DISEASE_NAMES_BN`.
