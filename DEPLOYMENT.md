# Deployment Guide

## Table of Contents
- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Production Deployment](#production-deployment)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#troubleshooting)

## Local Development

### Prerequisites
- Python 3.11+
- pip or conda
- Git

### Setup

1. **Clone the repository:**
```bash
git clone <repository-url>
cd LeafDiseaseDetection
```

2. **Create virtual environment:**
```bash
python -m venv .venv

# On Windows
.\.venv\Scripts\Activate.ps1

# On macOS/Linux
source .venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Prepare dataset (optional):**
```bash
python src/prepare_dataset.py --source <source_path> --output dataset --valid-ratio 0.2
```

5. **Train model (optional):**
```bash
python src/train.py --data-dir dataset --epochs 20 --batch-size 32
```

6. **Run development server:**
```bash
python app.py
```

Access the app at: `http://localhost:5000`

## Docker Deployment

### Prerequisites
- Docker 20.10+
- Docker Compose 2.0+

### Build and Run

1. **Build Docker image:**
```bash
docker build -t leaf-disease-detection:latest .
```

2. **Using Docker Compose (recommended):**
```bash
# Start the service
docker-compose up -d

# View logs
docker-compose logs -f leaf-disease-api

# Stop the service
docker-compose down
```

3. **Using Docker directly:**
```bash
docker run -p 5000:5000 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/static/uploads:/app/static/uploads \
  -e FLASK_ENV=production \
  leaf-disease-detection:latest
```

### Health Check
```bash
curl http://localhost:5000/api/v1/health
```

## Production Deployment

### Using Gunicorn

1. **Install production dependencies:**
```bash
pip install gunicorn
```

2. **Run with Gunicorn:**
```bash
gunicorn --bind 0.0.0.0:5000 \
  --workers 4 \
  --timeout 120 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  app:app
```

### Using Nginx Reverse Proxy

1. **Create Nginx configuration:**
```nginx
upstream leaf_disease_api {
    server 127.0.0.1:5000;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://leaf_disease_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_read_timeout 120;
        proxy_connect_timeout 120;
    }

    location /api/ {
        proxy_pass http://leaf_disease_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

2. **Enable HTTPS with Let's Encrypt:**
```bash
sudo certbot --nginx -d your-domain.com
```

### Cloud Deployment

#### AWS EC2
```bash
# Launch EC2 instance with Ubuntu 22.04
# SSH into instance
ssh -i key.pem ubuntu@instance-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Deploy with Docker Compose
git clone <repository-url>
cd LeafDiseaseDetection
docker-compose up -d
```

#### Google Cloud Run
```bash
# Set project
gcloud config set project YOUR_PROJECT_ID

# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/leaf-disease-detection

# Deploy
gcloud run deploy leaf-disease-detection \
  --image gcr.io/YOUR_PROJECT_ID/leaf-disease-detection \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --timeout 120 \
  --allow-unauthenticated
```

## Environment Variables

Create `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `FLASK_ENV` | Flask environment (development/production) | production |
| `SECRET_KEY` | Flask secret key (change in production!) | dev-secret-key |
| `LEAF_MODEL_PATH` | Path to trained model | models/leaf_disease_model.keras |
| `LOG_LEVEL` | Logging level | INFO |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 5000 |
| `MAX_CONTENT_LENGTH` | Max file upload size | 8388608 (8MB) |
| `MIN_CONFIDENCE` | Model confidence threshold | 0.7 |
| `MIN_CONFIDENCE_MARGIN` | Confidence margin threshold | 0.1 |

## Troubleshooting

### Model Not Found
```
Error: Missing metadata file: models/class_names.json
```
**Solution:** Train the model first or provide a valid model path via `LEAF_MODEL_PATH`

### Out of Memory
```
Error: CUDA out of memory
```
**Solution:** Reduce batch size or use CPU by setting `CUDA_VISIBLE_DEVICES=-1`

### Port Already in Use
```
Error: Address already in use
```
**Solution:** Change port in config or kill process using port 5000

### Docker Build Fails
```
Error during build: "libopencv-dev not found"
```
**Solution:** Ensure Docker daemon is running and dependencies are correctly specified

### Health Check Failing
```
Unhealthy: model_loaded=false
```
**Solution:** Verify model file exists at configured path and check logs

## Monitoring

### Check Application Logs
```bash
# Development
tail -f logs/app_*.log

# Docker
docker-compose logs -f leaf-disease-api
```

### Performance Monitoring
```bash
# Check memory usage
docker stats

# Check CPU usage
top -p $(pgrep -f gunicorn)
```

## Scaling

### Horizontal Scaling
Use multiple worker processes with Gunicorn:
```bash
gunicorn --workers 8 app:app
```

Or use container orchestration (Kubernetes):
```bash
kubectl scale deployment leaf-disease --replicas=3
```

### Caching
Implement Redis caching for predictions:
```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'redis'})
```

## Security

1. **Change SECRET_KEY in production**
2. **Use HTTPS/SSL certificates**
3. **Enable rate limiting**
4. **Validate file uploads**
5. **Run containers as non-root**
6. **Keep dependencies updated**

## Support

For issues or questions, please refer to:
- [API Documentation](API.md)
- [README](README.md)
- GitHub Issues
