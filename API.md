# REST API Documentation

## Base URL
```
http://localhost:5000/api/v1
```

## Authentication
Currently, no authentication is required. For production, implement JWT or API key authentication.

## Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

Check if the application and model are running correctly.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "metadata_loaded": true,
  "timestamp": "2024-07-24T10:30:45.123456"
}
```

**Status Codes:**
- `200 OK` - Service is healthy
- `503 Service Unavailable` - Model or metadata not loaded

**Example:**
```bash
curl http://localhost:5000/api/v1/health
```

---

### 2. Single Image Prediction

**Endpoint:** `POST /predict`

Predict disease from a single leaf image.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `image` (file, required): Image file (.jpg, .jpeg, .png, .webp)

**Response:**
```json
{
  "status": "ok",
  "class_name": "Tomato___Leaf_Mold",
  "confidence": 0.95,
  "margin": 0.12,
  "crop_bn": "টমেটো",
  "disease_bn": "ছাঁচ রোগ",
  "name_bn": "টমেটো - ছাঁচ রোগ",
  "solution_bn": "আক্রান্ত পাতা তুলে নষ্ট করুন...",
  "visual_stats": {
    "green_ratio": 0.35,
    "vegetation_ratio": 0.42,
    "largest_ratio": 0.28,
    "solidity": 0.92,
    "extent": 0.85,
    "face_detected": false
  }
}
```

**Status Codes:**
- `200 OK` - Prediction successful
- `400 Bad Request` - No image provided or invalid format
- `413 Payload Too Large` - File exceeds size limit (8MB)
- `503 Service Unavailable` - Model not loaded
- `500 Internal Server Error` - Prediction failed

**Result Status Values:**
- `ok` - Successfully identified disease
- `not_leaf` - Image doesn't contain crop leaf
- `uncertain` - Confidence below threshold

**Example:**
```bash
curl -X POST -F "image=@leaf_sample.jpg" \
  http://localhost:5000/api/v1/predict
```

**Python:**
```python
import requests

with open('leaf_sample.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post(
        'http://localhost:5000/api/v1/predict',
        files=files
    )
    result = response.json()
    print(result)
```

---

### 3. Batch Image Prediction

**Endpoint:** `POST /predict-batch`

Predict diseases from multiple leaf images.

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `images` (files, required): Multiple image files

**Response:**
```json
{
  "results": [
    {
      "filename": "leaf1.jpg",
      "prediction": {
        "status": "ok",
        "class_name": "Tomato___Healthy",
        "confidence": 0.92,
        "name_bn": "টমেটো - সুস্থ",
        "solution_bn": "রোগ শনাক্ত হয়নি..."
      }
    },
    {
      "filename": "leaf2.jpg",
      "prediction": {
        "status": "not_leaf",
        "name_bn": "ফসলের পাতার ছবি নয়",
        "solution_bn": "পরিষ্কার ফসলের পাতার ছবি দিন..."
      }
    },
    {
      "filename": "invalid.txt",
      "error": "Invalid file type"
    }
  ]
}
```

**Status Codes:**
- `200 OK` - Batch processed (check individual results for errors)
- `400 Bad Request` - No images provided
- `503 Service Unavailable` - Model not loaded
- `500 Internal Server Error` - Batch processing failed

**Example:**
```bash
curl -X POST \
  -F "images=@leaf1.jpg" \
  -F "images=@leaf2.jpg" \
  -F "images=@leaf3.jpg" \
  http://localhost:5000/api/v1/predict-batch
```

**Python:**
```python
import requests

files = [
    ('images', open('leaf1.jpg', 'rb')),
    ('images', open('leaf2.jpg', 'rb')),
    ('images', open('leaf3.jpg', 'rb')),
]

response = requests.post(
    'http://localhost:5000/api/v1/predict-batch',
    files=files
)
results = response.json()
print(results)
```

---

### 4. Model Information

**Endpoint:** `GET /model-info`

Get information about the loaded model and supported crops/diseases.

**Response:**
```json
{
  "model_path": "models/leaf_disease_model.keras",
  "metadata_path": "models/class_names.json",
  "classes_count": 2,
  "crops": ["Tomato"],
  "supported_diseases": {
    "Tomato": ["Healthy", "Leaf_Mold"]
  },
  "min_confidence": 0.7,
  "min_confidence_margin": 0.1
}
```

**Status Codes:**
- `200 OK` - Model info retrieved
- `503 Service Unavailable` - Model not loaded
- `500 Internal Server Error` - Failed to retrieve info

**Example:**
```bash
curl http://localhost:5000/api/v1/model-info
```

---

## Error Responses

All error responses follow this format:

```json
{
  "error": "Error type",
  "message": "Detailed error message",
  "status_code": 400,
  "timestamp": "2024-07-24T10:30:45.123456"
}
```

### Common Error Codes

| Code | Error | Solution |
|------|-------|----------|
| 400 | Bad Request | Check request format and parameters |
| 404 | Not Found | Endpoint doesn't exist |
| 405 | Method Not Allowed | Use correct HTTP method |
| 413 | Payload Too Large | Reduce file size (max 8MB) |
| 500 | Internal Server Error | Check server logs |
| 503 | Service Unavailable | Model not loaded, check /health |

---

## File Upload Specifications

### Supported Formats
- JPEG (.jpg, .jpeg)
- PNG (.png)
- WebP (.webp)

### Size Limits
- Maximum file size: **8 MB**
- Recommended image size: **224x224 pixels**
- Aspect ratio: Any

### Image Quality Requirements
- Clear visibility of leaf affected area
- Good lighting (avoid shadows)
- Minimal background clutter
- No hands or tools in frame
- Recent capture (not old photos)

---

## Response Fields

### Prediction Result

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Result status (ok, not_leaf, uncertain) |
| `class_name` | string | Predicted class (e.g., "Tomato___Leaf_Mold") |
| `confidence` | float | Model confidence (0-1) |
| `margin` | float | Confidence margin between top two predictions |
| `crop_bn` | string | Crop name in Bengali |
| `disease_bn` | string | Disease name in Bengali |
| `name_bn` | string | Full result name in Bengali |
| `solution_bn` | string | Management advice in Bengali |
| `visual_stats` | object | Image analysis statistics |

### Visual Statistics

| Field | Description |
|-------|-------------|
| `green_ratio` | Percentage of green color in image |
| `vegetation_ratio` | Percentage of vegetation detected |
| `red_ratio` | Percentage of red color in image |
| `largest_ratio` | Size of largest connected component |
| `solidity` | Compactness of largest component |
| `extent` | How well component fits bounding box |
| `face_detected` | Whether face detected in image |

---

## Rate Limiting

Currently, there is no rate limiting. For production, implement:

```python
from flask_limiter import Limiter

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

@limiter.limit("5 per minute")
@api_bp.route("/predict", methods=["POST"])
def predict():
    # ...
```

---

## CORS Configuration

To enable CORS for cross-origin requests:

```python
from flask_cors import CORS

CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "https://yourdomain.com"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})
```

---

## Integration Examples

### JavaScript/Fetch
```javascript
async function predictLeaf(imageFile) {
  const formData = new FormData();
  formData.append('image', imageFile);
  
  const response = await fetch(
    'http://localhost:5000/api/v1/predict',
    { method: 'POST', body: formData }
  );
  
  const result = await response.json();
  return result;
}
```

### Python/Requests
```python
import requests

def predict_leaf(image_path):
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(
            'http://localhost:5000/api/v1/predict',
            files=files
        )
    return response.json()

result = predict_leaf('leaf_sample.jpg')
print(f"Disease: {result['name_bn']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### cURL
```bash
# Single prediction
curl -X POST -F "image=@leaf.jpg" http://localhost:5000/api/v1/predict

# Pretty print JSON
curl -X POST -F "image=@leaf.jpg" http://localhost:5000/api/v1/predict | python -m json.tool

# Save response to file
curl -X POST -F "image=@leaf.jpg" http://localhost:5000/api/v1/predict > result.json
```

---

## Versioning

API versioning is handled via URL path:
- Current version: `/api/v1`
- Future versions: `/api/v2`, `/api/v3`, etc.

---

## Support & Feedback

For API issues or feature requests:
- GitHub Issues
- Email: support@leafdisease.local
- Documentation: [DEPLOYMENT.md](DEPLOYMENT.md)
