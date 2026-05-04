# Leaf Disease Detection

TensorFlow image classification project for tomato leaf disease detection. The training script uses MobileNetV2 as a frozen base model and adds custom dense layers for classification.

## Dataset

Expected folder layout:

```text
dataset/
  train/
    Tomato___Healthy/
    Tomato___Leaf_Mold/
  valid/
    Tomato___Healthy/
    Tomato___Leaf_Mold/
```

Images are loaded with Keras directory generators, resized to `224x224`, and normalized with MobileNetV2 preprocessing.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Train

```powershell
python src\train.py --epochs 10
```

MobileNetV2 uses ImageNet weights by default. If you are offline or do not want pretrained weights, run:

```powershell
python src\train.py --epochs 10 --base-weights none
```

The script saves:

```text
models/leaf_disease_model.keras
models/class_names.json
outputs/training_history.png
```

## Predict

```powershell
python src\predict.py path\to\leaf.jpg
```

## Web App

```powershell
python app.py
```

Open `http://127.0.0.1:5000`, upload a leaf image, and the app shows the Bangla disease name and solution.
