# Leaf Disease Detection

TensorFlow image classification project for tomato leaf disease detection.

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
