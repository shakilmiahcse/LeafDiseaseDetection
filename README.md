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

Current included dataset has only two classes: `Tomato___Healthy` and `Tomato___Leaf_Mold`. A softmax classifier trained with only two classes must choose one of those two labels for every image, including other leaves or tomato fruit. The web app now blocks obvious unsupported/uncertain images, but a reliable user-facing system should be retrained with:

- all tomato leaf disease classes you want to report,
- a `Not_Tomato_Leaf` or `Unknown` negative class containing tomato fruit, other leaves, soil, flowers, hands, and random objects,
- validation images captured from the same kind of phone/camera users will use.

Common tomato classes you can add as folders are:

```text
Tomato___Bacterial_spot
Tomato___Early_blight
Tomato___Late_blight
Tomato___Leaf_Mold
Tomato___Septoria_leaf_spot
Tomato___Spider_mites Two-spotted_spider_mite
Tomato___Target_Spot
Tomato___Tomato_Yellow_Leaf_Curl_Virus
Tomato___Tomato_mosaic_virus
Tomato___Healthy
Not_Tomato_Leaf
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
