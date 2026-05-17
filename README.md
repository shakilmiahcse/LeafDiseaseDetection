# Multi-Crop Leaf Disease Detection

TensorFlow + Flask image classification project for crop leaf disease detection. The code is now multi-crop-ready: it reads class folders such as `Potato___Late_blight`, `Corn_(maize)___Common_rust_`, or `Rice___Brown_spot`, predicts the class, and shows a Bangla disease name with general management advice.

Important: the model can only recognize classes it was trained on. The dataset currently included in this workspace still has only:

```text
Tomato___Healthy
Tomato___Leaf_Mold
```

To detect many crops, add a full multi-crop leaf dataset and retrain the model.

## Quick Start

1. Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install pinned dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Run the web app:

```powershell
python app.py
```

4. Open your browser to:

```text
http://127.0.0.1:5000
```

If the app cannot load the default model file, set `LEAF_MODEL_PATH` to a valid saved model path before starting the app.

## Dataset Layout

Use one folder per class under `train/` and `valid/`:

```text
dataset/
  train/
    Apple___Apple_scab/
    Corn_(maize)___Common_rust_/
    Potato___Late_blight/
    Tomato___Leaf_Mold/
    Not_Leaf/
  valid/
    Apple___Apple_scab/
    Corn_(maize)___Common_rust_/
    Potato___Late_blight/
    Tomato___Leaf_Mold/
    Not_Leaf/
```

Recommended negative classes:

```text
Not_Leaf
Not_Crop_Leaf
Unknown
Background
```

Put non-leaf images there, for example fruits, flowers, soil, hands, tools, and random objects. This helps the model avoid confident wrong disease predictions.

## Prepare Dataset

If you have a PlantVillage-style source folder where each class is a subfolder, create a train/valid split:

```powershell
python src\prepare_dataset.py --source _plantvillage_src\raw\color --output dataset --valid-ratio 0.2 --dry-run
python src\prepare_dataset.py --source _plantvillage_src\raw\color --output dataset --valid-ratio 0.2 --clear-output
```

You can include specific classes only:

```powershell
python src\prepare_dataset.py --source path\to\all_classes --classes Potato___Late_blight Potato___Healthy --output dataset
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## GitHub-ready repo

The repo is configured so local artifacts are ignored. The following are excluded by `.gitignore`:

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
