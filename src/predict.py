import argparse
import json
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np

from .logger import setup_logger

try:
    from src.disease_advice import (
        NEGATIVE_CLASSES as NEGATIVE_CLASS_LABELS,
        get_bangla_result as build_bangla_advice,
    )
except ModuleNotFoundError:
    from disease_advice import (
        NEGATIVE_CLASSES as NEGATIVE_CLASS_LABELS,
        get_bangla_result as build_bangla_advice,
    )

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

logger = setup_logger(__name__)


DEFAULT_MIN_CONFIDENCE = 0.7
DEFAULT_MIN_CONFIDENCE_MARGIN = 0.1
MIN_GREEN_RATIO = 0.06
MIN_VEGETATION_RATIO = 0.08

REJECTION_RESULTS = {
    "not_leaf": {
        "name_bn": "ফসলের পাতার ছবি নয়",
        "solution_bn": "পরিষ্কার ফসলের পাতার ছবি দিন। ফল, ফুল, মাটি, মানুষ বা অন্য বস্তুর ছবিতে রোগের ফলাফল দেখানো নিরাপদ নয়।",
    },
    "uncertain": {
        "name_bn": "নিশ্চিতভাবে শনাক্ত করা যায়নি",
        "solution_bn": "মডেলের আত্মবিশ্বাস যথেষ্ট নয়, তাই রোগের নাম বলা নিরাপদ নয়। ভালো আলোতে পরিষ্কার ফসলের পাতার ছবি দিন অথবা কৃষি বিশেষজ্ঞের পরামর্শ নিন।",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict crop leaf disease class for one image.")
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument("--model", default="models/leaf_disease_model.keras", help="Trained Keras model path.")
    parser.add_argument("--metadata", default="models/class_names.json", help="Class metadata JSON path.")
    return parser.parse_args()


def load_metadata(path):
    metadata_path = Path(path)
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    logger.info(f"Loading metadata from {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_image(image_path, image_size):
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Could not read image: {image_path}")
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    import tensorflow as tf
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    logger.debug(f"Image loaded and preprocessed: {image_path}")
    return np.expand_dims(image, axis=0)


def detect_face(image):
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(str(cascade_path))
    if face_cascade.empty():
        return False, 0.0

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(55, 55))
    if len(faces) == 0:
        return False, 0.0

    image_area = float(image.shape[0] * image.shape[1])
    largest_face_area = max(float(width * height) for _, _, width, height in faces)
    return True, largest_face_area / image_area


def analyze_image_content(image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.resize(image, (384, 384), interpolation=cv2.INTER_AREA)
    face_detected, face_ratio = detect_face(image)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, np.array([25, 35, 35]), np.array([100, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([12, 45, 45]), np.array([35, 255, 255]))
    red_mask_low = cv2.inRange(hsv, np.array([0, 70, 45]), np.array([10, 255, 255]))
    red_mask_high = cv2.inRange(hsv, np.array([170, 70, 45]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask_low, red_mask_high)
    vegetation_mask = cv2.bitwise_or(green_mask, yellow_mask)

    kernel = np.ones((5, 5), np.uint8)
    vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel)
    vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)

    image_area = float(image.shape[0] * image.shape[1])
    green_ratio = float(np.count_nonzero(green_mask) / image_area)
    vegetation_ratio = float(np.count_nonzero(vegetation_mask) / image_area)
    red_ratio = float(np.count_nonzero(red_mask) / image_area)

    contours, _ = cv2.findContours(vegetation_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_ratio = 0.0
    solidity = 0.0
    extent = 0.0
    if contours:
        largest = max(contours, key=cv2.contourArea)
        largest_area = float(cv2.contourArea(largest))
        largest_ratio = largest_area / image_area

        hull = cv2.convexHull(largest)
        hull_area = float(cv2.contourArea(hull))
        if hull_area > 0:
            solidity = largest_area / hull_area

        _, _, width, height = cv2.boundingRect(largest)
        box_area = float(width * height)
        if box_area > 0:
            extent = largest_area / box_area

    return {
        "green_ratio": green_ratio,
        "vegetation_ratio": vegetation_ratio,
        "red_ratio": red_ratio,
        "largest_ratio": largest_ratio,
        "solidity": solidity,
        "extent": extent,
        "face_detected": face_detected,
        "face_ratio": face_ratio,
    }


def validate_input_image(image_path):
    stats = analyze_image_content(image_path)

    if stats["face_detected"] and stats["face_ratio"] > 0.03:
        logger.warning(f"Face detected in image: {image_path}")
        return False, "not_leaf", stats

    if stats["red_ratio"] > 0.2 and stats["red_ratio"] > stats["green_ratio"] * 1.5:
        logger.warning(f"High red ratio detected: {image_path}")
        return False, "not_leaf", stats

    if stats["green_ratio"] < MIN_GREEN_RATIO:
        logger.warning(f"Low green ratio: {image_path}")
        return False, "not_leaf", stats

    if stats["vegetation_ratio"] < MIN_VEGETATION_RATIO:
        logger.warning(f"Low vegetation ratio: {image_path}")
        return False, "not_leaf", stats

    simple_green_graphic = (
        stats["green_ratio"] < 0.18
        and stats["largest_ratio"] > 0.16
        and stats["solidity"] > 0.95
        and stats["extent"] > 0.7
    )
    if simple_green_graphic:
        logger.warning(f"Simple green graphic detected: {image_path}")
        return False, "not_leaf", stats

    logger.debug(f"Image validation passed: {image_path}")
    return True, None, stats


def get_bangla_result(class_name):
    return build_bangla_advice(class_name)


def build_rejection_result(reason, visual_stats=None, predicted_class=None, confidence=None, margin=None):
    result = REJECTION_RESULTS[reason]
    return {
        "status": reason,
        "class_name": predicted_class,
        "confidence": confidence,
        "margin": margin,
        "crop_bn": None,
        "disease_bn": None,
        "name_bn": result["name_bn"],
        "solution_bn": result["solution_bn"],
        "visual_stats": visual_stats or {},
    }


def get_prediction_thresholds(metadata):
    return (
        float(metadata.get("min_confidence", DEFAULT_MIN_CONFIDENCE)),
        float(metadata.get("min_confidence_margin", DEFAULT_MIN_CONFIDENCE_MARGIN)),
    )


def predict_leaf(image_path, model, metadata):
    class_names = metadata["class_names"]
    image_size = metadata["image_size"]
    min_confidence, min_confidence_margin = get_prediction_thresholds(metadata)

    is_valid, rejection_reason, visual_stats = validate_input_image(image_path)
    if not is_valid:
        return build_rejection_result(rejection_reason, visual_stats=visual_stats)

    image_batch = load_image(image_path, image_size)
    predictions = model.predict(image_batch, verbose=0)[0]

    class_index = int(np.argmax(predictions))
    predicted_class = class_names[class_index]
    sorted_predictions = np.sort(predictions)
    confidence = float(predictions[class_index])
    margin = confidence if len(sorted_predictions) == 1 else float(sorted_predictions[-1] - sorted_predictions[-2])

    if predicted_class in NEGATIVE_CLASS_LABELS:
        return build_rejection_result(
            "not_leaf",
            visual_stats=visual_stats,
            predicted_class=predicted_class,
            confidence=confidence,
            margin=margin,
        )

    if confidence < min_confidence or margin < min_confidence_margin:
        return build_rejection_result(
            "uncertain",
            visual_stats=visual_stats,
            predicted_class=predicted_class,
            confidence=confidence,
            margin=margin,
        )

    result = get_bangla_result(predicted_class)
    return {
        "status": "ok",
        "class_name": predicted_class,
        "confidence": confidence,
        "margin": margin,
        "crop_bn": result.get("crop_bn"),
        "disease_bn": result.get("disease_bn"),
        "name_bn": result["name_bn"],
        "solution_bn": result["solution_bn"],
        "visual_stats": visual_stats,
    }


def main():
    args = parse_args()
    metadata = load_metadata(args.metadata)
    # Import TensorFlow only when running the CLI, so importing this module
    # doesn't require TensorFlow to be installed in all environments.
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    model = tf.keras.models.load_model(args.model, compile=False)
    result = predict_leaf(args.image, model, metadata)
    print(f"রোগের নাম: {result['name_bn']}")
    print(f"সমাধান: {result['solution_bn']}")
    if result["confidence"] is not None:
        print(f"আত্মবিশ্বাস: {result['confidence'] * 100:.2f}%")


def load_model_from_path(path):
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")

    p = Path(path)
    try:
        return tf.keras.models.load_model(str(p), compile=False)
    except Exception as exc:
        # Attempt sensible fallbacks if the primary file isn't a valid Keras archive.
        candidates = [
            p.with_suffix(".h5"),
            p.parent / "leaf_model.h5",
            p.parent / "leaf_disease_model_retrained.keras",
        ]
        for cand in candidates:
            if cand.exists():
                try:
                    return tf.keras.models.load_model(str(cand), compile=False)
                except Exception:
                    continue
        # Nothing worked — re-raise the original exception for visibility.
        raise


if __name__ == "__main__":
    main()
