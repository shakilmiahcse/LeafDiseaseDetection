import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Predict tomato leaf class for one image.")
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument("--model", default="models/leaf_disease_model.keras", help="Trained Keras model path.")
    parser.add_argument("--metadata", default="models/class_names.json", help="Class metadata JSON path.")
    return parser.parse_args()


def load_metadata(path):
    metadata_path = Path(path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def load_image(image_path, image_size):
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32)
    return np.expand_dims(image, axis=0)


def main():
    args = parse_args()
    metadata = load_metadata(args.metadata)
    class_names = metadata["class_names"]
    image_size = metadata["image_size"]

    model = tf.keras.models.load_model(args.model)
    image_batch = load_image(args.image, image_size)
    predictions = model.predict(image_batch, verbose=0)[0]

    class_index = int(np.argmax(predictions))
    confidence = float(predictions[class_index])
    print(f"Prediction: {class_names[class_index]}")
    print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()
