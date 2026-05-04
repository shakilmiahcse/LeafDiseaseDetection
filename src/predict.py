import argparse
import json
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


DISEASE_SOLUTIONS = {
    "Tomato___Healthy": {
        "name_bn": "সুস্থ টমেটো পাতা",
        "solution_bn": "কোনো রোগ শনাক্ত হয়নি। নিয়মিত পর্যবেক্ষণ করুন, পর্যাপ্ত আলো-বাতাস রাখুন এবং সঠিকভাবে পানি ও সার প্রয়োগ করুন।",
    },
    "Tomato___Leaf_Mold": {
        "name_bn": "টমেটোর পাতার ছাঁচ রোগ",
        "solution_bn": "আক্রান্ত পাতা সরিয়ে ফেলুন, গাছের চারপাশে বাতাস চলাচল বাড়ান, পাতায় পানি জমতে দেবেন না এবং প্রয়োজন হলে কৃষি বিশেষজ্ঞের পরামর্শে উপযুক্ত ছত্রাকনাশক ব্যবহার করুন।",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description="Predict tomato leaf class for one image.")
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument("--model", default="models/leaf_model.h5", help="Trained Keras model path.")
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
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return np.expand_dims(image, axis=0)


def get_bangla_result(class_name):
    return DISEASE_SOLUTIONS.get(
        class_name,
        {
            "name_bn": class_name,
            "solution_bn": "এই শ্রেণির জন্য কোনো নির্দিষ্ট সমাধান পাওয়া যায়নি।",
        },
    )


def predict_leaf(image_path, model, metadata):
    class_names = metadata["class_names"]
    image_size = metadata["image_size"]

    image_batch = load_image(image_path, image_size)
    predictions = model.predict(image_batch, verbose=0)[0]

    class_index = int(np.argmax(predictions))
    predicted_class = class_names[class_index]
    result = get_bangla_result(predicted_class)
    return {
        "class_name": predicted_class,
        "confidence": float(predictions[class_index]),
        "name_bn": result["name_bn"],
        "solution_bn": result["solution_bn"],
    }


def main():
    args = parse_args()
    metadata = load_metadata(args.metadata)
    model = tf.keras.models.load_model(args.model, compile=False)
    result = predict_leaf(args.image, model, metadata)
    print(f"রোগের নাম: {result['name_bn']}")
    print(f"সমাধান: {result['solution_bn']}")


if __name__ == "__main__":
    main()
