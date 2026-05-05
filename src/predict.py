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


MIN_CONFIDENCE = 0.9
MIN_CONFIDENCE_MARGIN = 0.2
MIN_GREEN_RATIO = 0.12
NEGATIVE_CLASSES = {"Not_Tomato_Leaf", "Unknown"}

DISEASE_SOLUTIONS = {
    "Tomato___Healthy": {
        "name_bn": "সুস্থ টমেটো পাতা",
        "solution_bn": "কোনো রোগ শনাক্ত হয়নি। নিয়মিত পর্যবেক্ষণ করুন, পর্যাপ্ত আলো-বাতাস রাখুন এবং সঠিকভাবে পানি ও সার প্রয়োগ করুন।",
    },
    "Tomato___Bacterial_spot": {
        "name_bn": "টমেটোর ব্যাকটেরিয়াল স্পট",
        "solution_bn": "আক্রান্ত পাতা সরিয়ে ফেলুন, পাতায় পানি জমতে দেবেন না, পরিষ্কার বীজ/চারা ব্যবহার করুন এবং প্রয়োজন হলে কৃষি বিশেষজ্ঞের পরামর্শে কপার-ভিত্তিক ব্যাকটেরিসাইড ব্যবহার করুন।",
    },
    "Tomato___Early_blight": {
        "name_bn": "টমেটোর আর্লি ব্লাইট",
        "solution_bn": "আক্রান্ত পাতা ছেঁটে ফেলুন, গাছের গোড়ায় মালচ দিন, ফসল পর্যায়ক্রম বজায় রাখুন এবং প্রয়োজন হলে অনুমোদিত ছত্রাকনাশক ব্যবহার করুন।",
    },
    "Tomato___Late_blight": {
        "name_bn": "টমেটোর লেট ব্লাইট",
        "solution_bn": "আক্রান্ত অংশ দ্রুত সরিয়ে ফেলুন, জমিতে বাতাস চলাচল বাড়ান, পাতায় পানি দেওয়া এড়িয়ে চলুন এবং দ্রুত কৃষি বিশেষজ্ঞের পরামর্শ নিন।",
    },
    "Tomato___Leaf_Mold": {
        "name_bn": "টমেটোর পাতার ছাঁচ রোগ",
        "solution_bn": "আক্রান্ত পাতা সরিয়ে ফেলুন, গাছের চারপাশে বাতাস চলাচল বাড়ান, পাতায় পানি জমতে দেবেন না এবং প্রয়োজন হলে কৃষি বিশেষজ্ঞের পরামর্শে উপযুক্ত ছত্রাকনাশক ব্যবহার করুন।",
    },
    "Tomato___Septoria_leaf_spot": {
        "name_bn": "টমেটোর সেপটোরিয়া লিফ স্পট",
        "solution_bn": "আক্রান্ত পাতা অপসারণ করুন, গাছের নিচের অংশ পরিষ্কার রাখুন, ওপর থেকে পানি দেওয়া এড়িয়ে চলুন এবং প্রয়োজন হলে অনুমোদিত ছত্রাকনাশক প্রয়োগ করুন।",
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "name_bn": "টমেটোর স্পাইডার মাইট আক্রমণ",
        "solution_bn": "পাতার নিচের অংশ পরীক্ষা করুন, আক্রান্ত পাতা সরান, গাছে পর্যাপ্ত আর্দ্রতা বজায় রাখুন এবং প্রয়োজন হলে কৃষি বিশেষজ্ঞের পরামর্শে মাইটনাশক ব্যবহার করুন।",
    },
    "Tomato___Target_Spot": {
        "name_bn": "টমেটোর টার্গেট স্পট",
        "solution_bn": "আক্রান্ত পাতা সরিয়ে ফেলুন, গাছের ঘনত্ব কমান, জমি পরিষ্কার রাখুন এবং প্রয়োজন হলে অনুমোদিত ছত্রাকনাশক ব্যবহার করুন।",
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "name_bn": "টমেটোর ইয়েলো লিফ কার্ল ভাইরাস",
        "solution_bn": "সাদা মাছি নিয়ন্ত্রণ করুন, আক্রান্ত গাছ আলাদা বা সরিয়ে ফেলুন, আগাছা পরিষ্কার রাখুন এবং রোগমুক্ত চারা ব্যবহার করুন।",
    },
    "Tomato___Tomato_mosaic_virus": {
        "name_bn": "টমেটোর মোজাইক ভাইরাস",
        "solution_bn": "আক্রান্ত গাছের অংশ সরান, হাত ও যন্ত্রপাতি পরিষ্কার রাখুন, আক্রান্ত গাছ স্পর্শের পর সুস্থ গাছ স্পর্শ করবেন না এবং রোগমুক্ত বীজ/চারা ব্যবহার করুন।",
    },
    "Not_Tomato_Leaf": {
        "name_bn": "টমেটো পাতা নয়",
        "solution_bn": "ছবিটি টমেটো পাতার রোগ শনাক্তের জন্য উপযুক্ত নয়। পরিষ্কার টমেটো পাতার ছবি আপলোড করুন।",
    },
    "Unknown": {
        "name_bn": "অজানা বা অসমর্থিত ছবি",
        "solution_bn": "ছবিটি মডেলের শেখা শ্রেণিগুলোর মধ্যে পড়ে না। ভুল ফলাফল এড়াতে রোগ নির্ণয় দেখানো হয়নি।",
    },
}

REJECTION_RESULTS = {
    "not_leaf": {
        "name_bn": "সমর্থিত ছবি নয়",
        "solution_bn": "এই সিস্টেম শুধু পরিষ্কার টমেটো পাতার ছবি বিশ্লেষণ করে। টমেটো ফল, ফুল, মানুষ, মাটি বা অন্য বস্তুর ছবি দিলে রোগের ফলাফল দেখানো নিরাপদ নয়।",
    },
    "non_tomato_leaf_like": {
        "name_bn": "টমেটো পাতা নিশ্চিত নয়",
        "solution_bn": "ছবির পাতার গঠন বর্তমান টমেটো পাতা ডেটাসেটের সাথে যথেষ্ট মেলেনি। ভুল নির্দেশনা এড়াতে রোগের নাম দেখানো হয়নি। পরিষ্কার টমেটো পাতার ছবি দিন বা মডেলটিকে non-tomato leaf ডেটা দিয়ে পুনরায় ট্রেইন করুন।",
    },
    "uncertain": {
        "name_bn": "নিশ্চিতভাবে শনাক্ত করা যায়নি",
        "solution_bn": "মডেলের আত্মবিশ্বাস যথেষ্ট নয়, তাই রোগের নাম বলা নিরাপদ নয়। ভালো আলোতে একটি পরিষ্কার টমেটো পাতার ছবি দিন অথবা কৃষি বিশেষজ্ঞের পরামর্শ নিন।",
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
        return False, "not_leaf", stats

    if stats["red_ratio"] > 0.2 and stats["red_ratio"] > stats["green_ratio"] * 1.5:
        return False, "not_leaf", stats

    if stats["green_ratio"] < MIN_GREEN_RATIO:
        return False, "not_leaf", stats

    if stats["vegetation_ratio"] < 0.08:
        return False, "not_leaf", stats

    simple_green_graphic = (
        stats["green_ratio"] < 0.18
        and stats["largest_ratio"] > 0.16
        and stats["solidity"] > 0.95
        and stats["extent"] > 0.7
    )
    if simple_green_graphic:
        return False, "not_leaf", stats

    broad_simple_leaf = (
        stats["green_ratio"] > 0.28
        and stats["largest_ratio"] > 0.25
        and stats["solidity"] > 0.9
        and stats["extent"] > 0.55
        and stats["red_ratio"] < 0.1
    )
    if broad_simple_leaf:
        return False, "non_tomato_leaf_like", stats

    return True, None, stats


def get_bangla_result(class_name):
    return DISEASE_SOLUTIONS.get(
        class_name,
        {
            "name_bn": class_name,
            "solution_bn": "এই শ্রেণির জন্য এখনো নির্দিষ্ট সমাধান যোগ করা হয়নি। রোগ নিশ্চিত করতে কৃষি বিশেষজ্ঞের পরামর্শ নিন।",
        },
    )


def build_rejection_result(reason, visual_stats=None, predicted_class=None, confidence=None, margin=None):
    result = REJECTION_RESULTS[reason]
    return {
        "status": reason,
        "class_name": predicted_class,
        "confidence": confidence,
        "margin": margin,
        "name_bn": result["name_bn"],
        "solution_bn": result["solution_bn"],
        "visual_stats": visual_stats or {},
    }


def predict_leaf(image_path, model, metadata):
    class_names = metadata["class_names"]
    image_size = metadata["image_size"]

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

    if predicted_class in NEGATIVE_CLASSES:
        return build_rejection_result(
            "not_leaf",
            visual_stats=visual_stats,
            predicted_class=predicted_class,
            confidence=confidence,
            margin=margin,
        )

    if confidence < MIN_CONFIDENCE or margin < MIN_CONFIDENCE_MARGIN:
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
        "name_bn": result["name_bn"],
        "solution_bn": result["solution_bn"],
        "visual_stats": visual_stats,
    }


def main():
    args = parse_args()
    metadata = load_metadata(args.metadata)
    model = tf.keras.models.load_model(args.model, compile=False)
    result = predict_leaf(args.image, model, metadata)
    print(f"রোগের নাম: {result['name_bn']}")
    print(f"সমাধান: {result['solution_bn']}")
    if result["confidence"] is not None:
        print(f"আত্মবিশ্বাস: {result['confidence'] * 100:.2f}%")


if __name__ == "__main__":
    main()
