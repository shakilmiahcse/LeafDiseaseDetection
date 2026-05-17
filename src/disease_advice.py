from __future__ import annotations

import re


NEGATIVE_CLASSES = {
    "Background",
    "Non_Plant",
    "Not_Crop_Leaf",
    "Not_Leaf",
    "Not_Tomato_Leaf",
    "Unknown",
}

CROP_NAMES_BN = {
    "Apple": "আপেল",
    "Banana": "কলা",
    "Bean": "শিম",
    "Blueberry": "ব্লুবেরি",
    "Brinjal": "বেগুন",
    "Cherry_(including_sour)": "চেরি",
    "Chili": "মরিচ",
    "Chilli": "মরিচ",
    "Corn_(maize)": "ভুট্টা",
    "Cotton": "তুলা",
    "Cucumber": "শসা",
    "Eggplant": "বেগুন",
    "Garlic": "রসুন",
    "Grape": "আঙুর",
    "Jute": "পাট",
    "Lentil": "মসুর",
    "Mango": "আম",
    "Mustard": "সরিষা",
    "Okra": "ঢেঁড়স",
    "Onion": "পেঁয়াজ",
    "Orange": "কমলা",
    "Papaya": "পেঁপে",
    "Peach": "পীচ",
    "Pepper,_bell": "ক্যাপসিকাম",
    "Potato": "আলু",
    "Raspberry": "রাস্পবেরি",
    "Rice": "ধান",
    "Soybean": "সয়াবিন",
    "Squash": "স্কোয়াশ",
    "Strawberry": "স্ট্রবেরি",
    "Sugarcane": "আখ",
    "Tea": "চা",
    "Tomato": "টমেটো",
    "Wheat": "গম",
}

DISEASE_NAMES_BN = {
    "apple scab": "স্ক্যাব",
    "bacterial leaf blight": "ব্যাকটেরিয়াল লিফ ব্লাইট",
    "bacterial spot": "ব্যাকটেরিয়াল স্পট",
    "black rot": "ব্ল্যাক রট",
    "blast": "ব্লাস্ট",
    "brown spot": "ব্রাউন স্পট",
    "cedar apple rust": "সিডার অ্যাপল রাস্ট",
    "cercospora leaf spot gray leaf spot": "সারকোস্পোরা/গ্রে লিফ স্পট",
    "common rust": "কমন রাস্ট",
    "early blight": "আর্লি ব্লাইট",
    "esca black measles": "এসকা/ব্ল্যাক মিজলস",
    "haunglongbing citrus greening": "সাইট্রাস গ্রিনিং",
    "healthy": "সুস্থ",
    "late blight": "লেট ব্লাইট",
    "leaf blight isariopsis leaf spot": "লিফ ব্লাইট",
    "leaf mold": "ছাঁচ রোগ",
    "leaf scorch": "লিফ স্কর্চ",
    "mosaic virus": "মোজাইক ভাইরাস",
    "northern leaf blight": "নর্দার্ন লিফ ব্লাইট",
    "powdery mildew": "পাউডারি মিলডিউ",
    "septoria leaf spot": "সেপটোরিয়া লিফ স্পট",
    "sheath blight": "শিথ ব্লাইট",
    "spider mites two spotted spider mite": "স্পাইডার মাইট আক্রমণ",
    "target spot": "টার্গেট স্পট",
    "tomato mosaic virus": "মোজাইক ভাইরাস",
    "tomato yellow leaf curl virus": "ইয়েলো লিফ কার্ল ভাইরাস",
    "yellow leaf curl virus": "ইয়েলো লিফ কার্ল ভাইরাস",
}

ADVICE_BY_CATEGORY = {
    "healthy": (
        "রোগ শনাক্ত হয়নি। নিয়মিত পর্যবেক্ষণ করুন, সুষম সার-পানি দিন, জমিতে পানি জমতে দেবেন না "
        "এবং পাতা ভেজা অবস্থায় অপ্রয়োজনীয় নাড়াচাড়া কম রাখুন।"
    ),
    "bacterial": (
        "আক্রান্ত পাতা বা গাছের অংশ সরিয়ে ফেলুন, পাতায় পানি জমতে দেবেন না, বীজ/চারা ও যন্ত্রপাতি "
        "পরিষ্কার রাখুন, একই জমিতে একই ফসল বারবার চাষ কমান এবং প্রয়োজন হলে স্থানীয় কৃষি কর্মকর্তার "
        "পরামর্শে অনুমোদিত ব্যাকটেরিয়া দমন ব্যবস্থা নিন।"
    ),
    "fungal": (
        "আক্রান্ত পাতা তুলে নষ্ট করুন, গাছের চারপাশে বাতাস চলাচল বাড়ান, উপর থেকে পানি দেওয়া এড়িয়ে চলুন, "
        "জমির আবর্জনা পরিষ্কার রাখুন এবং রোগ ছড়ালে স্থানীয় কৃষি বিশেষজ্ঞের পরামর্শে অনুমোদিত ছত্রাকনাশক ব্যবহার করুন।"
    ),
    "insect_mite": (
        "পাতার নিচের অংশ পরীক্ষা করুন, আক্রান্ত পাতা আলাদা করুন, আগাছা ও বিকল্প আশ্রয়স্থল কমান, উপকারী পোকা "
        "নষ্ট না করে সমন্বিত বালাই ব্যবস্থাপনা অনুসরণ করুন এবং প্রয়োজন হলে অনুমোদিত মাইট/পোকা দমন ব্যবস্থা নিন।"
    ),
    "virus": (
        "ভাইরাস আক্রান্ত গাছ দ্রুত আলাদা বা অপসারণ করুন, সাদা মাছি/এফিডের মতো বাহক পোকা নিয়ন্ত্রণ করুন, "
        "আগাছা পরিষ্কার রাখুন, রোগমুক্ত বীজ/চারা ব্যবহার করুন এবং হাত ও যন্ত্রপাতি পরিষ্কার রাখুন।"
    ),
    "unknown": (
        "এই শ্রেণির জন্য নির্দিষ্ট পরামর্শ এখনো যোগ করা হয়নি। আক্রান্ত পাতা আলাদা রাখুন, ছড়িয়ে পড়া পর্যবেক্ষণ করুন "
        "এবং রোগ নিশ্চিত করতে স্থানীয় কৃষি অফিস বা কৃষি বিশেষজ্ঞের পরামর্শ নিন।"
    ),
}

KEYWORD_CATEGORIES = (
    ("virus", ("virus", "mosaic", "curl")),
    ("insect_mite", ("mite", "spider", "aphid", "whitefly", "thrip", "borer")),
    ("bacterial", ("bacterial",)),
    (
        "fungal",
        (
            "anthracnose",
            "blight",
            "esca",
            "mildew",
            "mold",
            "rot",
            "rust",
            "scab",
            "scorch",
            "spot",
        ),
    ),
)


def normalize_label(label: str) -> str:
    cleaned = re.sub(r"[_\-,()]+", " ", label)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip().lower()


def humanize_label(label: str) -> str:
    cleaned = re.sub(r"___", " - ", label)
    cleaned = re.sub(r"[_]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def split_class_name(class_name: str) -> tuple[str | None, str]:
    if "___" not in class_name:
        return None, class_name
    crop, disease = class_name.split("___", 1)
    return crop, disease


def get_crop_name_bn(crop: str | None) -> str:
    if not crop:
        return "ফসল"
    return CROP_NAMES_BN.get(crop, humanize_label(crop))


def get_disease_name_bn(disease: str) -> str:
    disease_key = normalize_label(disease)
    if disease_key in DISEASE_NAMES_BN:
        return DISEASE_NAMES_BN[disease_key]
    return humanize_label(disease)


def classify_disease(disease: str) -> str:
    disease_key = normalize_label(disease)
    if disease_key == "healthy":
        return "healthy"

    for category, keywords in KEYWORD_CATEGORIES:
        if any(keyword in disease_key for keyword in keywords):
            return category

    return "unknown"


def get_bangla_result(class_name: str) -> dict[str, str | None]:
    if class_name in NEGATIVE_CLASSES:
        return {
            "crop_bn": None,
            "disease_bn": None,
            "name_bn": "সমর্থিত ফসলের পাতা নয়",
            "solution_bn": "পরিষ্কার ফসলের পাতার ছবি দিন। ফল, ফুল, মাটি, মানুষ বা অন্য বস্তুর ছবিতে রোগ নির্ণয় দেখানো নিরাপদ নয়।",
        }

    crop, disease = split_class_name(class_name)
    crop_bn = get_crop_name_bn(crop)
    disease_bn = get_disease_name_bn(disease)
    category = classify_disease(disease)
    advice = ADVICE_BY_CATEGORY[category]

    if category == "healthy":
        name_bn = f"সুস্থ {crop_bn} পাতা"
    elif crop:
        name_bn = f"{crop_bn} পাতার {disease_bn}"
    else:
        name_bn = disease_bn

    return {
        "crop_bn": crop_bn,
        "disease_bn": disease_bn,
        "name_bn": name_bn,
        "solution_bn": advice,
    }


def summarize_supported_classes(metadata: dict) -> dict[str, object]:
    class_names = metadata.get("class_names", [])
    crops = []

    for class_name in class_names:
        if class_name in NEGATIVE_CLASSES:
            continue
        crop, _ = split_class_name(class_name)
        if crop:
            crops.append(get_crop_name_bn(crop))

    unique_crops = sorted(set(crops))
    return {
        "class_count": len(class_names),
        "crop_count": len(unique_crops),
        "crops": unique_crops,
        "crops_text_bn": ", ".join(unique_crops) if unique_crops else "ডেটাসেট নির্ধারিত হয়নি",
        "is_multi_crop": len(unique_crops) > 1,
    }
