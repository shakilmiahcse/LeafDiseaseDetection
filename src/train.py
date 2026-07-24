import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

try:
    from src.disease_advice import summarize_supported_classes
    from src.logger import setup_logger
except ModuleNotFoundError:
    from disease_advice import summarize_supported_classes
    from logger import setup_logger

logger = setup_logger(__name__)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Train a multi-crop leaf disease classifier.")
    parser.add_argument("--data-dir", default="dataset", help="Dataset root with train/ and valid/ folders.")
    parser.add_argument("--model-out", default="models/leaf_disease_model.keras", help="Path to save the model.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--img-size", type=int, default=224, help="Square input image size.")
    parser.add_argument(
        "--base-weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Pretrained weights for MobileNetV2. Use 'none' to train the classifier head without downloading weights.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--min-confidence", type=float, default=0.7, help="Prediction confidence gate saved in metadata.")
    parser.add_argument(
        "--min-confidence-margin",
        type=float,
        default=0.1,
        help="Minimum gap between top two predictions saved in metadata.",
    )
    return parser.parse_args()


def list_image_files(root_dir, class_names):
    image_paths = []
    labels = []

    for class_index, class_name in enumerate(class_names):
        class_dir = root_dir / class_name
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                image_paths.append(str(image_path))
                labels.append(class_index)

    return image_paths, labels


def build_image_dataset(image_paths, labels, num_classes, image_size, batch_size, shuffle=False, seed=42):
    preprocessing_fn = tf.keras.applications.mobilenet_v2.preprocess_input
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)

    def load_and_preprocess(image_path, label):
        image = tf.io.read_file(image_path)
        image = tf.io.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [image_size, image_size])
        image = preprocessing_fn(tf.cast(image, tf.float32))
        label = tf.one_hot(label, num_classes)
        return image, label

    autotune = tf.data.AUTOTUNE
    return dataset.map(load_and_preprocess, num_parallel_calls=autotune).batch(batch_size).prefetch(autotune)


def create_data_generators(data_dir, image_size, batch_size, seed):
    train_dir = Path(data_dir) / "train"
    valid_dir = Path(data_dir) / "valid"

    if not train_dir.exists() or not valid_dir.exists():
        logger.error(f"Dataset directories not found: {train_dir} or {valid_dir}")
        raise FileNotFoundError("Expected dataset/train and dataset/valid folders.")

    class_names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    if not class_names:
        logger.error(f"No class folders found in {train_dir}")
        raise FileNotFoundError(f"No class folders found in {train_dir}")

    train_paths, train_labels = list_image_files(train_dir, class_names)
    valid_paths, valid_labels = list_image_files(valid_dir, class_names)

    if not train_paths:
        logger.error(f"No training images found in {train_dir}")
        raise FileNotFoundError(f"No training images found in {train_dir}")
    if not valid_paths:
        logger.error(f"No validation images found in {valid_dir}")
        raise FileNotFoundError(f"No validation images found in {valid_dir}")

    logger.info(f"Data generators created: {len(class_names)} classes, {len(train_paths)} train, {len(valid_paths)} valid")

    train_dataset = build_image_dataset(
        train_paths,
        train_labels,
        len(class_names),
        image_size,
        batch_size,
        shuffle=True,
        seed=seed,
    )
    valid_dataset = build_image_dataset(
        valid_paths,
        valid_labels,
        len(class_names),
        image_size,
        batch_size,
        shuffle=False,
        seed=seed,
    )

    return train_dataset, valid_dataset, class_names


def build_model(num_classes, image_size, base_weights="imagenet"):
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.1),
        ],
        name="data_augmentation",
    )

    inputs = tf.keras.Input(shape=(image_size, image_size, 3))
    x = data_augmentation(inputs)

    weights = None if base_weights == "none" else base_weights
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(image_size, image_size, 3),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_plot(history, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history.history["loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history.history["accuracy"], label="Train")
    plt.plot(epochs, history.history["val_accuracy"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history.history["loss"], label="Train")
    plt.plot(epochs, history.history["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png", dpi=150)
    plt.close()


def main():
    args = parse_args()
    tf.keras.utils.set_random_seed(args.seed)
    
    logger.info(f"Training started with args: data_dir={args.data_dir}, epochs={args.epochs}, batch_size={args.batch_size}")

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path("outputs")

    try:
        train_generator, valid_generator, class_names = create_data_generators(
            args.data_dir,
            args.img_size,
            args.batch_size,
            args.seed,
        )

        print(f"Classes: {class_names}")
        supported_summary = summarize_supported_classes({"class_names": class_names})
        print(f"Supported crops: {supported_summary['crops_text_bn']}")
        logger.info(f"Dataset loaded: {len(class_names)} classes")

        model = build_model(len(class_names), args.img_size, args.base_weights)
        logger.info("Model built successfully")
        model.summary()

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=4,
                restore_best_weights=True,
                verbose=1,
            ),
        ]

        print(f"\nTraining for {args.epochs} epochs...")
        logger.info(f"Starting training for {args.epochs} epochs...")
        history = model.fit(
            train_generator,
            validation_data=valid_generator,
            epochs=args.epochs,
            callbacks=callbacks,
            verbose=1,
        )
        logger.info("Training completed successfully")

        # Save training history plot
        save_training_plot(history, output_dir)
        logger.info(f"Training plot saved to {output_dir / 'training_history.png'}")
        print(f"Training plot saved to {output_dir / 'training_history.png'}")

        # Save metadata
        metadata = {
            "class_names": class_names,
            "min_confidence": args.min_confidence,
            "min_confidence_margin": args.min_confidence_margin,
            "image_size": args.img_size,
        }
        metadata_path = model_path.parent / "class_names.json"
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        logger.info(f"Metadata saved to {metadata_path}")
        print(f"Metadata saved to {metadata_path}")

        print(f"\nTraining completed!")
        print(f"Model saved to: {model_path}")
        print(f"Metadata saved to: {metadata_path}")
        print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
        logger.info(f"All artifacts saved. Model: {model_path}, Metadata: {metadata_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_generator,
        validation_data=valid_generator,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=2,
    )

    loss, accuracy = model.evaluate(valid_generator, verbose=0)
    model.save(model_path)

    metadata = {
        "task": "multi_crop_leaf_disease",
        "class_names": class_names,
        "class_count": len(class_names),
        "supported_crops": supported_summary["crops"],
        "supported_crop_count": supported_summary["crop_count"],
        "image_size": args.img_size,
        "base_model": "MobileNetV2",
        "base_weights": args.base_weights,
        "base_trainable": False,
        "preprocessing": "mobilenet_v2.preprocess_input",
        "min_confidence": args.min_confidence,
        "min_confidence_margin": args.min_confidence_margin,
        "validation_loss": float(loss),
        "validation_accuracy": float(accuracy),
    }
    (model_path.parent / "class_names.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_training_plot(history, output_dir)

    print(f"Saved model: {model_path}")
    print(f"Saved metadata: {model_path.parent / 'class_names.json'}")
    print(f"Saved plot: {output_dir / 'training_history.png'}")
    print(f"Validation accuracy: {accuracy:.4f}")


if __name__ == "__main__":
    main()
