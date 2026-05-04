import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Train a tomato leaf disease classifier.")
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
    return parser.parse_args()


def create_data_generators(data_dir, image_size, batch_size, seed):
    train_dir = Path(data_dir) / "train"
    valid_dir = Path(data_dir) / "valid"

    if not train_dir.exists() or not valid_dir.exists():
        raise FileNotFoundError("Expected dataset/train and dataset/valid folders.")

    preprocessing_fn = tf.keras.applications.mobilenet_v2.preprocess_input
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocessing_fn,
    )
    valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=preprocessing_fn,
    )

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True,
        seed=seed,
    )
    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(image_size, image_size),
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False,
    )

    class_names = list(train_generator.class_indices.keys())
    return train_generator, valid_generator, class_names


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

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    output_dir = Path("outputs")

    train_generator, valid_generator, class_names = create_data_generators(
        args.data_dir,
        args.img_size,
        args.batch_size,
        args.seed,
    )

    print(f"Classes: {class_names}")
    model = build_model(len(class_names), args.img_size, args.base_weights)
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
    )

    loss, accuracy = model.evaluate(valid_generator, verbose=0)
    model.save(model_path)

    metadata = {
        "class_names": class_names,
        "image_size": args.img_size,
        "base_model": "MobileNetV2",
        "base_weights": args.base_weights,
        "base_trainable": False,
        "preprocessing": "mobilenet_v2.preprocess_input",
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
