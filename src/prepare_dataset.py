from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create train/valid folders from class directories such as Crop___Disease."
    )
    parser.add_argument("--source", required=True, help="Folder containing one subfolder per class.")
    parser.add_argument("--output", default="dataset", help="Output dataset folder.")
    parser.add_argument("--valid-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible splits.")
    parser.add_argument("--classes", nargs="*", help="Optional class folder names to include.")
    parser.add_argument(
        "--max-images-per-class",
        type=int,
        help="Optional maximum number of source images to use from each class.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print the split summary.")
    parser.add_argument(
        "--clear-output",
        action="store_true",
        help="Delete existing output train/ and valid/ folders before copying.",
    )
    return parser.parse_args()


def validate_args(args):
    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.exists():
        raise FileNotFoundError(f"Source folder does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a folder: {source_dir}")
    if not 0 < args.valid_ratio < 0.5:
        raise ValueError("--valid-ratio must be greater than 0 and less than 0.5")
    if args.max_images_per_class is not None and args.max_images_per_class < 2:
        raise ValueError("--max-images-per-class must be at least 2")

    return source_dir, output_dir


def find_class_dirs(source_dir, selected_classes=None):
    selected = set(selected_classes or [])
    class_dirs = [path for path in source_dir.iterdir() if path.is_dir()]

    if selected:
        class_dirs = [path for path in class_dirs if path.name in selected]
        missing = sorted(selected - {path.name for path in class_dirs})
        if missing:
            raise FileNotFoundError(f"Missing selected class folders: {', '.join(missing)}")

    return sorted(class_dirs, key=lambda path: path.name.lower())


def list_images(class_dir):
    return sorted(
        path
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def split_images(images, valid_ratio, rng, max_images_per_class=None):
    shuffled = list(images)
    rng.shuffle(shuffled)

    if max_images_per_class is not None:
        shuffled = shuffled[:max_images_per_class]

    if len(shuffled) <= 1:
        return shuffled, []

    valid_count = max(1, round(len(shuffled) * valid_ratio))
    valid_count = min(valid_count, len(shuffled) - 1)
    return shuffled[valid_count:], shuffled[:valid_count]


def ensure_safe_output(output_dir):
    resolved_output = output_dir.resolve()
    workspace = Path.cwd().resolve()

    if resolved_output == workspace:
        raise ValueError("Refusing to clear the project root.")
    if workspace not in resolved_output.parents:
        raise ValueError(f"Refusing to clear output outside this workspace: {resolved_output}")


def clear_existing_splits(output_dir):
    ensure_safe_output(output_dir)

    for split_name in ("train", "valid"):
        split_dir = output_dir / split_name
        if split_dir.exists():
            shutil.rmtree(split_dir)


def unique_target_path(target_dir, source_name):
    target = target_dir / source_name
    if not target.exists():
        return target

    stem = Path(source_name).stem
    suffix = Path(source_name).suffix
    counter = 2
    while True:
        candidate = target_dir / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def copy_images(images, output_dir, split_name, class_name):
    target_dir = output_dir / split_name / class_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for image_path in images:
        target_path = unique_target_path(target_dir, image_path.name)
        shutil.copy2(image_path, target_path)


def build_dataset(source_dir, output_dir, class_dirs, valid_ratio, seed, max_images_per_class=None, dry_run=False):
    rng = random.Random(seed)
    summary = []

    for class_dir in class_dirs:
        images = list_images(class_dir)
        train_images, valid_images = split_images(images, valid_ratio, rng, max_images_per_class)
        summary.append((class_dir.name, len(train_images), len(valid_images), len(images)))

        if dry_run:
            continue

        copy_images(train_images, output_dir, "train", class_dir.name)
        copy_images(valid_images, output_dir, "valid", class_dir.name)

    return summary


def print_summary(summary, dry_run=False):
    mode = "Dry run" if dry_run else "Created"
    print(f"{mode} dataset split:")
    for class_name, train_count, valid_count, total_count in summary:
        print(f"- {class_name}: train={train_count}, valid={valid_count}, total={total_count}")

    total_train = sum(item[1] for item in summary)
    total_valid = sum(item[2] for item in summary)
    print(f"Total: train={total_train}, valid={total_valid}, classes={len(summary)}")


def main():
    args = parse_args()
    source_dir, output_dir = validate_args(args)
    class_dirs = find_class_dirs(source_dir, args.classes)

    if args.clear_output and not args.dry_run:
        clear_existing_splits(output_dir)

    summary = build_dataset(
        source_dir=source_dir,
        output_dir=output_dir,
        class_dirs=class_dirs,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        max_images_per_class=args.max_images_per_class,
        dry_run=args.dry_run,
    )
    print_summary(summary, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
