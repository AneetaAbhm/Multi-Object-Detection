"""
STEP 1: Run this script LOCALLY on your computer.
It will create a clean 500-image subset from the 26-class dataset.

Usage:
    python create_subset.py --dataset_path "C:/path/to/your/dataset"

Output:
    A folder called 'yolo_subset' ready to zip and upload to Google Drive / Colab.
"""

import os
import shutil
import random
import argparse
from collections import defaultdict
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION — Change these if needed
# ─────────────────────────────────────────────

# Pick any 5 classes from the dataset's data.yaml
# Common classes in 26-class datasets (adjust after checking your data.yaml):
SELECTED_CLASSES = ['Person', 'bicycle', 'car', 'tree', 'door']

# Images per class, per split
IMAGES_PER_CLASS = {
    'train': 70,   # 70 × 5 = 350
    'valid': 20,   # 20 × 5 = 100
    'test':  10,   # 10 × 5 =  50
}                  # Total  = 500

OUTPUT_DIR = 'yolo_subset'
SPLITS = ['train', 'valid', 'test']

# ─────────────────────────────────────────────

def read_yaml_classes(dataset_path):
    """Read class names from data.yaml"""
    yaml_path = Path(dataset_path) / 'data.yaml'
    if not yaml_path.exists():
        yaml_path = list(Path(dataset_path).rglob('data.yaml'))
        yaml_path = yaml_path[0] if yaml_path else None

    if yaml_path is None:
        print("⚠️  data.yaml not found. Please check your dataset path.")
        return None

    with open(yaml_path, 'r') as f:
        content = f.read()

    print(f"\n📄 data.yaml found at: {yaml_path}")
    print("─" * 50)
    print(content)
    print("─" * 50)
    return yaml_path


def get_class_ids(dataset_path, selected_classes):
    """Map selected class names to their integer IDs from data.yaml"""
    import yaml  # pip install pyyaml

    yaml_path = Path(dataset_path) / 'data.yaml'
    if not yaml_path.exists():
        found = list(Path(dataset_path).rglob('data.yaml'))
        yaml_path = found[0] if found else None

    if yaml_path is None:
        raise FileNotFoundError("data.yaml not found!")

    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    all_classes = data.get('names', [])
    if isinstance(all_classes, dict):
        all_classes = list(all_classes.values())

    print(f"\n📦 All classes in dataset ({len(all_classes)} total):")
    for i, name in enumerate(all_classes):
        marker = "✅" if name in selected_classes else "  "
        print(f"  {marker} [{i:2d}] {name}")

    class_id_map = {}
    for cls in selected_classes:
        if cls in all_classes:
            class_id_map[cls] = all_classes.index(cls)
        else:
            print(f"\n❌ Class '{cls}' NOT found in dataset!")
            print(f"   Available: {all_classes}")
            raise ValueError(f"Class '{cls}' not in dataset. Fix SELECTED_CLASSES.")

    print(f"\n🎯 Selected class → ID mapping:")
    for cls, cid in class_id_map.items():
        print(f"   {cls} → {cid}")

    return class_id_map, all_classes


def label_has_class(label_path, target_class_ids):
    """Return True if label file contains at least one of the target classes"""
    if not label_path.exists():
        return False
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if parts and int(parts[0]) in target_class_ids:
                return True
    return False


def remap_label(label_path, old_to_new):
    """Read a label file and remap class IDs; return only lines with selected classes"""
    lines = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            old_id = int(parts[0])
            if old_id in old_to_new:
                new_id = old_to_new[old_id]
                lines.append(f"{new_id} " + " ".join(parts[1:]))
    return lines


def collect_images_per_class(images_dir, labels_dir, class_id_map):
    """For each class, collect image paths that contain that class."""
    class_to_images = defaultdict(list)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    img_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in img_extensions:
            continue
        label_path = labels_dir / (img_path.stem + '.txt')
        if not label_path.exists():
            continue
        with open(label_path, 'r') as f:
            found_classes = set()
            for line in f:
                parts = line.strip().split()
                if parts:
                    found_classes.add(int(parts[0]))

        for cls_name, cls_id in class_id_map.items():
            if cls_id in found_classes:
                class_to_images[cls_name].append(img_path)

    return class_to_images


def create_subset(dataset_path, output_dir, selected_classes, images_per_class):
    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)

    # Step 1: Read class IDs
    class_id_map, all_classes = get_class_ids(dataset_path, selected_classes)
    old_to_new = {v: i for i, (k, v) in enumerate(class_id_map.items())}
    new_class_names = list(class_id_map.keys())

    # Step 2: Create output folders
    for split in SPLITS:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

    total_copied = 0
    split_summary = {}

    for split in SPLITS:
        print(f"\n{'═'*50}")
        print(f"  Processing split: {split.upper()}")
        print(f"{'═'*50}")

        # Handle alternate folder names (valid vs validation vs val)
        possible_dirs = [split, 'val' if split == 'valid' else split, 'validation' if split == 'valid' else split]
        split_dir = None
        for d in possible_dirs:
            candidate = dataset_path / d
            if candidate.exists():
                split_dir = candidate
                break

        if split_dir is None:
            print(f"  ⚠️  Folder '{split}' not found, skipping.")
            continue

        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'

        if not images_dir.exists():
            print(f"  ⚠️  images/ folder not found in {split_dir}")
            continue

        # Collect images per class
        class_to_images = collect_images_per_class(images_dir, labels_dir, class_id_map)

        target = images_per_class.get(split, 20)
        selected_images = set()
        split_summary[split] = {}

        for cls_name in selected_classes:
            available = class_to_images.get(cls_name, [])
            random.shuffle(available)
            chosen = []
            for img in available:
                if img not in selected_images:
                    chosen.append(img)
                    selected_images.add(img)
                if len(chosen) >= target:
                    break

            split_summary[split][cls_name] = len(chosen)
            print(f"  📁 {cls_name}: {len(chosen)}/{target} images selected")

        # Copy selected images + remapped labels
        copied = 0
        for img_path in selected_images:
            label_path = labels_dir / (img_path.stem + '.txt')
            if not label_path.exists():
                continue

            remapped = remap_label(label_path, old_to_new)
            if not remapped:
                continue

            shutil.copy2(img_path, output_dir / split / 'images' / img_path.name)
            with open(output_dir / split / 'labels' / (img_path.stem + '.txt'), 'w') as f:
                f.write('\n'.join(remapped))
            copied += 1

        total_copied += copied
        print(f"  ✅ {copied} image-label pairs copied for {split}")

    # Step 3: Write new data.yaml
    yaml_content = f"""# YOLO Subset Dataset
# Classes: {new_class_names}
# Created from: 26-class object detection dataset

path: .  # dataset root (update this in Colab)
train: train/images
val: valid/images
test: test/images

nc: {len(new_class_names)}
names: {new_class_names}
"""
    with open(output_dir / 'data.yaml', 'w') as f:
        f.write(yaml_content)

    # Step 4: Summary
    print(f"\n{'═'*50}")
    print(f"  ✅ SUBSET CREATION COMPLETE")
    print(f"{'═'*50}")
    print(f"  Total images: {total_copied}")
    print(f"  Output folder: {output_dir.resolve()}")
    print(f"\n  Split breakdown:")
    for split, cls_counts in split_summary.items():
        print(f"    {split}: {sum(cls_counts.values())} images")
        for cls, count in cls_counts.items():
            print(f"      - {cls}: {count}")

    print(f"\n📌 Next step: Zip the '{output_dir}' folder and upload to Google Drive.")
    print(f"   Then open the Colab notebook provided.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create YOLO subset dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the downloaded Kaggle dataset folder')
    parser.add_argument('--output', type=str, default=OUTPUT_DIR,
                        help='Output folder name (default: yolo_subset)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    random.seed(args.seed)

    print("🚀 YOLO Subset Creator")
    print(f"   Dataset: {args.dataset_path}")
    print(f"   Classes: {SELECTED_CLASSES}")

    # First, show available classes by reading yaml
    read_yaml_classes(args.dataset_path)

    print("\n⚠️  If the class names above don't match SELECTED_CLASSES,")
    print("   edit the SELECTED_CLASSES list at the top of this script and rerun.\n")

    confirm = input("Continue with subset creation? (y/n): ").strip().lower()
    if confirm == 'y':
        create_subset(args.dataset_path, args.output, SELECTED_CLASSES, IMAGES_PER_CLASS)
    else:
        print("Aborted. Edit SELECTED_CLASSES and rerun.")