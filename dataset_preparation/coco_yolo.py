"""
COCO to YOLO Format Converter
==============================

Usage:
    python coco_to_yolo.py --dataset_path "C:/path/to/your/dataset"

What it does:
    Reads _annotations.coco.json from each split (train/valid/test)
    and creates corresponding YOLO .txt label files.

Output structure (in-place, labels/ folder added):
    dataset/
        train/
            images/       ← already exists
            labels/       ← CREATED by this script
        valid/
            images/
            labels/
        test/
            images/
            labels/
        data.yaml         ← CREATED by this script
"""

import os
import json
import argparse
from pathlib import Path


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

SPLITS = ['train', 'valid', 'test']

# Common annotation file names used in COCO datasets
POSSIBLE_ANNOTATION_FILENAMES = [
    '_annotations.coco.json',
    'annotations.json',
    '_annotations.json',
    'instances_train.json',
    'instances_val.json',
    'instances_test.json',
]

# ─────────────────────────────────────────────


def find_annotation_file(split_dir):
    """Search for the COCO annotation JSON in a split folder."""
    split_dir = Path(split_dir)

    # Check directly in split folder
    for name in POSSIBLE_ANNOTATION_FILENAMES:
        p = split_dir / name
        if p.exists():
            return p

    # Check in annotations/ subfolder
    ann_dir = split_dir / 'annotations'
    if ann_dir.exists():
        for name in POSSIBLE_ANNOTATION_FILENAMES:
            p = ann_dir / name
            if p.exists():
                return p

    # Fallback: find any .json file
    json_files = list(split_dir.rglob('*.json'))
    if json_files:
        print(f"   No standard annotation file found. Using: {json_files[0]}")
        return json_files[0]

    return None


def coco_bbox_to_yolo(image_width, image_height, bbox):
    """
    Convert COCO bbox [x_min, y_min, width, height]
    to YOLO format [cx, cy, w, h] normalized 0-1.
    """
    x_min, y_min, w, h = bbox

    cx = (x_min + w / 2) / image_width
    cy = (y_min + h / 2) / image_height
    w_norm = w / image_width
    h_norm = h / image_height

    # Clamp to [0, 1] to handle any boundary issues
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))

    return cx, cy, w_norm, h_norm


def convert_split(split_dir, split_name):
    """Convert one split (train/valid/test) from COCO to YOLO format."""
    split_dir = Path(split_dir)
    print(f"\n{'═'*55}")
    print(f"  Converting: {split_name.upper()}")
    print(f"{'═'*55}")

    # Find annotation file
    ann_file = find_annotation_file(split_dir)
    if ann_file is None:
        print(f"   No annotation JSON found in {split_dir}. Skipping.")
        return None, None

    print(f"   Annotation file: {ann_file.name}")

    # Load COCO JSON
    with open(ann_file, 'r') as f:
        coco = json.load(f)

    # ── Parse categories ──
    categories = coco.get('categories', [])
    # Sort by id to keep consistent ordering
    categories = sorted(categories, key=lambda x: x['id'])
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}
    cat_id_to_index = {cat['id']: i for i, cat in enumerate(categories)}

    print(f"\n  Categories found ({len(categories)}):")
    for cat in categories:
        idx = cat_id_to_index[cat['id']]
        print(f"     [{idx:2d}] {cat['name']}  (original id={cat['id']})")

    # ── Build image id → info map ──
    images = coco.get('images', [])
    img_id_to_info = {img['id']: img for img in images}
    print(f"\n   Total images in JSON: {len(images)}")

    # ── Build image id → list of annotations ──
    annotations = coco.get('annotations', [])
    print(f"   Total annotations:    {len(annotations)}")

    img_id_to_anns = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)

    # ── Create labels/ folder ──
    labels_dir = split_dir / 'labels'
    labels_dir.mkdir(exist_ok=True)
    print(f"\n  Labels will be saved to: {labels_dir}")

    # ── Convert each image ──
    converted = 0
    skipped = 0
    no_ann = 0

    for img_info in images:
        img_id = img_info['id']
        file_name = img_info['file_name']
        img_w = img_info['width']
        img_h = img_info['height']

        # Derive label filename from image filename
        stem = Path(file_name).stem
        label_path = labels_dir / f"{stem}.txt"

        anns = img_id_to_anns.get(img_id, [])

        if not anns:
            # Create empty label file (YOLO requires this for background images)
            label_path.touch()
            no_ann += 1
            continue

        lines = []
        for ann in anns:
            cat_id = ann['category_id']
            if cat_id not in cat_id_to_index:
                skipped += 1
                continue

            class_idx = cat_id_to_index[cat_id]
            bbox = ann.get('bbox', [])

            if len(bbox) != 4:
                skipped += 1
                continue

            # Skip invalid boxes
            if bbox[2] <= 0 or bbox[3] <= 0:
                skipped += 1
                continue

            cx, cy, w, h = coco_bbox_to_yolo(img_w, img_h, bbox)
            lines.append(f"{class_idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        with open(label_path, 'w') as f:
            f.write('\n'.join(lines))

        converted += 1

    print(f"\n   Converted:        {converted} images")
    print(f"   No annotations:  {no_ann} images (empty .txt created)")
    print(f"    Skipped boxes:   {skipped}")

    return cat_id_to_name, cat_id_to_index, categories


def create_data_yaml(dataset_path, categories):
    """Create a data.yaml file for YOLOv8."""
    class_names = [cat['name'] for cat in sorted(categories, key=lambda x: x['id'])]

    yaml_content = f"""# Auto-generated by coco_to_yolo.py
path: {Path(dataset_path).resolve()}
train: train/images
val: valid/images
test: test/images

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = Path(dataset_path) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    print(f"\n   data.yaml created at: {yaml_path}")
    return yaml_path


def verify_conversion(dataset_path):
    """Quick sanity check after conversion."""
    print(f"\n{'═'*55}")
    print("  VERIFICATION")
    print(f"{'═'*55}")

    for split in SPLITS:
        split_path = Path(dataset_path) / split
        img_dir = split_path / 'images'
        lbl_dir = split_path / 'labels'

        if not img_dir.exists():
            print(f"  {split}: images/ folder missing")
            continue

        imgs = list(img_dir.iterdir())
        lbls = list(lbl_dir.iterdir()) if lbl_dir.exists() else []

        img_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        imgs = [f for f in imgs if f.suffix.lower() in img_exts]

        print(f"\n  [{split}]")
        print(f"    Images : {len(imgs)}")
        print(f"    Labels : {len(lbls)}")

        # Check a few label files
        matched = 0
        for img in imgs[:5]:
            lbl = lbl_dir / (img.stem + '.txt')
            if lbl.exists():
                matched += 1
        print(f"    Sample match (first 5): {matched}/5" if matched == 5 else f"    ⚠️ Only {matched}/5 labels matched")


def main():
    parser = argparse.ArgumentParser(description='Convert COCO dataset to YOLO format')
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to the root of your dataset (contains train/, valid/, test/)')
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f" Dataset path not found: {dataset_path}")
        return

    print(" COCO → YOLO Converter")
    print(f"   Dataset: {dataset_path.resolve()}")

    all_categories = None

    for split in SPLITS:
        split_dir = dataset_path / split

        # Handle alternate names
        if not split_dir.exists() and split == 'valid':
            alt = dataset_path / 'val'
            if alt.exists():
                split_dir = alt

        if not split_dir.exists():
            print(f"\n  '{split}' folder not found, skipping.")
            continue

        result = convert_split(split_dir, split)
        if result and result[0] is not None:
            _, _, cats = result
            if all_categories is None:
                all_categories = cats

    # Create data.yaml
    if all_categories:
        print(f"\n{'═'*55}")
        print("  Creating data.yaml")
        print(f"{'═'*55}")
        create_data_yaml(dataset_path, all_categories)

    # Verify
    verify_conversion(dataset_path)

    print(f"\n{'═'*55}")
    print("  CONVERSION COMPLETE!")
    print(f"{'═'*55}")
    print("\n Next step: Run create_subset.py to pick 500 images.")
    print(f"   python create_subset.py --dataset_path \"{dataset_path}\"")


if __name__ == '__main__':
    main()