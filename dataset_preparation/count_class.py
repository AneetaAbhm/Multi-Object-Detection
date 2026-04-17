import os
from pathlib import Path
from collections import defaultdict

dataset_path = r"         "   

SPLITS = ['train', 'valid', 'test']

print(" Counting images per class in original dataset...\n")

for split in SPLITS:
    labels_dir = Path(dataset_path) / split / 'labels'
    
    if not labels_dir.exists():
        print(f"  {split} labels folder not found!")
        continue
    
    class_count = defaultdict(int)
    total_images_with_labels = 0
    empty_labels = 0
    
    for label_file in labels_dir.glob("*.txt"):
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            empty_labels += 1
            continue
            
        total_images_with_labels += 1
        seen_in_this_image = set()
        
        for line in lines:
            if line.strip():
                class_id = int(line.strip().split()[0])
                seen_in_this_image.add(class_id)
        
        for cid in seen_in_this_image:
            class_count[cid] += 1   # Count image once per class (not per box)

    # Print results for this split
    print(f" === {split.upper()} SPLIT ===")
    print(f"   Total images with labels : {total_images_with_labels}")
    print(f"   Empty label files       : {empty_labels}\n")
    
    # Sort by number of images (descending)
    sorted_classes = sorted(class_count.items(), key=lambda x: x[1], reverse=True)
    
    for class_id, count in sorted_classes:
        print(f"   Class {class_id:2d}  →  {count:5d} images")
    
    print("-" * 60)

print("\n Done! This shows how many images contain each class.")
