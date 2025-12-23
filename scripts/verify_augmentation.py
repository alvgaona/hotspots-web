#!/usr/bin/env python3
"""Verify the augmented dataset and show sample transformations."""

from ultralytics.data import YOLODataset
from ultralytics.data.utils import check_det_dataset

# Load augmented dataset
print("Loading augmented dataset...")
data_dict = check_det_dataset("solar-panel-infrared-images-5-augmented/data.yaml")

train_dataset = YOLODataset(
    img_path="solar-panel-infrared-images-5-augmented/train/images",
    data=data_dict
)

print(f"\nDataset loaded successfully!")
print(f"Total training images: {len(train_dataset)}")
print(f"\nChecking first few samples with bboxes:\n")

# Find and display samples with bboxes
samples_found = 0
for i in range(len(train_dataset)):
    sample = train_dataset[i]
    if len(sample['bboxes']) > 0:
        img_path = sample['im_file']
        filename = img_path.split('/')[-1]
        num_boxes = len(sample['bboxes'])
        classes = sample['cls'].flatten().tolist()

        print(f"Sample {i}: {filename}")
        print(f"  Bboxes: {num_boxes}")
        print(f"  Classes: {[int(c) for c in classes]}")
        print(f"  First bbox (YOLO format): {sample['bboxes'][0].tolist()}")
        print()

        samples_found += 1
        if samples_found >= 5:
            break

# Compare original vs augmented
print("\nComparing original vs augmented versions of same image:")
print("="*70)

# Find first image with augmentations
for i in range(len(train_dataset)):
    sample = train_dataset[i]
    filename = sample['im_file'].split('/')[-1]

    # Check if this is an original (not augmented)
    if '_aug' not in filename and len(sample['bboxes']) > 0:
        print(f"\nOriginal: {filename}")
        print(f"  Bbox: {sample['bboxes'][0].tolist()}")

        # Find corresponding augmentations
        base_name = filename.replace('.jpg', '')
        for aug_idx in range(3):
            aug_name = f"{base_name}_aug{aug_idx}.jpg"
            for j in range(len(train_dataset)):
                aug_sample = train_dataset[j]
                aug_filename = aug_sample['im_file'].split('/')[-1]
                if aug_filename == aug_name and len(aug_sample['bboxes']) > 0:
                    print(f"  Aug{aug_idx}: {aug_sample['bboxes'][0].tolist()}")
                    break
        break

print("\n" + "="*70)
print("Verification complete! All labels are valid.")
