#!/usr/bin/env python3

import cv2
import numpy as np
from pathlib import Path
import shutil
from typing import List, Tuple, Optional
import yaml

CONFIG = {
    'source_dataset': 'solar-panel-infrared-images-5',
    'output_dataset': 'solar-panel-infrared-images-5-augmented',
    'num_augmentations': 3,
    'displacement_range': (10, 50),  # pixels
    'random_seed': 42,
    'include_originals': True,  # Include original images in output
    'max_images': None,  # Set to None for full dataset, or number for testing
}

def generate_perspective_transform(
    img_width: int,
    img_height: int,
    displacement_range: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate random perspective transformation matrix.

    Args:
        img_width: Image width in pixels
        img_height: Image height in pixels
        displacement_range: (min_displacement, max_displacement) in pixels

    Returns:
        M: 3x3 perspective transformation matrix
        src_points: Original corner points
        dst_points: Displaced corner points
    """
    # Source points (original corners)
    src_points = np.float32([
        [0, 0],                      # top-left
        [img_width-1, 0],            # top-right
        [img_width-1, img_height-1], # bottom-right
        [0, img_height-1]            # bottom-left
    ])

    # Destination points (randomly displaced corners)
    dst_points = src_points.copy()
    _, max_disp = displacement_range

    for i in range(4):
        dx = np.random.randint(-max_disp, max_disp + 1)
        dy = np.random.randint(-max_disp, max_disp + 1)
        # Clamp to stay within reasonable bounds
        dst_points[i, 0] = np.clip(dst_points[i, 0] + dx, 0, img_width - 1)
        dst_points[i, 1] = np.clip(dst_points[i, 1] + dy, 0, img_height - 1)

    M = cv2.getPerspectiveTransform(src_points, dst_points)
    return M, src_points, dst_points


def transform_image(image: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    Apply perspective transformation to image.

    Args:
        image: Input image (H, W, C)
        M: 3x3 perspective transformation matrix

    Returns:
        Transformed image
    """
    h, w = image.shape[:2]
    transformed = cv2.warpPerspective(
        image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )
    return transformed


def transform_yolo_bbox(
    bbox: List[float],
    M: np.ndarray,
    img_width: int,
    img_height: int
) -> Optional[List[float]]:
    """
    Transform YOLO normalized bbox through perspective transformation.

    Args:
        bbox: [x_center, y_center, width, height] normalized [0,1]
        M: 3x3 perspective transformation matrix
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        Transformed bbox in YOLO format or None if invalid
    """
    x_center, y_center, w, h = bbox
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    w_px = w * img_width
    h_px = h * img_height

    x_min = x_center_px - w_px / 2
    y_min = y_center_px - h_px / 2
    x_max = x_center_px + w_px / 2
    y_max = y_center_px + h_px / 2

    corners = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ], dtype=np.float32)

    corners_reshaped = corners.reshape(1, -1, 2)
    transformed_corners = cv2.perspectiveTransform(corners_reshaped, M)
    transformed_corners = transformed_corners.reshape(-1, 2)

    # Compute bounding box
    x_min_new = np.min(transformed_corners[:, 0])
    y_min_new = np.min(transformed_corners[:, 1])
    x_max_new = np.max(transformed_corners[:, 0])
    y_max_new = np.max(transformed_corners[:, 1])

    # clip image to bounds
    x_min_new = np.clip(x_min_new, 0, img_width - 1)
    y_min_new = np.clip(y_min_new, 0, img_height - 1)
    x_max_new = np.clip(x_max_new, 0, img_width - 1)
    y_max_new = np.clip(y_max_new, 0, img_height - 1)

    # check if box is valid (has area)
    new_width = x_max_new - x_min_new
    new_height = y_max_new - y_min_new

    if new_width < 1 or new_height < 1:
        return None  # Box too small or invalid

    x_center_new = (x_min_new + x_max_new) / 2 / img_width
    y_center_new = (y_min_new + y_max_new) / 2 / img_height
    w_new = new_width / img_width
    h_new = new_height / img_height

    return [x_center_new, y_center_new, w_new, h_new]


def load_yolo_labels(label_path: Path) -> List[Tuple[int, List[float]]]:
    """
    Load YOLO format labels from file.

    Args:
        label_path: Path to .txt label file

    Returns:
        List of (class_id, [x_center, y_center, width, height])
    """
    labels = []
    if not label_path.exists():
        return labels

    with open(label_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            class_id = int(parts[0])
            bbox = [float(x) for x in parts[1:5]]
            labels.append((class_id, bbox))

    return labels


def save_yolo_labels(labels: List[Tuple[int, List[float]]], output_path: Path):
    """
    Save YOLO format labels to file.

    Args:
        labels: List of (class_id, [x_center, y_center, width, height])
        output_path: Output .txt file path
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for class_id, bbox in labels:
            bbox_str = ' '.join(f'{x:.6f}' for x in bbox)
            f.write(f'{class_id} {bbox_str}\n')


def augment_image_with_labels(
    image_path: Path,
    label_path: Path,
    output_dir_images: Path,
    output_dir_labels: Path,
    aug_index: int,
    config: dict
) -> bool:
    """
    Augment single image and its labels.

    Args:
        image_path: Path to source image
        label_path: Path to source label file
        output_dir_images: Output directory for images
        output_dir_labels: Output directory for labels
        aug_index: Augmentation index (0, 1, 2, ...)
        config: Configuration dictionary

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read image
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"  Warning: Could not read {image_path.name}")
            return False

        img_height, img_width = image.shape[:2]

        # Load labels
        labels = load_yolo_labels(label_path)

        # Generate transformation
        M, _, _ = generate_perspective_transform(
            img_width,
            img_height,
            config['displacement_range']
        )

        # Transform image
        transformed_image = transform_image(image, M)

        # Transform labels
        transformed_labels = []
        for class_id, bbox in labels:
            transformed_bbox = transform_yolo_bbox(bbox, M, img_width, img_height)
            if transformed_bbox is not None:
                transformed_labels.append((class_id, transformed_bbox))

        # Save augmented image
        output_image_name = f"{image_path.stem}_aug{aug_index}{image_path.suffix}"
        output_image_path = output_dir_images / output_image_name
        cv2.imwrite(str(output_image_path), transformed_image)

        # Save augmented labels
        output_label_name = f"{label_path.stem}_aug{aug_index}.txt"
        output_label_path = output_dir_labels / output_label_name
        save_yolo_labels(transformed_labels, output_label_path)

        return True

    except Exception as e:
        print(f"  Error processing {image_path.name}: {e}")
        return False


def create_output_structure(source_root: Path, output_root: Path, config: dict):
    """
    Create output directory structure and copy data.yaml.

    Args:
        source_root: Source dataset root directory
        output_root: Output dataset root directory
        config: Configuration dictionary
    """
    # Create output directories
    (output_root / 'train' / 'images').mkdir(parents=True, exist_ok=True)
    (output_root / 'train' / 'labels').mkdir(parents=True, exist_ok=True)

    # Copy valid and test sets unchanged
    for split in ['valid', 'test']:
        source_split = source_root / split
        if source_split.exists():
            output_split = output_root / split
            if output_split.exists():
                shutil.rmtree(output_split)
            shutil.copytree(source_split, output_split)
            print(f"Copied {split} set unchanged")

    # Copy and update data.yaml
    source_yaml = source_root / 'data.yaml'
    output_yaml = output_root / 'data.yaml'

    if source_yaml.exists():
        with open(source_yaml, 'r') as f:
            data = yaml.safe_load(f)

        # Update paths to be relative to data.yaml location
        data['train'] = 'train/images'
        data['val'] = 'valid/images'
        data['test'] = 'test/images'

        # Add augmentation metadata
        data['augmented'] = True
        data['augmentation_method'] = 'perspective_transform'
        data['augmentations_per_image'] = config['num_augmentations']

        with open(output_yaml, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        print(f"Created {output_yaml.name}")


def process_train_set(source_root: Path, output_root: Path, config: dict):
    """
    Process all training images and generate augmented dataset.

    Args:
        source_root: Source dataset root directory
        output_root: Output dataset root directory
        config: Configuration dictionary
    """
    # Set random seed for reproducibility
    np.random.seed(config['random_seed'])

    # Get paths
    source_images_dir = source_root / 'train' / 'images'
    source_labels_dir = source_root / 'train' / 'labels'
    output_images_dir = output_root / 'train' / 'images'
    output_labels_dir = output_root / 'train' / 'labels'

    # Get all image files
    image_files = sorted(source_images_dir.glob('*.jpg'))

    # Limit number of images if specified
    if config.get('max_images') is not None:
        image_files = image_files[:config['max_images']]
        print(f"Processing first {len(image_files)} images for testing\n")
    else:
        print(f"Processing {len(image_files)} images\n")

    # Process each image
    total_images = len(image_files)
    success_count = 0

    for idx, image_path in enumerate(image_files, 1):
        print(f"[{idx}/{total_images}] Processing {image_path.name}")

        # Get corresponding label file
        label_path = source_labels_dir / f"{image_path.stem}.txt"

        # Copy original if configured
        if config['include_originals']:
            shutil.copy2(image_path, output_images_dir / image_path.name)
            if label_path.exists():
                shutil.copy2(label_path, output_labels_dir / label_path.name)
            else:
                # Create empty label file for background images
                (output_labels_dir / label_path.name).touch()

        # Generate augmentations
        for aug_idx in range(config['num_augmentations']):
            success = augment_image_with_labels(
                image_path,
                label_path,
                output_images_dir,
                output_labels_dir,
                aug_idx,
                config
            )
            if success:
                success_count += 1

        print(f"  Generated {config['num_augmentations']} augmentations")

    # Summary
    expected_originals = total_images if config['include_originals'] else 0
    expected_augmented = total_images * config['num_augmentations']
    expected_total = expected_originals + expected_augmented

    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"{'='*60}")
    print(f"Source images processed: {total_images}")
    print(f"Augmentations per image: {config['num_augmentations']}")
    print(f"Include originals: {config['include_originals']}")
    print(f"Expected output images: {expected_total}")
    print(f"  - Originals: {expected_originals}")
    print(f"  - Augmented: {expected_augmented}")
    print(f"{'='*60}")


def main():
    print("="*60)
    print("Perspective Transformation Data Augmentation")
    print("="*60)
    print(f"Source dataset: {CONFIG['source_dataset']}")
    print(f"Output dataset: {CONFIG['output_dataset']}")
    print(f"Augmentations per image: {CONFIG['num_augmentations']}")
    print(f"Displacement range: {CONFIG['displacement_range']} pixels")
    print(f"Random seed: {CONFIG['random_seed']}")
    print(f"Include originals: {CONFIG['include_originals']}")
    if CONFIG.get('max_images'):
        print(f"Max images (test mode): {CONFIG['max_images']}")
    print("="*60)
    print()

    script_dir = Path(__file__).parent
    source_root = script_dir / CONFIG['source_dataset']
    output_root = script_dir / CONFIG['output_dataset']

    if not source_root.exists():
        print(f"Error: Source dataset not found at {source_root}")
        return

    print("Creating output directory structure...")
    create_output_structure(source_root, output_root, CONFIG)
    print()

    print("Processing training set...")
    process_train_set(source_root, output_root, CONFIG)

    print(f"\nAugmented dataset created at: {output_root}")
    print("\nDone!")


if __name__ == '__main__':
    main()
