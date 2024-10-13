#!/usr/bin/env python

import os
import numpy as np
from PIL import Image
import json
from tqdm import tqdm

# Paths
masks_dir = './output/training/masks'  # Directory with RV masks
annotations_dir = './output/training/annotations'  # Directory to save annotations
os.makedirs(annotations_dir, exist_ok=True)

# List of mask files
mask_files = sorted(os.listdir(masks_dir))


# Initialize list to store annotations
annotations = []

for mask_file in tqdm(mask_files, desc='Processing masks'):
    mask_path = os.path.join(masks_dir, mask_file)
    mask = Image.open(mask_path).convert('L')
    mask_np = np.array(mask)
    
    # Find coordinates where mask is non-zero
    indices = np.argwhere(mask_np > 128)
    
    if indices.size == 0:
        continue  # Skip if no RV present in the mask
    
    # Get bounding box coordinates
    y_min, x_min = indices.min(axis=0)
    y_max, x_max = indices.max(axis=0)
    
    # Width and height
    width = x_max - x_min
    height = y_max - y_min

    x_min = int(x_min)
    y_min = int(y_min)
    width = int(width)
    height = int(height)
    img_width = int(mask_np.shape[1])
    img_height = int(mask_np.shape[0])

    # Annotation format (example for COCO format)
    annotation = {
        'file_name': mask_file.replace('_mask.png', '.png'),
        'width': img_width,
        'height': img_height,
        'bbox': [x_min, y_min, width, height],
        'category_id': 1  # '1' corresponds to RV
    }
    annotations.append(annotation)

# Save annotations to a JSON file
with open(os.path.join(annotations_dir, 'annotations.json'), 'w') as f:
    json.dump(annotations, f)

# Directory to save YOLO annotations
yolo_labels_dir = './output/training/yolo_labels'
os.makedirs(yolo_labels_dir, exist_ok=True)

for annotation in annotations:
    img_width = annotation['width']
    img_height = annotation['height']
    x_min, y_min, box_width, box_height = annotation['bbox']
    
    # Calculate normalized coordinates
    x_center = (x_min + box_width / 2) / img_width
    y_center = (y_min + box_height / 2) / img_height
    width_norm = box_width / img_width
    height_norm = box_height / img_height
    
    # Create label file
    label_filename = annotation['file_name'].replace('.png', '.txt')
    label_path = os.path.join(yolo_labels_dir, label_filename)
    
    with open(label_path, 'w') as f:
        f.write(f"{annotation['category_id'] - 1} {x_center} {y_center} {width_norm} {height_norm}")
