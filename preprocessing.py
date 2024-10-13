#!/usr/bin/env python3

import os
import numpy as np
import nibabel as nib
import cv2
from tqdm import tqdm

# Define the paths to the dataset
DATASET_ROOT = './database'  # Update this to your dataset root path
OUTPUT_ROOT = './output'         # Update this to your desired output root path

# Define subfolders
DATASET_SUBFOLDERS = ['training', 'testing']

# Create output directories if they don't exist
for subset in DATASET_SUBFOLDERS:
    os.makedirs(os.path.join(OUTPUT_ROOT, subset, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, subset, 'masks'), exist_ok=True)

# Iterate over 'training' and 'testing' subsets
for subset in DATASET_SUBFOLDERS:
    subset_path = os.path.join(DATASET_ROOT, subset)
    output_image_path = os.path.join(OUTPUT_ROOT, subset, 'images')
    output_mask_path = os.path.join(OUTPUT_ROOT, subset, 'masks')
    
    # List all patient directories in the subset
    patient_dirs = [d for d in os.listdir(subset_path) if os.path.isdir(os.path.join(subset_path, d))]
    
    # Loop over all patient directories
    for patient_id in tqdm(patient_dirs, desc=f'Processing {subset} patients'):
        patient_path = os.path.join(subset_path, patient_id)
        
        # Read the info.cfg file to get ED and ES frame numbers
        info_cfg_path = os.path.join(patient_path, 'Info.cfg')
        if not os.path.exists(info_cfg_path):
            print(f"Info.cfg not found for patient {patient_id} in {subset}. Skipping.")
            continue
        
        # Read ED and ES frame numbers from the info.cfg file
        with open(info_cfg_path, 'r') as f:
            lines = f.readlines()
            ed_frame = None
            es_frame = None
            for line in lines:
                if line.startswith('ED:'):
                    ed_frame = int(line.strip().split(':')[-1])
                elif line.startswith('ES:'):
                    es_frame = int(line.strip().split(':')[-1])
            if ed_frame is None or es_frame is None:
                print(f"ED or ES frame not found in Info.cfg for patient {patient_id} in {subset}. Skipping.")
                continue
        
        # Prepare file names for ED and ES frames
        ed_image_file = f"{patient_id}_frame{ed_frame:02d}.nii.gz"
        ed_label_file = f"{patient_id}_frame{ed_frame:02d}_gt.nii.gz"
        es_image_file = f"{patient_id}_frame{es_frame:02d}.nii.gz"
        es_label_file = f"{patient_id}_frame{es_frame:02d}_gt.nii.gz"
        
        # List of frames to process
        frames = [
            {'phase': 'ED', 'image_file': ed_image_file, 'label_file': ed_label_file},
            {'phase': 'ES', 'image_file': es_image_file, 'label_file': es_label_file},
        ]
        
        # Process each frame (ED and ES)
        for frame in frames:
            image_path = os.path.join(patient_path, frame['image_file'])
            label_path = os.path.join(patient_path, frame['label_file'])
            
            # Check if files exist
            if not os.path.exists(image_path) or not os.path.exists(label_path):
                print(f"Missing image or label file for patient {patient_id}, phase {frame['phase']} in {subset}. Skipping.")
                continue
            
            # Load the MRI image and label volumes
            image_nii = nib.load(image_path)
            label_nii = nib.load(label_path)
            
            image_data = image_nii.get_fdata()
            label_data = label_nii.get_fdata()
            
            # Ensure the dimensions match
            if image_data.shape != label_data.shape:
                print(f"Image and label shapes do not match for patient {patient_id}, phase {frame['phase']} in {subset}. Skipping.")
                continue
            
            # Loop over the slices in the volume
            num_slices = image_data.shape[2]
            for slice_idx in range(num_slices):
                image_slice = image_data[:, :, slice_idx]
                label_slice = label_data[:, :, slice_idx]
                
                # Extract the RV mask (label 1)
                rv_mask = (label_slice == 1).astype(np.uint8)
                
                # Skip slices without RV segmentation to save space
                if rv_mask.sum() == 0:
                    continue
                
                # Normalize the image slice to [0, 255] and convert to uint8
                image_slice_norm = cv2.normalize(image_slice, None, 0, 255, cv2.NORM_MINMAX)
                image_slice_uint8 = image_slice_norm.astype(np.uint8)
                
                # Optionally resize images and masks to a consistent size (e.g., 256x256)
                desired_size = (256, 256)
                image_resized = cv2.resize(image_slice_uint8, desired_size, interpolation=cv2.INTER_AREA)
                mask_resized = cv2.resize(rv_mask, desired_size, interpolation=cv2.INTER_NEAREST)
                
                # Save the image and mask
                image_filename = f"{patient_id}_{frame['phase']}_slice_{slice_idx}.png"
                mask_filename = f"{patient_id}_{frame['phase']}_slice_{slice_idx}_mask.png"
                
                cv2.imwrite(os.path.join(output_image_path, image_filename), image_resized)
                cv2.imwrite(os.path.join(output_mask_path, mask_filename), mask_resized * 255)  # Multiply mask by 255 to make it visible
