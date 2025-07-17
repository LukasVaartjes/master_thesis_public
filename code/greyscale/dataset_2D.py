# This script defines a custom PyTorch Dataset for loading and preprocessing 2D image data.
# It reads image file paths from an Excel file, loads grayscale images,
# resize them if necessary, and preparing them as tensors with their labels

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2

class ImageDataset(Dataset):
    """
    - Dataset for loading 2D grayscale images and their corresponding labels.
    - Read image file names and labels from an Excel file.
    - Load images from specified dir
    - Preprocess images resize and convert to tensors.
    """
    def __init__(self, image_dir, description_data, target_size=(150, 150), transform=None):
        self.image_dir = image_dir 
        self.target_size = target_size
        self.transform = transform

        self.metadata = pd.read_excel(description_data)

        self.image_files = self.metadata['File_Name_PNG'].tolist()

        self.png_data_path = os.path.join(self.image_dir, 'png') 
        
        if not os.path.isdir(self.png_data_path):
            raise FileNotFoundError(f"png dir not found at: '{self.png_data_path}'")

        existing_image_files = set(os.listdir(self.png_data_path)) 
        
        self.metadata = self.metadata[self.metadata['File_Name_PNG'].isin(existing_image_files)].reset_index(drop=True)
        self.image_files = self.metadata['File_Name_PNG'].tolist() 

        if not self.image_files:
            raise ValueError(f"no valid image files were found in '{self.png_data_path}' that match entries in '{description_data}' after filtering")

        self.label_cols = ['label'] 
        
        missing_label_cols = [col for col in self.label_cols if col not in self.metadata.columns]
        if missing_label_cols:
            raise ValueError(f"missing expected label columns in {description_data}: {missing_label_cols}, column '{self.label_cols[0]}' does not exists.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_filename = self.image_files[idx]
        img_path = os.path.join(self.png_data_path, img_filename) 

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"image could not be found {img_path}")

        image = cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)

        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0) 

        image_tensor = torch.from_numpy(image)

        labels = self.metadata.iloc[idx][self.label_cols].to_numpy(dtype=np.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        additional_features_tensor = torch.empty(0, dtype=torch.float32)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, additional_features_tensor, label_tensor, img_filename