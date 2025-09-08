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
    def __init__(self, image_dir, description_data, target_size=(150, 150), transform=None, target_per_class=623, train=True):
        self.image_dir = image_dir
        self.target_size = target_size
        self.transform = transform
        self.train = train
        self.target_per_class = target_per_class
        

        self.metadata = pd.read_excel(description_data)
        
        existing_img_files = {f for f in os.listdir(self.image_dir) if f.endswith('.png')}
        self.metadata = self.metadata[self.metadata['png_file'].isin(existing_img_files)].reset_index(drop=True)

        if self.metadata.empty:
            raise ValueError("No valid image files found in the directory.")

        self.label_cols = ['Good_layer', 'Ditch', 'Crater', 'Waves']
        missing_label_cols = [col for col in self.label_cols if col not in self.metadata.columns]
        if missing_label_cols:
            raise ValueError(f"Missing label columns: {missing_label_cols}")

        self.metadata["aug_type"] = "none"

        if self.train:
            balanced_rows = []

            for class_name in self.label_cols:
                class_subset = self.metadata[self.metadata[class_name] == 1].copy()
                current_count = len(class_subset)

                if current_count >= self.target_per_class:
                    # Randomly sample target_per_class
                    balanced_rows.append(class_subset.sample(n=self.target_per_class, replace=False))
                else:
                    # Keep all existing and augment the rest
                    balanced_rows.append(class_subset)
                    needed = self.target_per_class - current_count
                    augmented_rows = []
                    for _ in range(needed):
                        row = class_subset.sample(n=1).iloc[0].copy()
                        row["aug_type"] = np.random.choice(["rotate180", "flip_x", "flip_y", "combo"])
                        augmented_rows.append(row)
                    if augmented_rows:
                        balanced_rows.append(pd.DataFrame(augmented_rows))

            # Combine all classes into final dataset
            self.metadata = pd.concat(balanced_rows, ignore_index=True)

        self.image_files = self.metadata['png_file'].tolist()

        # Show dataset distribution
        label_counts = self.metadata[self.label_cols].sum().to_dict()
        print(f"\n[{self.__class__.__name__}] Dataset label distribution:")
        for lbl, cnt in label_counts.items():
            print(f"  {lbl}: {int(cnt)} samples (target {self.target_per_class if self.train else 'N/A'})")
        print(f"  Total samples: {len(self.metadata)}\n")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_filename = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_filename) 

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if image is None:
            raise FileNotFoundError(f"image could not be found {img_path}")

        image = cv2.resize(image, (self.target_size[1], self.target_size[0]), interpolation=cv2.INTER_AREA)

        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0) 

        image_tensor = torch.from_numpy(image)

        aug_type = row["aug_type"]
        if aug_type == "rotate180":
            image_tensor = torch.rot90(image_tensor, 2, [1,2])
        elif aug_type == "flip_x":
            image_tensor = torch.flip(image_tensor, [1])
        elif aug_type == "flip_y":
            image_tensor = torch.flip(image_tensor, [2])
        elif aug_type == "combo":
            image_tensor = torch.flip(image_tensor, [1])
            image_tensor = torch.rot90(image_tensor, 2, [1,2])


        labels = self.metadata.iloc[idx][self.label_cols].to_numpy(dtype=np.float32)
        label_tensor = torch.tensor(labels, dtype=torch.float32)

        additional_features_tensor = torch.empty(0, dtype=torch.float32)

        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, additional_features_tensor, label_tensor, img_filename, aug_type