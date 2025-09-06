# This script defines a custom PyTorch Dataset for loading and preprocessing 3D pointcloud data.
# It reads pointclouds paths from an Excel file, loads ppointcloud data,
# resize them if necessary, and preparing them as tensors with their labels

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from plyfile import PlyData
import open3d as o3d

#Number of neighbors for statistical outlier removal
STATISTICAL_OUTLIER_NB_NEIGHBORS = 40
#Standard deviation ratio for statistical outlier removal
STATISTICAL_OUTLIER_STD_RATIO = 1.5
#Number of points for radius outlier removal
RADIUS_OUTLIER_NB_POINTS = 30
#Radius for radius outlier removal in millimeters
RADIUS_OUTLIER_RADIUS_MM = 0.8
#Voxel size for downsampling
VOXEL_DOWNSAMPLING_SIZE_MM = 0.2

class PointCloudDataset(Dataset):
    """
    - Dataset for loading 3D pointcloud data and their corresponding labels.
    - Read pointcloud file names and labels from an Excel file.
    - Load pointcloud from specified dir
    - Preprocess pointcloud resize and convert to tensors.
    """
    def __init__(self, pointcloud_dir, description_data, num_points, transform=None, target_per_class=623, train=True):
        self.pointcloud_dir = pointcloud_dir
        self.num_points = num_points
        self.transform = transform
        self.train = train
        self.target_per_class = target_per_class
        
        self.metadata = pd.read_excel(description_data)
        
        self.label_cols = ['Good_layer', 'Ditch', 'Crater', 'Waves'] 
        
        # Path to ply folder
        self.ply_data_path = os.path.join(self.pointcloud_dir, 'ply')
        
        # Check if the 'ply' directory exists
        if not os.path.isdir(self.ply_data_path):
            raise FileNotFoundError(f"ply directory not found at: '{self.ply_data_path}'")

        # get all existing .ply files from ply subdirectory
        existing_pc_files = {f for f in os.listdir(self.ply_data_path) if f.endswith('.ply')}
        
        # the column name for PLY filenames in the metadata
        pc_filename_column = 'ply_file' 
        if pc_filename_column not in self.metadata.columns:
            raise ValueError(f"Metadata Excel file '{description_data}' is missing the expected point cloud filename column: '{pc_filename_column}'")

        # Filter to only include datapoints where the ply file exists
        self.metadata = self.metadata[self.metadata[pc_filename_column].isin(existing_pc_files)].reset_index(drop=True)
        
        # no valid samples remain
        if not len(self.metadata): 
             raise ValueError(f"no valid pointcloud files were found in '{self.ply_data_path}' that match entries in '{description_data}'")

        missing_label_cols = [col for col in self.label_cols if col not in self.metadata.columns]
        if missing_label_cols:
            raise ValueError(f"missing expected label columns in {description_data}: {missing_label_cols}, column '{self.label_cols[0]}'")
        
        self.metadata["aug_type"] = "none"

        # Balance dataset only for training and define possible augmentations
        if self.train:
            augmented_rows = []
            for class_idx, class_name in enumerate(self.label_cols):
                class_subset = self.metadata[self.metadata[class_name] == 1]
                current_count = len(class_subset)
                needed = max(0, self.target_per_class - current_count)

                if needed > 0 and len(class_subset) > 0:
                    for _ in range(needed):
                        row = class_subset.sample(n=1).iloc[0].copy()
                        row["aug_type"] = np.random.choice(["rotate180", "flip_x", "jitter", "combo"])
                        augmented_rows.append(row)

            if augmented_rows:
                self.metadata = pd.concat([self.metadata, pd.DataFrame(augmented_rows)], ignore_index=True)

        # Show label distribution (count how many samples per class)
        label_counts = self.metadata[self.label_cols].sum().to_dict()
        print(f"\n[{self.__class__.__name__}] Dataset label distribution:")
        for lbl, cnt in label_counts.items():
            print(f"  {lbl}: {int(cnt)} samples (target {self.target_per_class if self.train else 'N/A'})")
        print(f"  Total samples: {len(self.metadata)}\n")
        

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        pc_filename = row['ply_file']
        pc_path = os.path.join(self.ply_data_path, pc_filename)

        plydata = PlyData.read(pc_path)
        vertices = plydata['vertex']
        points = np.vstack([vertices[t] for t in ['x', 'y', 'z']]).T
        
        #Convert to pointcloud to do downsampling and outlier removal operations.
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        points = np.asarray(pcd.points)    

        if points.shape[0] >= self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.num_points, replace=True)
        points = points[choice, :]

        # Apply augmentation steps
        aug_type = row["aug_type"]
        if aug_type == "rotate180":
            R = np.array([[-1,0,0],[0,-1,0],[0,0,1]])
            points = points @ R.T
        elif aug_type == "flip_x":
            points[:,0] = -points[:,0]
        elif aug_type == "jitter":
            points += np.random.normal(0, 0.01, points.shape)
        elif aug_type == "combo":
            points[:,0] = -points[:,0]
            points += np.random.normal(0, 0.01, points.shape)

        if self.transform:
            points = self.transform(points)

        points = points - np.mean(points, axis=0)
        points = points / np.max(np.linalg.norm(points, axis=1))

        labels = row[self.label_cols].to_numpy(dtype=np.float32) 
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        # Return an empty tensor for additional_features, similar to your ImageDataset
        additional_features_tensor = torch.empty(0, dtype=torch.float32)

        return torch.tensor(points.T, dtype=torch.float32), additional_features_tensor, labels_tensor, pc_filename, aug_type