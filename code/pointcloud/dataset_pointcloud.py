# This script defines a custom PyTorch Dataset for loading and preprocessing 3D pointcloud data.
# It reads pointclouds paths from an Excel file, loads ppointcloud data,
# resize them if necessary, and preparing them as tensors with their labels

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from plyfile import PlyData

class PointCloudDataset(Dataset):
    """
    - Dataset for loading 3D pointcloud data and their corresponding labels.
    - Read pointcloud file names and labels from an Excel file.
    - Load pointcloud from specified dir
    - Preprocess pointcloud resize and convert to tensors.
    """
    def __init__(self, pointcloud_dir, description_data, num_points, transform=None):
        self.pointcloud_dir = pointcloud_dir
        self.num_points = num_points
        self.transform = transform
        
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
        pc_filename_column = 'File_Name_PLY' 
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
        

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        pc_filename = row['File_Name_PLY']
        pc_path = os.path.join(self.ply_data_path, pc_filename)

        plydata = PlyData.read(pc_path)
        vertices = plydata['vertex']
        points = np.vstack([vertices[t] for t in ['x', 'y', 'z']]).T

        if points.shape[0] >= self.num_points:
            choice = np.random.choice(points.shape[0], self.num_points, replace=False)
        else:
            choice = np.random.choice(points.shape[0], self.num_points, replace=True)
        points = points[choice, :]

        if self.transform:
            points = self.transform(points)

        points = points - np.mean(points, axis=0)
        points = points / np.max(np.linalg.norm(points, axis=1))

        labels = row[self.label_cols].to_numpy(dtype=np.float32) 
        labels_tensor = torch.tensor(labels, dtype=torch.float32)

        # Return an empty tensor for additional_features, similar to your ImageDataset
        additional_features_tensor = torch.empty(0, dtype=torch.float32)

        return torch.tensor(points.T, dtype=torch.float32), additional_features_tensor, labels_tensor, pc_filename