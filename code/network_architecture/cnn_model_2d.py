# Simple Convolutional Neural Network (CNN) architecture designed for greyscale image classification using PyTorch. 
# The model uses convolutional layers, batch normalization, max pooling, global average pooling, and fully connected layers.
# where extra_features can be added

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleImageCNN(nn.Module):
    # Definez and initialize the layers  
    # Convolutional layers (nn.Conv2d) to extract features from grayscale images (1 input channel)
    # Batch normalization (nn.BatchNorm2d/1d) to stabilize and accelerate training
    # Max pooling layers (nn.MaxPool2d) to downsample and retain key features
    # Global average pooling (nn.AdaptiveAvgPool2d with output size (1, 1)) to reduce spatial dimensions
    # Fully connected layers (nn.Linear) to perform classification
    # Dropout layers (nn.Dropout) to reduce overfitting
    def __init__(self, num_labels, extra_features_dim=0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.extra_features_dim = extra_features_dim
        self.fc_input_dim = 128 + extra_features_dim
        self.fc1 = nn.Linear(self.fc_input_dim, 128)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(p=0.3)

        self.fc3 = nn.Linear(64, num_labels)

    # Defines forward function 
    def forward(self, x, extra_features):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x) 
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)

        if self.extra_features_dim > 0:
            # Add extra features
            x = torch.cat([x, extra_features], dim=1)

        x = F.relu(self.bn4(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x