# Defines single set abstraction layer for PointNet++. 
# It learns local features within a area of the point cloud 
# It first performs Farthest Point Sampling to retieve set of important points
# For this set, it retrieves sets of neighboring points via ball query function
# A shared Multi-Layer Perceptron (MLP) uses these grouped points, performs max pooling
# to reteieve a feature vector for a local area.

import torch
import torch.nn as nn
import torch.nn.functional as F

class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample

        layers = []
        last_channel = in_channel + 3
        for out_channel in mlp:
            layers.append(nn.Conv2d(last_channel, out_channel, 1))
            layers.append(nn.BatchNorm2d(out_channel))
            layers.append(nn.ReLU())
            last_channel = out_channel
        self.mlp_convs = nn.Sequential(*layers)

    def forward(self, xyz, points):
        new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1)
        new_points = self.mlp_convs(new_points)
        new_points = torch.max(new_points, 2)[0]
        new_points = new_points.permute(0, 2, 1)
        return new_xyz, new_points


# Defines the layers and architecture of the PointNet++ model
# It uses multiple PointNetSetAbstraction layers and puts them in a hierarchical order
# point cloud is downsamples and features are extracted at different scales 
# from local regions to global regions.
# feature learning, global max pooling puts all features into a representation of the entire point cloud, 
# this is then fed into fully connected layers to do the final classification
class PointNetPlusPlusClassifier(nn.Module):
    def __init__(self, num_classes, extra_features_dim=0):
        super().__init__()
        self.extra_features_dim = extra_features_dim
        
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.1, nsample=32, in_channel=0, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.2, nsample=64, in_channel=128, mlp=[128, 128, 256])
        self.sa3 = PointNetSetAbstraction(npoint=32, radius=0.4, nsample=128, in_channel=256, mlp=[256, 512, 1024])

        self.fc1 = nn.Linear(1024 + extra_features_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, xyz, extra_features=None):
        B, C, N = xyz.shape
        xyz = xyz.permute(0, 2, 1)
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)
        # global max pooling
        x = l3_points.max(dim=1)[0] 
        if self.extra_features_dim > 0 and extra_features is not None:
            # (B, 1024 + extra_features_dim)
            x = torch.cat([x, extra_features], dim=1)  

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.fc3(x)
        return x

#  helper function to get points from a group of point clouds based on indices
def index_points(points, idx):
    device = points.device
    B = points.shape[0]

    # Create B indices of shape (B, 1) or (B, 1, 1)
    view_shape = [B] + [1] * (idx.dim() - 1)
    batch_indices = torch.arange(B, dtype=torch.long, device=device).view(*view_shape)
    batch_indices = batch_indices.expand_as(idx)

    return points[batch_indices, idx]

# function to perform farthest point sampling (FPS)
# npoint points are selected from the point cloud
# FPS ensures selected points are maximally distant from each other, providing
# uniform coverage and efficient downsampling for hierarchical feature learning.
def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, _ = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=1)[1]
    
    return centroids


# Function to find all neightboring pixels within specified radius in 3D space. 
# It returns the indices of the `nsample` nearest neighbors for each query point.
def query_ball_point(radius, nsample, xyz, new_xyz):
    B, N, _ = xyz.shape
    S = new_xyz.shape[1]
    
    group_idx = torch.arange(N, device=xyz.device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]  # pick closest nsample

    # Handle insufficient points
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat(1, 1, nsample)
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

# Function to calculate the distance between 2 points in 3D space
def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

# Function to sample and group points in a point cloud done by performing farthest point sampling and querying neighboring points
def sample_and_group(npoint, radius, nsample, xyz, points):
    B, N, C = xyz.shape
    S = npoint
    # FPS (Farthest Point Sampling)
    fps_idx = farthest_point_sample(xyz, npoint)
    new_xyz = index_points(xyz, fps_idx)
    # Ball query (local neighborhood search)
    group_idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, group_idx)
    grouped_xyz -= new_xyz.unsqueeze(2)
    if points is not None:
        grouped_points = index_points(points, group_idx)
        new_points = torch.cat([grouped_xyz, grouped_points], dim=-1)
    else:
        new_points = grouped_xyz

    return new_xyz, new_points