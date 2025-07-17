# This script is responsible for calulcating the global min and max z values over the whole dataset
# These global values are necesarry for consistent global normalization of the greyscale images
# 1. For each point cloud, plane detrending is applied to flatten the surface and remove any overall tilt,
# 2. border is removed to remove any noise/errors close to the edges of the point cloud
# 3. The final global minimum and maximum Z-values are saved to a text file for later
# Use requirements.txt to install the required packages

import os
import open3d as o3d
import numpy as np


#VARIABLES:
#Input folder containing point clouds
INPUT_FOLDER = "./data/pointclouds"
#The border in mm that is removed from the point cloud
BORDER_MM = 5
#Output file where the global min and max z values are saved
OUTPUT_FILE = "./dataset/z_min_max.txt"


def detrend_plane(pcd):
    """
    Removes a fitted plane from the Z-coordinates of the point cloud
    THis normalizes the surface and flattens the surface

    Args:
        pcd : Open3D point cloud.

    Returns:
        Detrended point cloud
    """
    points = np.asarray(pcd.points)

    if points.shape[0] < 3: 
        print("too little points")
        return pcd

    A = np.c_[points[:, 0], points[:, 1], np.ones(points.shape[0])]
    coeffs, _, _, _ = np.linalg.lstsq(A, points[:, 2], rcond=None)
    z_fit = A @ coeffs
    points[:, 2] = points[:, 2] - z_fit

    detrended = o3d.geometry.PointCloud()
    detrended.points = o3d.utility.Vector3dVector(points)
    return detrended

def filter_border_from_point_cloud(pcd):
    """
    Filters out points within a specified border distance bu constant variable from the min/max X '
    and Y bounds of the point cloud

    Args:
        pcd: The input Open3D point cloud.

    Returns:
        Filtered pointcloud with border removed 
    """
    points = np.asarray(pcd.points)
    if points.shape[0] == 0:
        print("pc empty after border removal")
        return o3d.geometry.PointCloud()

    x_coords = points[:, 0]
    y_coords = points[:, 1]
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    x_min_bound = min_x + BORDER_MM
    x_max_bound = max_x - BORDER_MM
    y_min_bound = min_y + BORDER_MM
    y_max_bound = max_y - BORDER_MM

    in_bounds = (x_coords >= x_min_bound) & (x_coords <= x_max_bound) & \
                (y_coords >= y_min_bound) & (y_coords <= y_max_bound)

    filtered_points = points[in_bounds]

    if filtered_points.shape[0] == 0:
        print("no points left after border removal.")
        return o3d.geometry.PointCloud()

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    return filtered_pcd

def find_global_min_max_z():
    global_min_z = float('inf')
    global_max_z = float('-inf')

    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".ply") or filename.endswith(".xyz") or filename.endswith(".pcd"):
            filepath = os.path.join(INPUT_FOLDER, filename)
            try:
                pcd = o3d.io.read_point_cloud(filepath)

                pcd = detrend_plane(pcd)

                pcd = filter_border_from_point_cloud(pcd)

                points = np.asarray(pcd.points)
                if points.shape[0] == 0:
                    continue

                z_values = points[:, 2]
                min_z = z_values.min()
                max_z = z_values.max()

                global_min_z = min(global_min_z, min_z)
                global_max_z = max(global_max_z, max_z)

                print(f"{filename} → min_z: {min_z:.3f}, max_z: {max_z:.3f}")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")

    return global_min_z, global_max_z

if __name__ == "__main__":
    min_z, max_z = find_global_min_max_z()
    print(f"\nglobal  min z value: {min_z}")
    print(f"global max z value: {max_z}")

    with open(OUTPUT_FILE, "w") as f:
        f.write(f"Global Min Z: {min_z}\n")
        f.write(f"Global Max Z: {max_z}\n")
