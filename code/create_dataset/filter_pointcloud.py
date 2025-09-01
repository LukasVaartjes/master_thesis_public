import glob
import os
import open3d as o3d

# Filtering parameters
VOXEL_DOWNSAMPLING_SIZE_MM = 0.2
STATISTICAL_OUTLIER_NB_NEIGHBORS = 10 
STATISTICAL_OUTLIER_STD_RATIO = 2.5  
RADIUS_OUTLIER_NB_POINTS = 5  
RADIUS_OUTLIER_RADIUS_MM = 1.0 

POINTCLOUD_FOLDER = "./dataset/pointcloud"

def apply_filtering_pointcloud(pcd):
    """Applies voxel downsampling, statistical outlier removal, and radius outlier removal"""
    if len(pcd.points) == 0:
        return pcd

    # Voxel downsampling
    if VOXEL_DOWNSAMPLING_SIZE_MM > 0:
        pcd = pcd.voxel_down_sample(VOXEL_DOWNSAMPLING_SIZE_MM)

    #Statistical outlier removal
    if STATISTICAL_OUTLIER_NB_NEIGHBORS > 0:
        pcd, ind = pcd.remove_statistical_outlier(
            nb_neighbors=STATISTICAL_OUTLIER_NB_NEIGHBORS,
            std_ratio=STATISTICAL_OUTLIER_STD_RATIO
        )

    #Radius outlier removal
    if RADIUS_OUTLIER_NB_POINTS > 0:
        pcd, ind = pcd.remove_radius_outlier(
            nb_points=RADIUS_OUTLIER_NB_POINTS,
            radius=RADIUS_OUTLIER_RADIUS_MM
        )
    print(f"Points remaining after filtering: {len(pcd.points)}")
    return pcd

# Loop over all point clouds in the folder and apply fitlering
for file_path in glob.glob(os.path.join(POINTCLOUD_FOLDER, "*.ply")):
    print(f"Processing: {file_path}")
    pcd = o3d.io.read_point_cloud(file_path)
    filtered_pcd = apply_filtering_pointcloud(pcd)

    # Save with same name to overwrite old file
    o3d.io.write_point_cloud(file_path, filtered_pcd)
    print(f"Saved filtered point cloud: {file_path}\n")
