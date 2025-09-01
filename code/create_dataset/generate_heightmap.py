'''
Code to generate height map.
Apply PCA for displaying height map correctly centered across axes.
Apply local filtering to remove extreme outliers caused by the scanner.
Filter all points within 7mm from the border.
'''

import numpy as np
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
from sklearn.decomposition import PCA
from scipy.interpolate import griddata
from scipy.ndimage import median_filter

# Load point cloud
pointcloud = PyntCloud.from_file("./dataset/pointcloud/00_117_box_5.ply")
# pointcloud = PyntCloud.from_file("./code/data/experiment_1/mannitol/pc/06_pc.ply")
# pointcloud = PyntCloud.from_file("./code/data/init_testing_scanner/metal_plate_pc.ply")
df = pointcloud.points

# Extract X, Y, Z
X, Y, Z = df["x"].values, df["y"].values, df["z"].values

# Apply PCA
points = np.column_stack((X, Y, Z))

#Do not apply PCA for augmentation of trainingsdata
# pca = PCA(n_components=3)
# pca.fit(points)
# # Transform points to align with axes
# # aligned_points = pca.transform(points)

aligned_points = points

# Flip transformed points around the horizontal axis and negate Z-values
aligned_points[:, 1] = -aligned_points[:, 1]
aligned_points[:, 2] = -aligned_points[:, 2]

# Border threshold that will be removed
border = 0

# Min/max in PCA space
pca_x_min, pca_x_max = aligned_points[:, 0].min(), aligned_points[:, 0].max()
pca_y_min, pca_y_max = aligned_points[:, 1].min(), aligned_points[:, 1].max()

# Filter points within border
filtered_mask = (
    (aligned_points[:, 0] > pca_x_min + border) & (aligned_points[:, 0] < pca_x_max - border) & 
    (aligned_points[:, 1] > pca_y_min + border) & (aligned_points[:, 1] < pca_y_max - border)
)
filtered_pca_xyz = aligned_points[filtered_mask]

# Extract filtered coordinates
X_filtered, Y_filtered, Z_filtered = filtered_pca_xyz[:, 0], filtered_pca_xyz[:, 1], filtered_pca_xyz[:, 2]

# Create height map using the filtered points
grid_x, grid_y = np.meshgrid(
    np.linspace(X_filtered.min(), X_filtered.max(), 500),
    np.linspace(Y_filtered.min(), Y_filtered.max(), 500)
)

grid_z = griddata((X_filtered, Y_filtered), Z_filtered, (grid_x, grid_y), method='cubic')

# Replace NaN values with median Z
grid_z = np.nan_to_num(grid_z, nan=np.nanmedian(Z_filtered))

# Median filter
grid_z_filtered = median_filter(grid_z, size=3)

# Flip Z values by negating them so they are displayed correctly
grid_z_filtered_flipped = grid_z_filtered

# Calculate surface roughness (Ra)
mean_height = np.nanmean(grid_z_filtered_flipped)
deviation_from_mean = np.abs(grid_z_filtered_flipped - mean_height)
Ra = np.mean(deviation_from_mean)

### Create a single figure with 3 subplots ###
fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# --- Figure 1: Point Cloud with Height Map Overlay ---
im1 = axes[0].scatter(X_filtered, Y_filtered, c=Z_filtered, cmap='jet', s=1, alpha=0.8, label="Filtered Points")
axes[0].scatter(aligned_points[~filtered_mask, 0], aligned_points[~filtered_mask, 1], c='lightgray', s=1, alpha=0.5, label="Removed Points")

# Draw bounding box
axes[0].plot([pca_x_min + border, pca_x_max - border, pca_x_max - border, pca_x_min + border, pca_x_min + border],
             [pca_y_min + border, pca_y_min + border, pca_y_max - border, pca_y_max - border, pca_y_min + border], 
             'w-', linewidth=2, label="Bounding Box")

fig.colorbar(im1, ax=axes[0], label="Height (Z)")
axes[0].set_xlabel("PCA X Coordinate")
axes[0].set_ylabel("PCA Y Coordinate")
axes[0].set_title("Point Cloud with Height Map Overlay")
axes[0].legend()
axes[0].axis("equal")

# Height map with interpolation
im2 = axes[1].imshow(grid_z_filtered_flipped, extent=(X_filtered.min(), X_filtered.max(), Y_filtered.min(), Y_filtered.max()),
                      origin="lower", cmap="jet")
fig.colorbar(im2, ax=axes[1], label="Height (Z)")
axes[1].set_xlabel("X Coordinate")
axes[1].set_ylabel("Y Coordinate")
axes[1].set_title(f"Interpolated Height Map\nSurface Roughness (Ra) = {Ra:.6f} mm")

# Height map scatter (so just the points)
im3 = axes[2].scatter(X_filtered, Y_filtered, c=-Z_filtered, cmap="jet", s=1, alpha=0.8)
fig.colorbar(im3, ax=axes[2], label="Height (Z)")
axes[2].set_xlabel("X Coordinate")
axes[2].set_ylabel("Y Coordinate")
axes[2].set_title("Scattered Height Map (raw point data)")

plt.tight_layout()
plt.show()
