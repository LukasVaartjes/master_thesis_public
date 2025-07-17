# This script processes 3D point cloud data (.ply files) from a specified input folder (INPUT_FOLDER) variable.
# It performs the following operations:
# 1. Loads point clouds and converts them to Open3D format
# 2. Detrends the Z-values to remove any global plane tilt
# 3. Trims an initial border from the point cloud to focus on the central area and remove noise/errors at the edges
# 4. Applies statistical and radius outlier removal filters to clean the point cloud data and reduce nr of points while preserving structure
# 5. Converts the processed 3D point cloud data into a 2D greyscale image representation based on Z-values
# 6. Extracts fixed-size 2D bounding box segments (both point cloud and greyscale image) from the point cloud, based on predefined offsets
# 7. Visualizes the placement of these boxes on a combined greyscale image and point cloud plot
# 8. Saves the extracted point cloud segments and greyscale image segments to specified output directories + description features in excel file
# The script relies on a pre-calculated global Z-min/max for consistent greyscale normalization
# Use requirements.txt to install the required packages

import glob
import os
from pyntcloud import PyntCloud
import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

#Base directory for all generated output data
OUTPUT_FOLDER = "./dataset"
#Base directory containing raw input point cloud files.
INPUT_FOLDER = "./data"
#Path for Excel file to describe point cloud outputs
EXCEL_POINTCLOUD_OUTPUT_PATH = "./dataset/description_pointcloud.xlsm"
#Path for Excel file to describe greyscale outputs
EXCEL_GREYSCALE_OUTPUT_PATH = "./dataset/description_greyscale.xlsm"
#Text file containing pre-calculated global min/max Z-values for normalization
GLOBAL_Z_VALUES_TXT = "./dataset/z_min_max.txt" # Corrected path to be a file
#The side length in millimeters for each square bounding box segment
BOX_SIZE_MM = 10
#The width of the border (in mm) to remove from the point cloud's border
INITIAL_BORDER_TRIM_MM = 5
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
#Maximum allowed overlap between randomly placed boxes
MAX_BOX_OVERLAP_RATIO = 0
# Minimum distance from the trimmed point cloud's edge where boxes can be placed
MIN_BORDER_FOR_BOX_PLACEMENT_MM = 0
#Defines the spatial resolution (mm per pixel) when converting point clouds to images
RESOLUTION = 0.1
#Variable to store the globally determined minimum Z-value for consistent greyscale scaling
GLOBAL_Z_MIN = None
#Variable to store the globally determined maximum Z-value for consistent greyscale scaling
GLOBAL_Z_MAX = None
#The desired final pixel dimensions (width, height) for the full greyscale image.
TARGET_IMAGE_SIZE = (400, 700)
#The desired final pixel dimensions (width, height) for each extracted greyscale segment.
TARGET_SEGMENT_SIZE = (150, 150)
#A list of (x_offset, y_offset) tuples in millimeters, defines he starting point of a fixed 10x10mm box
FIXED_BOX_OFFSETS_MM = [
    (1, 1),    # Box 0
    (1, 12),   # Box 1
    (1, 23),   # Box 2
    (1, 34),   # Box 3

    (13, 1),   # Box 4
    (13, 12),  # Box 5
    (13, 23),  # Box 6
    (13, 34)   # Box 7
]
#Automatically set based on the number of defined fixed box offsets.
NUMBER_OF_BOXES_PER_POINTCLOUD = len(FIXED_BOX_OFFSETS_MM)


def load_pointcloud(name_data):
    """
    Load point cloud from a .ply file, converts to an Open3D object
    detrends it, and then performs border trim

    Args:
        name_data (str): Filename of the point cloud

    Returns:
        Detrended, and border trim point cloud
    """
    filepath = f"{INPUT_FOLDER}/pointclouds/{name_data}"
    print(f"Loading point cloud data from: {filepath}")
    pointcloud = PyntCloud.from_file(filepath)
    df = pointcloud.points

    # Convert to Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(df[["x", "y", "z"]].values)

    pcd = detrend_plane(pcd)

    # Remove border from pointcloud immediately
    filtered_pcd = filter_border_from_point_cloud(pcd)

    print(f"Trimmed outer {INITIAL_BORDER_TRIM_MM}mm border, kept total of {len(filtered_pcd.points)} points, removed {len(df) - len(filtered_pcd.points)} points")
    return filtered_pcd


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
        print("Dimension of pc is too small")
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
        print("no points in point cloud to filter border from")
        return o3d.geometry.PointCloud()

    x_coords = points[:, 0]
    y_coords = points[:, 1]

    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)

    x_min_bound = min_x + INITIAL_BORDER_TRIM_MM
    x_max_bound = max_x - INITIAL_BORDER_TRIM_MM
    y_min_bound = min_y + INITIAL_BORDER_TRIM_MM
    y_max_bound = max_y - INITIAL_BORDER_TRIM_MM

    in_bounds = (x_coords >= x_min_bound) & (x_coords <= x_max_bound) & \
                (y_coords >= y_min_bound) & (y_coords <= y_max_bound)

    filtered_points = points[in_bounds]

    if filtered_points.shape[0] == 0:
        print("no points left after removal of border")
        return o3d.geometry.PointCloud()

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)
    return filtered_pcd

def load_min_max_z():
    """
    Loads pre-calculated global minimum and maximum Z-values from text file
    These global values are used to normalize greyscale images consistently across different point clouds
    No local constrast adjustment is done, as this removes the signal in the data
    Returns:
        tuple: A tuple containing (global_min_z, global_max_z)
    """
    global_min_z = None
    global_max_z = None
    try:
        # Corrected path to include the filename
        with open(GLOBAL_Z_VALUES_TXT, 'r', encoding='utf-8-sig') as f:
            for line in f:
                if "Global Min Z:" in line:
                    global_min_z = float(line.split(":")[1].strip())
                    print(f"Global Min Z loaded: {global_min_z}")
                elif "Global Max Z:" in line:
                    global_max_z = float(line.split(":")[1].strip())
                    print(f"Global Max Z loaded: {global_max_z}")
    except FileNotFoundError:
        print(f"file not found at {GLOBAL_Z_VALUES_TXT}. Please ensure this file exists with global Z values.")
    except Exception as e:
        print(f"Error during reading {GLOBAL_Z_VALUES_TXT}: {e}")
    return global_min_z, global_max_z


def point_cloud_to_image(pcd):
    """
    Converts a 3D point cloud to a 2D greyscale image. Intensity of each pixel
    is averages according to the z-values of the points
    Image is then normalized according to the global Z min/max values

    Args:
        pcd The input Open3D point cloud

    Returns:
            - The 2D greyscale image
            - X-offset (minimum X-coordinate of the point cloud)
            - Y-offset (minimum Y-coordinate of the point cloud)
            - Original width (in pixels) of the image before resizing
            - Original height (in pixels) of the image before resizing
    """
    points = np.asarray(pcd.points)

    if len(points) == 0:
        print("Point cloud is empty, cannot convert to image.")
        # Return appropriate empty image and offsets based on whether global Z values are loaded
        if GLOBAL_Z_MIN is None and GLOBAL_Z_MAX is None:
            return np.zeros((0, 0), dtype=np.float32), 0, 0, 0, 0
        else:
            return np.zeros((TARGET_IMAGE_SIZE[1], TARGET_IMAGE_SIZE[0]), dtype=np.uint8), 0, 0, 0, 0


    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    x_offset = np.min(x)
    y_offset = np.min(y)
    x_coords_shifted = x - x_offset
    y_coords_shifted = y - y_offset

    img_w_unscaled = int(np.ceil(np.max(x_coords_shifted) / RESOLUTION)) + 1
    img_h_unscaled = int(np.ceil(np.max(y_coords_shifted) / RESOLUTION)) + 1

    # Ensure image dimensions are at least 1x1 to prevent errors with cv2.resize later
    img_w_unscaled = max(1, img_w_unscaled)
    img_h_unscaled = max(1, img_h_unscaled)

    img = np.zeros((img_h_unscaled, img_w_unscaled), dtype=np.float32)
    count = np.zeros((img_h_unscaled, img_w_unscaled), dtype=np.int32)

    for xi_shifted, yi_shifted, zi in zip(x_coords_shifted, y_coords_shifted, z):
        i = int(yi_shifted / RESOLUTION)
        j = int(xi_shifted / RESOLUTION)

        if 0 <= i < img_h_unscaled and 0 <= j < img_w_unscaled:
            img[i, j] += zi
            count[i, j] += 1

    mask = count > 0

    # Avoid division by zero for pixels with no points
    img[mask] /= count[mask]

    if np.any(mask):
        min_val = np.min(img[mask])
        max_val = np.max(img[mask])
        print(f"min Z in raw float image: {min_val:.4f}")
        print(f"max Zin raw float image: {max_val:.4f}")
    else:
        print("raw image is empty ")
        # Return an empty image of target size if no points, as it will be resized anyway
        return np.zeros(TARGET_IMAGE_SIZE[::-1], dtype=np.uint8), x_offset, y_offset, img_w_unscaled, img_h_unscaled

    if GLOBAL_Z_MIN is not None and GLOBAL_Z_MAX is not None:
        print("using global z values for normalization")
        final_image_for_display = np.zeros_like(img, dtype=np.uint8)
        # Ensure that GLOBAL_Z_MAX - GLOBAL_Z_MIN is not zero to prevent division by zero
        if (GLOBAL_Z_MAX - GLOBAL_Z_MIN) > 0:
            final_image_for_display = ((img - GLOBAL_Z_MIN) / (GLOBAL_Z_MAX - GLOBAL_Z_MIN) * 255).astype(np.uint8)
        else:
            # If min and max Z are the same, assign a mid-gray value
            final_image_for_display = np.full(img.shape, 128, dtype=np.uint8)
    # If no global z values, adjust contrast locally
    else:
        if np.any(mask):
            final_image_for_display = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        else:
            final_image_for_display = np.zeros_like(img, dtype=np.uint8)

    print(f"original image size (before target resize) {final_image_for_display.shape[1]}x{final_image_for_display.shape[0]}")

    # Store dimensions before resizing to TARGET_IMAGE_SIZE
    pre_resize_width = final_image_for_display.shape[1]
    pre_resize_height = final_image_for_display.shape[0]

    # Resize to TARGET_IMAGE_SIZE
    if final_image_for_display.shape[0] != TARGET_IMAGE_SIZE[1] or final_image_for_display.shape[1] != TARGET_IMAGE_SIZE[0]:
        print(f"resizing full image to {TARGET_IMAGE_SIZE[0]}x{TARGET_IMAGE_SIZE[1]}")
        final_image_for_display = cv2.resize(final_image_for_display, TARGET_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

    return final_image_for_display, x_offset, y_offset, pre_resize_width, pre_resize_height


def fixed_box_visualize(pcd_df, pc_id, full_img, x_offset_pc, y_offset_pc, pre_resize_width, pre_resize_height):
    """
    Extracts fixed-size point cloud and greyscale image segments based on predefined offsets.
    It also visualizes the segments in the greyscale image for easy referencing
    and saves segment metadata to an Excel file.

    Args:
        pcd_df: dataFrame of the point cloud (x, y, z columns) after preprocessing.
        pc_id: unique identifier for the current point cloud
        full_img: the full 2D greyscale image generated from the point cloud
        x_offset_pc: the minimum X-coordinate of the full point cloud
        y_offset_pc: the minimum Y-coordinate of the full point cloud
        pre_resize_width: the width of the greyscale image before TARGET_IMAGE_SIZE resizing
        pre_resize_height: the height of the greyscale image before TARGET_IMAGE_SIZE resizing
    """
    coords = pcd_df[["x", "y", "z"]].values

    # Get bound of the filtered point cloud
    min_x_pc_current, min_y_pc_current = np.min(coords[:, 0]), np.min(coords[:, 1])
    max_x_pc_current, max_y_pc_current = np.max(coords[:, 0]), np.max(coords[:, 1])

    # Adjust for the inner border for box placement
    min_bound_box_placement = np.array([min_x_pc_current + MIN_BORDER_FOR_BOX_PLACEMENT_MM,
                                          min_y_pc_current + MIN_BORDER_FOR_BOX_PLACEMENT_MM])
    max_bound_box_placement = np.array([max_x_pc_current - MIN_BORDER_FOR_BOX_PLACEMENT_MM,
                                          max_y_pc_current - MIN_BORDER_FOR_BOX_PLACEMENT_MM])

    os.makedirs(os.path.join(OUTPUT_FOLDER, "pointcloud"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "greyscale"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FOLDER, "visualizations"), exist_ok=True)
    os.makedirs(os.path.dirname(EXCEL_GREYSCALE_OUTPUT_PATH), exist_ok=True) # Ensure directory for Excel exists

    saved = 0
    greyscale_segment_data = [] # List to store data for Excel sheet

    fig_2d_viz, ax_2d_viz = plt.subplots(figsize=(12, 10))

    # Plot the full greyscale image as background, mapped to point cloud coordinates
    ax_2d_viz.imshow(full_img, cmap='gray', vmin=0, vmax=255, origin='lower',
                     extent=[x_offset_pc, x_offset_pc + (pre_resize_width * RESOLUTION),
                             y_offset_pc, y_offset_pc + (pre_resize_height * RESOLUTION)])

    # Plot the full point cloud for reference
    ax_2d_viz.scatter(coords[:, 0], coords[:, 1], s=0.1, c='blue', alpha=0.3, label='Full Point Cloud')

    # Create a single copy of the full greyscale image to draw all boxes on
    greyscale_viz_image_combined = full_img.copy()

    for i, (offset_x, offset_y) in enumerate(FIXED_BOX_OFFSETS_MM):
        box_min_mm = np.array([min_bound_box_placement[0] + offset_x, min_bound_box_placement[1] + offset_y])
        box_max_mm = box_min_mm + BOX_SIZE_MM

        # Check if the fixed box is within the allowed placement bounds
        if not (min_bound_box_placement[0] <= box_min_mm[0] and
                min_bound_box_placement[1] <= box_min_mm[1] and
                max_bound_box_placement[0] >= box_max_mm[0] and
                max_bound_box_placement[1] >= box_max_mm[1]):
            print(f"Segment {i} (min: {box_min_mm}, max: {box_max_mm}) is out of bounds for pc_id {pc_id}. Skipping.")
            continue

        print(f"Placing segment {i} for {pc_id}: box min {box_min_mm}, box max {box_max_mm}")

        mask = np.all((coords[:, :2] >= box_min_mm) & (coords[:, :2] <= box_max_mm), axis=1)
        box_points_in_mm = coords[mask]

        if len(box_points_in_mm) == 0:
            print(f"Segment {i} contains no points for pc_id {pc_id}. Skipping.")
            continue

        box_pcd_o3d = o3d.geometry.PointCloud()
        box_pcd_o3d.points = o3d.utility.Vector3dVector(box_points_in_mm)

        output_pointcloud_path = os.path.join(OUTPUT_FOLDER, "pointcloud", f"{pc_id}_box_{saved}.ply")
        if len(box_pcd_o3d.points) > 0:
            o3d.io.write_point_cloud(output_pointcloud_path, box_pcd_o3d)
            print(f"Saved point cloud segment with {len(box_pcd_o3d.points)} points to {output_pointcloud_path}")

            # Add the selected points to visualise plot in a different color
            ax_2d_viz.scatter(np.asarray(box_pcd_o3d.points)[:, 0], np.asarray(box_pcd_o3d.points)[:, 1],
                              s=2, c='red', alpha=0.8, label=f'Box {saved} Pts' if saved == 0 else "")
        else:
            print(f"Point cloud segment for box {saved} is empty for pc_id {pc_id}. Skipping.")
            continue

        box_min_shifted_mm = box_min_mm - np.array([x_offset_pc, y_offset_pc])
        box_max_shifted_mm = box_max_mm - np.array([x_offset_pc, y_offset_pc])

        box_min_unscaled_px = box_min_shifted_mm / RESOLUTION
        box_max_unscaled_px = box_max_shifted_mm / RESOLUTION

        scale_x = TARGET_IMAGE_SIZE[0] / pre_resize_width
        scale_y = TARGET_IMAGE_SIZE[1] / pre_resize_height

        box_min_target_px = (box_min_unscaled_px * np.array([scale_x, scale_y])).astype(int)
        box_max_target_px = (box_max_unscaled_px * np.array([scale_x, scale_y])).astype(int)

        # check coordinates are within bounds
        box_min_target_px[0] = max(0, box_min_target_px[0])
        box_min_target_px[1] = max(0, box_min_target_px[1])
        box_max_target_px[0] = min(TARGET_IMAGE_SIZE[0] - 1, box_max_target_px[0])
        box_max_target_px[1] = min(TARGET_IMAGE_SIZE[1] - 1, box_max_target_px[1])

        # make sure that box_max is at least box_min for slicing, or return 0
        box_max_target_px[0] = max(box_min_target_px[0] + 1, box_max_target_px[0])
        box_max_target_px[1] = max(box_min_target_px[1] + 1, box_max_target_px[1])

        # make sure box_max not greater than target image size
        box_max_target_px[0] = min(box_max_target_px[0], TARGET_IMAGE_SIZE[0])
        box_max_target_px[1] = min(box_max_target_px[1], TARGET_IMAGE_SIZE[1])

        # Extract the greyscale segment BEFORE resizing to TARGET_SEGMENT_SIZE
        greyscale_segment_original_slice = full_img[box_min_target_px[1]:box_max_target_px[1],
                                                    box_min_target_px[0]:box_max_target_px[0]]

        if greyscale_segment_original_slice.size > 0:
            # Store the original slice coordinates for the Excel sheet
            row_min_excel = box_min_target_px[1]
            col_min_excel = box_min_target_px[0]
            row_max_excel = box_max_target_px[1]
            col_max_excel = box_max_target_px[0]

            greyscale_segment = greyscale_segment_original_slice.copy() # Work on a copy

            # resize to 150x150
            if greyscale_segment.shape[0] != TARGET_SEGMENT_SIZE[1] or \
               greyscale_segment.shape[1] != TARGET_SEGMENT_SIZE[0]:
                print(f"resizing greyscale segment from {greyscale_segment.shape[1]}x{greyscale_segment.shape[0]} to {TARGET_SEGMENT_SIZE[0]}x{TARGET_SEGMENT_SIZE[1]}...")
                greyscale_segment = cv2.resize(greyscale_segment, TARGET_SEGMENT_SIZE, interpolation=cv2.INTER_AREA)

            output_greyscale_path = os.path.join(OUTPUT_FOLDER, "greyscale", f"{pc_id}_box_{saved}.png")
            cv2.imwrite(output_greyscale_path, greyscale_segment)
            print(f"Saved greyscale segment to {output_greyscale_path}")

            cv2.rectangle(greyscale_viz_image_combined, tuple(box_min_target_px), tuple(box_max_target_px), 255, 1)

            # set segment number in visualization for easy referencing
            text = str(saved)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            font_thickness = 1
            text_color = (255)
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = box_min_target_px[0] + 2
            text_y = box_min_target_px[1] + text_size[1] + 2

            cv2.putText(greyscale_viz_image_combined, text, (text_x, text_y),
                        font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            # Collect data for Excel sheet from the *resized* greyscale_segment
            min_pixel_value = np.min(greyscale_segment)
            max_pixel_value = np.max(greyscale_segment)
            pixel_value_difference = max_pixel_value - min_pixel_value

            greyscale_segment_data.append({
                "Original_Image_ID": pc_id,
                "Segment_ID_In_Original": saved,
                "Row_Min": row_min_excel,
                "Col_Min": col_min_excel,
                "Row_Max": row_max_excel,
                "Col_Max": col_max_excel,
                "Segment_Width": TARGET_SEGMENT_SIZE[0],
                "Segment_Height": TARGET_SEGMENT_SIZE[1],
                "Min_Pixel_Value": min_pixel_value,
                "Max_Pixel_Value": max_pixel_value,
                "Pixel_Value_Difference": pixel_value_difference,
                "Label (0 = no defect, 1 = defect)": "" 
            })

        else:
            print(f"Greyscale segment for box {saved} is empty for pc_id {pc_id}. Skipping.")
            continue

        # draw the box outline in the 2D visualization plot
        rect = plt.Rectangle(box_min_mm, BOX_SIZE_MM, BOX_SIZE_MM,
                             linewidth=1, edgecolor='red', facecolor='none', label=f'Box Outline' if saved == 0 else "")
        ax_2d_viz.add_patch(rect)
        ax_2d_viz.text(box_min_mm[0], box_min_mm[1] + BOX_SIZE_MM + 0.5, str(saved), color='red', fontsize=9, ha='center', va='bottom')

        saved += 1

    # save greyscale segment visualization
    if saved > 0:  # only save if at least 1 segment was created
        cv2.imwrite(os.path.join(OUTPUT_FOLDER, "visualizations", f"{pc_id}_greyscale_all_boxes_viz.png"), greyscale_viz_image_combined)
        print(f"Saved combined greyscale box visualization to {os.path.join(OUTPUT_FOLDER, 'visualizations', f'{pc_id}_greyscale_all_boxes_viz.png')}")
        plt.close(fig_2d_viz) # Close the plot to free memory
    else:
        print(f"No greyscale boxes were successfully created for {pc_id} to generate combined visualization.")

    # Save greyscale segment metadata to Excel
    if greyscale_segment_data:
        df_greyscale = pd.DataFrame(greyscale_segment_data)
        # Check if the file exists and append if it does, otherwise create a new one
        if os.path.exists(EXCEL_GREYSCALE_OUTPUT_PATH):
            existing_df = pd.read_excel(EXCEL_GREYSCALE_OUTPUT_PATH)
            df_greyscale = pd.concat([existing_df, df_greyscale], ignore_index=True)
            print(f"Appended greyscale segment metadata for {pc_id} to {EXCEL_GREYSCALE_OUTPUT_PATH}")
        else:
            print(f"Created new greyscale segment metadata file at {EXCEL_GREYSCALE_OUTPUT_PATH}")
        df_greyscale.to_excel(EXCEL_GREYSCALE_OUTPUT_PATH, index=False)
    else:
        print(f"No greyscale segment metadata to save for {pc_id}.")

GLOBAL_Z_MIN, GLOBAL_Z_MAX = load_min_max_z()
if __name__ == "__main__":

    # Create parent directories if they don't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    os.makedirs(os.path.dirname(EXCEL_GREYSCALE_OUTPUT_PATH), exist_ok=True)

    # If the Excel file exists from a previous run, delete it to ensure a fresh start
    # or modify this logic if you intend to append across multiple script runs.
    if os.path.exists(EXCEL_GREYSCALE_OUTPUT_PATH):
        os.remove(EXCEL_GREYSCALE_OUTPUT_PATH)
        print(f"Removed existing Excel file: {EXCEL_GREYSCALE_OUTPUT_PATH}")


    # Loop over all pc files in folder
    os.makedirs(f"{OUTPUT_FOLDER}/pointcloud", exist_ok=True)
    os.makedirs(f"{OUTPUT_FOLDER}/greyscale", exist_ok=True)
    for file_path in glob.glob(os.path.join(f"{INPUT_FOLDER}/pointclouds", "*.ply")):
        filename = os.path.basename(file_path)
        pc_name = os.path.splitext(filename)[0]
        pc_id = pc_name.split("_")[0]

        # First load pointcloud + directly detrend plane and remove border
        pcd = load_pointcloud(filename)
        # Make greyscale image from loaded pointcloud for visualization later on
        img_full, x_offset_pc, y_offset_pc, pre_resize_width, pre_resize_height = point_cloud_to_image(pcd)
        pcd_np = np.asarray(pcd.points) # Renamed to avoid confusion with PyntCloud object
        if pcd_np.shape[0] > 0: # Check if point cloud is not empty
            local_min_z, local_max_z = (np.min(pcd_np[:, 2])), np.max(pcd_np[:, 2])
            print(f"Local min Z: {local_min_z:.4f}, Local max Z: {local_max_z:.4f}")
        # Skip to next file if pcd is empty
        else:
            print(f"Point cloud {pc_id} is empty after preprocessing, skipping to next file.")
            continue

        pcd_df = pd.DataFrame(pcd_np, columns=["x", "y", "z"])

        fixed_box_visualize(pcd_df, pc_id=pc_id, full_img=img_full,
                            x_offset_pc=x_offset_pc, y_offset_pc=y_offset_pc,
                            pre_resize_width=pre_resize_width, pre_resize_height=pre_resize_height)