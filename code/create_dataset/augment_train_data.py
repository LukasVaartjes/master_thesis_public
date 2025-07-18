import os
import pandas as pd
from PIL import Image
import numpy as np
import open3d as o3d # For point cloud manipulation

def rotate_pointcloud(points, angle_degrees, axis='z'):
    """
    Rotates a point cloud around a specified axis.

    Args:
        points (np.ndarray): Nx3 numpy array of point coordinates.
        angle_degrees (int): Rotation angle in degrees.
        axis (str): Axis to rotate around ('x', 'y', or 'z').

    Returns:
        np.ndarray: Rotated point cloud.
    """
    angle_radians = np.deg2rad(angle_degrees)
    cos_theta = np.cos(angle_radians)
    sin_theta = np.sin(angle_radians)

    if axis == 'x':
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, cos_theta, -sin_theta],
            [0, sin_theta, cos_theta]
        ])
    elif axis == 'y':
        rotation_matrix = np.array([
            [cos_theta, 0, sin_theta],
            [0, 1, 0],
            [-sin_theta, 0, cos_theta]
        ])
    elif axis == 'z':
        rotation_matrix = np.array([
            [cos_theta, -sin_theta, 0],
            [sin_theta, cos_theta, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y', or 'z'")

    return np.dot(points, rotation_matrix.T) # Transpose for correct multiplication

def augment_data_rotations(
    dataset_dir,
    dataset_name,
    image_folder_name="png",
    pointcloud_folder_name="ply",
    description_file_name="description.xlsx",
    png_col_name="File_Name_PNG",
    ply_col_name="File_Name_PLY"
):
    """
    Augments grayscale images and point clouds by rotating them 0, 90, 180, and 270 degrees.
    Updates the description Excel file with new entries for the rotated files.

    Args:
        dataset_dir (str): Base directory containing datasets (e.g., "./code/data/datasets").
        dataset_name (str): Specific dataset path (e.g., "split_output/train").
        image_folder_name (str): Name of the folder containing grayscale images (e.g., "png").
        pointcloud_folder_name (str): Name of the folder containing point clouds (e.g., "ply").
        description_file_name (str): Name of the Excel description file.
        png_col_name (str): Column name for PNG filenames in the Excel file.
        ply_col_name (str): Column name for PLY filenames in the Excel file.
    """

    full_dataset_path = os.path.join(dataset_dir, dataset_name)
    image_dir = os.path.join(full_dataset_path, image_folder_name)
    pointcloud_dir = os.path.join(full_dataset_path, pointcloud_folder_name)
    description_file_path = os.path.join(full_dataset_path, description_file_name)

    print(f"🔄 Starting data augmentation for: {dataset_name}")
    print(f"📁 Image directory: {image_dir}")
    print(f"☁️ Point cloud directory: {pointcloud_dir}")
    print(f"📄 Description file: {description_file_path}")

    try:
        df = pd.read_excel(description_file_path)
    except FileNotFoundError:
        print(f"❌ Error: Description file not found at {description_file_path}")
        return
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        return

    # Ensure required columns exist
    if png_col_name not in df.columns or ply_col_name not in df.columns:
        print(f"❌ Error: Missing '{png_col_name}' or '{ply_col_name}' column in {description_file_path}.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    new_rows_for_excel = []
    processed_files_count = 0
    
    # Angles for rotation
    angles = [0, 90, 180, 270]

    for index, row in df.iterrows():
        original_png_filename = row[png_col_name]
        original_ply_filename = row[ply_col_name]

        base_png_name, png_ext = os.path.splitext(original_png_filename)
        base_ply_name, ply_ext = os.path.splitext(original_ply_filename)

        # Process each rotation angle
        for angle in angles:
            # --- Handle Image Rotation ---
            if original_png_filename and os.path.exists(os.path.join(image_dir, original_png_filename)):
                image_path = os.path.join(image_dir, original_png_filename)
                new_png_filename = f"{base_png_name}_{angle}{png_ext}"
                new_image_path = os.path.join(image_dir, new_png_filename)

                if os.path.exists(new_image_path) and angle != 0: # Avoid re-processing if exists, but always check 0-degree
                    # print(f"  Skipping image {new_png_filename}: Already exists.")
                    pass
                else:
                    try:
                        img = Image.open(image_path).convert('L') # Convert to grayscale
                        if angle == 0:
                            rotated_img = img # No rotation needed
                        else:
                            rotated_img = img.rotate(angle, expand=False, fillcolor=0) # fill with black for greyscale

                        rotated_img.save(new_image_path)
                        processed_files_count += 1
                        # print(f"  Saved rotated image: {new_png_filename}")
                    except Exception as e:
                        print(f"Error processing image {original_png_filename} for {angle} degrees: {e}")
            else:
                new_png_filename = None # No PNG to process, so no new filename

            # --- Handle Point Cloud Rotation ---
            if original_ply_filename and os.path.exists(os.path.join(pointcloud_dir, original_ply_filename)):
                pointcloud_path = os.path.join(pointcloud_dir, original_ply_filename)
                new_ply_filename = f"{base_ply_name}_{angle}{ply_ext}"
                new_pointcloud_path = os.path.join(pointcloud_dir, new_ply_filename)

                if os.path.exists(new_pointcloud_path) and angle != 0: # Avoid re-processing if exists, but always check 0-degree
                    # print(f"  Skipping point cloud {new_ply_filename}: Already exists.")
                    pass
                else:
                    try:
                        pcd = o3d.io.read_point_cloud(pointcloud_path)
                        points_np = np.asarray(pcd.points)

                        if angle == 0:
                            rotated_points = points_np
                        else:
                            # For simplicity, rotating around Z-axis (up-axis in many 3D contexts)
                            rotated_points = rotate_pointcloud(points_np, angle, axis='z')

                        rotated_pcd = o3d.geometry.PointCloud()
                        rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points)
                        # If your point clouds have colors/normals, you'll need to handle them here as well
                        if pcd.has_colors():
                            rotated_pcd.colors = pcd.colors
                        if pcd.has_normals():
                            rotated_pcd.normals = pcd.normals

                        o3d.io.write_point_cloud(new_pointcloud_path, rotated_pcd)
                        processed_files_count += 1
                        # print(f"  Saved rotated point cloud: {new_ply_filename}")
                    except Exception as e:
                        print(f"Error processing point cloud {original_ply_filename} for {angle} degrees: {e}")
            else:
                new_ply_filename = None # No PLY to process, so no new filename
            
            # --- Prepare new row for Excel ---
            new_row_data = row.copy()
            if new_png_filename:
                new_row_data[png_col_name] = new_png_filename
            else:
                new_row_data[png_col_name] = np.nan # If no PNG, set to NaN
            
            if new_ply_filename:
                new_row_data[ply_col_name] = new_ply_filename
            else:
                new_row_data[ply_col_name] = np.nan # If no PLY, set to NaN

            new_rows_for_excel.append(new_row_data)

        if (index + 1) % 10 == 0:
            print(f"✅ Processed {index + 1} original entries.")


    if not new_rows_for_excel:
        print("\nNo new data generated for the Excel file.")
        return

    df_new_rows = pd.DataFrame(new_rows_for_excel)
    
    # Filter out original rows from the new dataframe, as we are creating all rotations for each original
    # We want to replace the original rows with their 0-degree rotated versions, and then add 90, 180, 270.
    # To do this correctly, we will create a new DataFrame from scratch.

    final_df_rows = []
    # Add unique original entries (or their 0-degree rotated versions) and then all other rotations
    
    # Create a set of original (base) filenames to avoid duplicates if 0-degree is explicitly handled later
    # This logic assumes the first entry for each original file (base_name_0) should be considered the 'new' original
    
    # Re-iterate through the original DataFrame to ensure each set of rotations is added correctly
    for index, row in df.iterrows():
        original_png_filename = row[png_col_name]
        original_ply_filename = row[ply_col_name]

        base_png_name, png_ext = os.path.splitext(original_png_filename) if pd.notna(original_png_filename) else (None, None)
        base_ply_name, ply_ext = os.path.splitext(original_ply_filename) if pd.notna(original_ply_filename) else (None, None)

        for angle in angles:
            current_row = row.copy()
            
            if base_png_name:
                current_row[png_col_name] = f"{base_png_name}_{angle}{png_ext}"
            else:
                current_row[png_col_name] = np.nan

            if base_ply_name:
                current_row[ply_col_name] = f"{base_ply_name}_{angle}{ply_ext}"
            else:
                current_row[ply_col_name] = np.nan
            
            final_df_rows.append(current_row)

    df_final = pd.DataFrame(final_df_rows)

    try:
        df_final.to_excel(description_file_path, index=False)
        print(f"\n🎉 Successfully augmented and updated {processed_files_count} files.")
        print(f"📊 New total entries in {description_file_name}: {len(df_final)}")
    except Exception as e:
        print(f"❌ Error saving updated Excel file: {e}")

if __name__ == "__main__":
    # Ensure you have Open3D installed: pip install open3d
    # And Pillow: pip install Pillow
    # And pandas: pip install pandas
    
    # Example Usage: Adjust these paths to match your setup
    # Your dataset structure:
    # dataset/split_output/train/png/
    # dataset/split_output/train/ply/
    # dataset/split_output/train/description.xlsx

    BASE_DATASET_PATH = "./dataset" # The root of your 'dataset' folder
    
    # This assumes your description.xlsx is inside 'train' folder
    AUGMENT_DATASET_PATH = "split_output/train" 
    
    augment_data_rotations(
        dataset_dir=BASE_DATASET_PATH,
        dataset_name=AUGMENT_DATASET_PATH,
        image_folder_name="png", # Folder containing your grayscale images
        pointcloud_folder_name="ply", # Folder containing your point clouds
        description_file_name="train_labels.xlsx", # Your description file name
        png_col_name="File_Name_PNG", # Column name for PNG files
        ply_col_name="File_Name_PLY" # Column name for PLY files
    )

    print("Data augmentation process complete for all rotations (0, 90, 180, 270 degrees).")