import os
import pandas as pd
from PIL import Image
import numpy as np
import open3d as o3d

def rotate_pointcloud(points, angle_degrees, axis='z'):
    """
    Rotates a point cloud around a specified axis.

    Args:
        points : numpy array of point coordinates.
        angle_degrees: Rotation angle in degrees.
        axis : Axis to rotate around ('x', 'y', or 'z').

    Returns:
        rotated point cloud.
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

    return np.dot(points, rotation_matrix.T)

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
    The original 0-degree files are renamed to include "_0" in their filenames.

    Args:
        dataset_dir: dataset directory
        dataset_name: dataset name
        image_folder_name: name of the folder with grayscale images
        pointcloud_folder_name : name of the folder with point clouds
        description_file_name: excel descritption file
        png_col_name: column name for png filenames
        ply_col_name: column name for ply filenames
    """

    full_dataset_path = os.path.join(dataset_dir, dataset_name)
    image_dir = os.path.join(full_dataset_path, image_folder_name)
    pointcloud_dir = os.path.join(full_dataset_path, pointcloud_folder_name)
    description_file_path = os.path.join(full_dataset_path, description_file_name)

    print(f"Starting data augmentation for: {dataset_name}")
    print(f"Image directory: {image_dir}")
    print(f"Point cloud directory: {pointcloud_dir}")
    print(f"Description file: {description_file_path}")

    try:
        df = pd.read_excel(description_file_path)
    except FileNotFoundError:
        print(f"Error: Description file not found at {description_file_path}")
        return
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return

    if png_col_name not in df.columns or ply_col_name not in df.columns:
        print(f"Error: Missing '{png_col_name}' or '{ply_col_name}' column in {description_file_path}.")
        print(f"Available columns: {df.columns.tolist()}")
        return

    new_rows_for_excel = []
    processed_files_count = 0

    # Angles for rotation
    angles = [0, 90, 180, 270]

    for index, row in df.iterrows():
        original_png_filename = row[png_col_name]
        original_ply_filename = row[ply_col_name]

        base_png_name_original, png_ext = os.path.splitext(original_png_filename) if pd.notna(original_png_filename) else (None, None)
        base_ply_name_original, ply_ext = os.path.splitext(original_ply_filename) if pd.notna(original_ply_filename) else (None, None)

        # Determine the name for the 0-degree file
        renamed_png_filename_0 = f"{base_png_name_original}_0{png_ext}" if base_png_name_original else None
        renamed_ply_filename_0 = f"{base_ply_name_original}_0{ply_ext}" if base_ply_name_original else None

        # Handle PNG renaming for 0-degree
        if original_png_filename and os.path.exists(os.path.join(image_dir, original_png_filename)):
            original_image_path = os.path.join(image_dir, original_png_filename)
            renamed_image_path_0 = os.path.join(image_dir, renamed_png_filename_0)
            if original_png_filename != renamed_png_filename_0:
                if not os.path.exists(renamed_image_path_0):
                    try:
                        os.rename(original_image_path, renamed_image_path_0)
                        processed_files_count += 1
                    except Exception as e:
                        print(f"Error renaming original image {original_png_filename} to {renamed_png_filename_0}: {e}")
                else:
                    pass
            else: 
                pass
        else:
            renamed_png_filename_0 = None

        # Handle PLY renaming for 0-degree
        if original_ply_filename and os.path.exists(os.path.join(pointcloud_dir, original_ply_filename)):
            original_pointcloud_path = os.path.join(pointcloud_dir, original_ply_filename)
            renamed_pointcloud_path_0 = os.path.join(pointcloud_dir, renamed_ply_filename_0)
            if original_ply_filename != renamed_ply_filename_0:
                if not os.path.exists(renamed_pointcloud_path_0):
                    try:
                        os.rename(original_pointcloud_path, renamed_pointcloud_path_0)
                        processed_files_count += 1
                    except Exception as e:
                        print(f"Error renaming original point cloud {original_ply_filename} to {renamed_ply_filename_0}: {e}")
                else:
                    
                    pass
            else: 
                pass
        else:
            renamed_ply_filename_0 = None

        for angle in angles:
            current_png_filename = None
            current_ply_filename = None

            if angle == 0:
                # For 0-degree, use the renamed original filenames
                current_png_filename = renamed_png_filename_0
                current_ply_filename = renamed_ply_filename_0
            else:
                # For other angles, perform rotation from the _0 file
                # Image rotation
                if renamed_png_filename_0 and os.path.exists(os.path.join(image_dir, renamed_png_filename_0)):
                    image_path_source = os.path.join(image_dir, renamed_png_filename_0)
                    current_png_filename = f"{base_png_name_original}_{angle}{png_ext}"
                    new_image_path = os.path.join(image_dir, current_png_filename)

                    if not os.path.exists(new_image_path):
                        try:
                            img = Image.open(image_path_source).convert('L')
                            rotated_img = img.rotate(angle, expand=False, fillcolor=0)
                            rotated_img.save(new_image_path)
                            processed_files_count += 1
                        except Exception as e:
                            print(f"Error processing image {renamed_png_filename_0} for {angle} degrees: {e}")
                else:
                    current_png_filename = None

                # Point cloud rotation
                if renamed_ply_filename_0 and os.path.exists(os.path.join(pointcloud_dir, renamed_ply_filename_0)):
                    pointcloud_path_source = os.path.join(pointcloud_dir, renamed_ply_filename_0)
                    current_ply_filename = f"{base_ply_name_original}_{angle}{ply_ext}"
                    new_pointcloud_path = os.path.join(pointcloud_dir, current_ply_filename)

                    if not os.path.exists(new_pointcloud_path): # Only create if it doesn't exist
                        try:
                            pcd = o3d.io.read_point_cloud(pointcloud_path_source)
                            points_np = np.asarray(pcd.points)
                            rotated_points = rotate_pointcloud(points_np, angle, axis='z')
                            rotated_pcd = o3d.geometry.PointCloud()
                            rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points)

                            if pcd.has_colors():
                                rotated_pcd.colors = pcd.colors
                            if pcd.has_normals():
                                rotated_pcd.normals = pcd.normals

                            o3d.io.write_point_cloud(new_pointcloud_path, rotated_pcd)
                            processed_files_count += 1
                        except Exception as e:
                            print(f"Error processing point cloud {renamed_ply_filename_0} for {angle} degrees: {e}")
                else:
                    current_ply_filename = None

            # Add the entry to the list for the DataFrame
            new_row_data = row.copy()
            new_row_data[png_col_name] = current_png_filename
            new_row_data[ply_col_name] = current_ply_filename
            new_rows_for_excel.append(new_row_data)

        if (index + 1) % 10 == 0:
            print(f"processed {index + 1} files")


    if not new_rows_for_excel:
        print("No new data generated for the Excel file")
        return

    df_final = pd.DataFrame(new_rows_for_excel)

    try:
        df_final.to_excel(description_file_path, index=False)
        print(f"ugmented and updated {processed_files_count} files")
        print(f"totale new files in {description_file_name}: {len(df_final)}")
    except Exception as e:
        print(f"saving updated Excel file went wrong {e}")

if __name__ == "__main__":

    BASE_DATASET_PATH = "./dataset"

    AUGMENT_DATASET_PATH = "split_output/train"

    augment_data_rotations(
        dataset_dir=BASE_DATASET_PATH,
        dataset_name=AUGMENT_DATASET_PATH,
        image_folder_name="png",
        pointcloud_folder_name="ply",
        description_file_name="train_labels.xlsx",
        png_col_name="File_Name_PNG",
        ply_col_name="File_Name_PLY"
    )

    print("Augmentation complete")