import os
import shutil
import random
import pandas as pd
from PIL import Image
import numpy as np

DATASET_DIR = "./dataset/"

# Helper function to load image and calculate pixel stats
def get_image_pixel_stats(filepath):
    """
    Loads a grayscale image and calculates its min, max, and difference in pixel values.
    """
    try:
        img = Image.open(filepath).convert('L') # Convert to grayscale
        img_array = np.array(img)

        min_pixel = np.min(img_array)
        max_pixel = np.max(img_array)
        diff_pixel = max_pixel - min_pixel
        return min_pixel, max_pixel, diff_pixel
    except Exception as e:
        print(f"Error processing image {filepath}: {e}")
        return None, None, None

# MODIFIED: split_dataset now accepts separate paths for PNG and PLY folders
def split_dataset(greyscale_folder, pointcloud_folder, description_file, output_base, train_ratio=0.7, validate_ratio=0.20):

    if not os.path.exists(description_file):
        print(f"Description file '{description_file}' not found. Exiting.")
        return

    df = pd.read_excel(description_file)

    if 'File_Name' not in df.columns or 'Original_Image_ID' not in df.columns or 'Segment_ID_In_Original' not in df.columns:
        print("Required columns ('File_Name', 'Original_Image_ID', 'Segment_ID_In_Original') not found in excel file.")
        return

    # Standardize File_Name to always end with .png for matching purposes
    df['File_Name'] = df['File_Name'].astype(str).apply(lambda name: name if name.endswith('.png') else f"{name}.png")

    # Create a 'ply_file_name' column based on Original_Image_ID and Segment_ID_In_Original
    df['ply_file_name'] = df.apply(lambda row: f"{row['Original_Image_ID']:02d}_box_{row['Segment_ID_In_Original']}.ply", axis=1)

    df['label'] = -1
    df.loc[df['Perfect layer'] == 1, 'label'] = 0

    defect_columns = ['Ditch', 'Crater', 'Waves']
    for col in defect_columns:
        if col in df.columns:
            df.loc[df[col] == 1, 'label'] = 1
        else:
            print(f"Defect column '{col}' not found in excel file. Please ensure '{col}' exists for complete labeling.")

    # Filter out rows with unassigned labels
    df_processed = df[['File_Name', 'ply_file_name', 'label']].copy()
    initial_rows = len(df_processed)
    df_processed = df_processed[df_processed['label'] != -1]
    if len(df_processed) < initial_rows:
        print(f"{initial_rows - len(df_processed)} rows with missing or unassigned labels were removed.")

    # MODIFIED: Get actual files from their respective folders
    all_png_files_in_folder = set(os.listdir(greyscale_folder))
    all_ply_files_in_folder = set(os.listdir(pointcloud_folder))

    # Filter for pairs where both PNG and PLY files actually exist in their respective folders AND have a label
    valid_pairs_with_labels = []
    for index, row in df_processed.iterrows():
        png_file = row['File_Name']
        ply_file = row['ply_file_name']
        
        # MODIFIED: Check in respective folders
        if png_file in all_png_files_in_folder and ply_file in all_ply_files_in_folder:
            valid_pairs_with_labels.append({
                'png_file': png_file,
                'ply_file': ply_file,
                'label': row['label']
            })
    
    if not valid_pairs_with_labels:
        print("No corresponding .png and .ply file pairs found with labels that exist in their specified folders.")
        return

    # Shuffle the list of valid pairs to ensure random splitting
    random.shuffle(valid_pairs_with_labels)

    total_files = len(valid_pairs_with_labels)
    train_size = int(total_files * train_ratio)
    validate_size = int(total_files * validate_ratio)
    test_size = total_files - train_size - validate_size

    if test_size < 0:
        print(f"Combined train_ratio ({train_ratio}) and validate_ratio ({validate_ratio}) exceed 1.0. Please adjust ratios.")
        return

    train_pairs = valid_pairs_with_labels[:train_size]
    validate_pairs = valid_pairs_with_labels[train_size : train_size + validate_size]
    test_pairs = valid_pairs_with_labels[train_size + validate_size :]

    train_dir = os.path.join(output_base, "train")
    validate_dir = os.path.join(output_base, "validate")
    test_dir = os.path.join(output_base, "test")

    # Create subdirectories for png and ply within each split set
    os.makedirs(os.path.join(train_dir, "png"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "ply"), exist_ok=True)
    os.makedirs(os.path.join(validate_dir, "png"), exist_ok=True)
    os.makedirs(os.path.join(validate_dir, "ply"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "png"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "ply"), exist_ok=True)

    train_data_list = []
    validate_data_list = []
    test_data_list = []

    # Iterate through the shuffled valid pairs
    for file_pair in train_pairs + validate_pairs + test_pairs:
        png_file = file_pair['png_file']
        ply_file = file_pair['ply_file']
        label = file_pair['label']

        # MODIFIED: Use separate source paths
        src_png_path = os.path.join(greyscale_folder, png_file)
        src_ply_path = os.path.join(pointcloud_folder, ply_file)

        min_p, max_p, diff_p = get_image_pixel_stats(src_png_path)
        if min_p is None:
            print(f"Skipping pair {png_file} due to image loading error for {png_file}.")
            continue

        file_data = {
            'File_Name_PNG': png_file, 
            'File_Name_PLY': ply_file, 
            'label': label,
            'Min_Pixel_Value': min_p,
            'Max_Pixel_Value': max_p,
            'Pixel_Value_Difference': diff_p
        }

        if file_pair in train_pairs: # Check membership in the actual lists
            shutil.copy(src_png_path, os.path.join(train_dir, "png", png_file))
            shutil.copy(src_ply_path, os.path.join(train_dir, "ply", ply_file))
            train_data_list.append(file_data)
        elif file_pair in validate_pairs:
            shutil.copy(src_png_path, os.path.join(validate_dir, "png", png_file))
            shutil.copy(src_ply_path, os.path.join(validate_dir, "ply", ply_file))
            validate_data_list.append(file_data)
        elif file_pair in test_pairs:
            shutil.copy(src_png_path, os.path.join(test_dir, "png", png_file))
            shutil.copy(src_ply_path, os.path.join(test_dir, "ply", ply_file))
            test_data_list.append(file_data)
        else:
            print(f"File pair {png_file} and {ply_file} not assigned to any split. This should not happen.")

    train_df = pd.DataFrame(train_data_list)
    validate_df = pd.DataFrame(validate_data_list)
    test_df = pd.DataFrame(test_data_list)

    train_excel_path = os.path.join(train_dir, "train_labels.xlsx")
    validate_excel_path = os.path.join(validate_dir, "validate_labels.xlsx")
    test_excel_path = os.path.join(test_dir, "test_labels.xlsx")

    train_df.to_excel(train_excel_path, index=False)
    validate_df.to_excel(validate_excel_path, index=False)
    test_df.to_excel(test_excel_path, index=False)

    print(f"Dataset split completed:")
    print(f"  Train: {len(train_df)} file pairs saved to {train_dir}")
    print(f"  Validate: {len(validate_df)} file pairs saved to {validate_dir}")
    print(f"  Test: {len(test_df)} file pairs saved to {test_dir}")

def create_empty_description_file(filepath):
    """
    Creates an empty Excel file with predefined headers required by split_dataset.
    """
    # Updated required columns based on your description file structure
    required_columns = ["Original_Image_ID", "Segment_ID_In_Original", 
                        "Row_Min", "Col_Min", "Row_Max", "Col_Max", "Segment_Width",
                        "Segment_Height", "Min_Pixel_Value", "Max_Pixel_Value",
                        "Pixel_Value_Difference", "Label (0 = no defect, 1 = defect)", "Perfect layer", "Ditch", "Crater", "Waves"]
    empty_df = pd.DataFrame(columns=required_columns)
    try:
        empty_df.to_excel(filepath, index=False)
        print(f"Created empty description file at: {filepath}")
        
    except Exception as e:
        print(f"Error creating empty description file at {filepath}: {e}")

if __name__ == "__main__":
    # Define paths for greyscale and pointcloud folders separately
    greyscale_data_folder = os.path.join(DATASET_DIR, "greyscale")
    pointcloud_data_folder = os.path.join(DATASET_DIR, "pointcloud")
    
    # This description file should be for the .png files primarily, as labels are image-centric
    description_file = os.path.join(DATASET_DIR, "description.xlsm")
    output_base = os.path.join(DATASET_DIR, "split_output/test")

    # Ensure the parent directory for description_file exists
    os.makedirs(os.path.dirname(description_file), exist_ok=True)

    if not os.path.exists(description_file):
        create_empty_description_file(description_file)
        print("\nExiting. Please populate the description file and run the script again.")
        exit()

    os.makedirs(output_base, exist_ok=True)

    # Pass both folder paths to the split_dataset function
    split_dataset(greyscale_data_folder, pointcloud_data_folder, description_file, output_base, train_ratio=0.7, validate_ratio=0.20)