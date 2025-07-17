# File to split a dataset consisting # of greyscale images and their corresponding point cloud files. 
# Including functions to calculate pixel statistics for images and to split the dataset
# into training, validation, and test sets and store Excel description file.
# Use requirement to install required packages

import os
import shutil
import random
import pandas as pd
from PIL import Image
import numpy as np

DATASET_DIR = "./dataset/"
#Ratio of train and validate, so that the test set is the remaining part
TRAIN_RATIO = 0.7
VALIDATE_RATIO = 0.2
GREYSCALE_DATA_FOLDER = f"{DATASET_DIR}greyscale"
POINTCLOUD_DATA_FOLDER = f"{DATASET_DIR}pointcloud"
DESCRIPTION_FILE  = f"{DATASET_DIR}description_greyscale.xlsm"
OUTPUT_PATH = f"{DATASET_DIR}split_output/test"

def get_image_pixel_stats(filepath):
    """
    Calculates the minimum, maximum, and difference in pixel values for a given greyscale image.

    Args:
        filepath (str): The greyscale segment image

    Returns:
        tuple: A tuple containing (min_pixel, max_pixel, diff_pixel)
    """
    try:
        img = Image.open(filepath).convert('L')
        img_array = np.array(img)

        min_pixel = np.min(img_array)
        max_pixel = np.max(img_array)
        diff_pixel = max_pixel - min_pixel
        return min_pixel, max_pixel, diff_pixel
    except Exception as e:
        print(f"error processing image {filepath}: {e}")
        return None, None, None


def split_dataset():
    """
    Splits the combined dataset of greyscale images and point cloud files into
    training, validation, and test subsets based on the ratios. It reads
    metadata from an Excel description file, verifies the existence of corresponding
    image and point cloud files, copies them to their respective new directories,
    and generates separate Excel metadata files for each split.
    """
    if not os.path.exists(DESCRIPTION_FILE):
        print(f"{DESCRIPTION_FILE} not found.")
        return

    df = pd.read_excel(DESCRIPTION_FILE)
    if 'Original_Image_ID' not in df.columns or 'Segment_ID_In_Original' not in df.columns:
        print("column ('Original_Image_ID', 'Segment_ID_In_Original') not found in excel file")
        return

    df['Original_Image_ID'] = pd.to_numeric(df['Original_Image_ID'], errors='coerce')
    df['Segment_ID_In_Original'] = pd.to_numeric(df['Segment_ID_In_Original'], errors='coerce')

    initial_rows_before_id_drop = len(df)
    df.dropna(subset=['Original_Image_ID', 'Segment_ID_In_Original'], inplace=True)
    if len(df) < initial_rows_before_id_drop:
        print(f"removed {initial_rows_before_id_drop - len(df)} rows due to missing or non-numeric Original_Image_ID or Segment_ID_In_Original header")
    
    df['Original_Image_ID'] = df['Original_Image_ID'].astype(np.int64)
    df['Segment_ID_In_Original'] = df['Segment_ID_In_Original'].astype(np.int64)

    df['ply_file_name'] = df.apply(
        lambda row: f"{int(row['Original_Image_ID']):02d}_box_{int(row['Segment_ID_In_Original']):d}.ply", 
        axis=1
    )

    df_processed = df[['Original_Image_ID', 'Segment_ID_In_Original', 'ply_file_name']].copy()
    df_processed['label'] = ""

    all_png_files_in_folder = set(os.listdir(GREYSCALE_DATA_FOLDER))
    all_ply_files_in_folder = set(os.listdir(POINTCLOUD_DATA_FOLDER))

    valid_pairs_with_labels = []
    
    for index, row in df_processed.iterrows():
        png_file = f"{int(row['Original_Image_ID']):02d}_box_{int(row['Segment_ID_In_Original']):d}.png"
        ply_file = row['ply_file_name']
        
        png_exists = png_file in all_png_files_in_folder
        ply_exists = ply_file in all_ply_files_in_folder

        if not png_exists or not ply_exists:
            continue 

        valid_pairs_with_labels.append({
            'png_file': png_file,
            'ply_file': ply_file,
            'label': row['label'] 
        })
    
    if not valid_pairs_with_labels:
        return

    random.shuffle(valid_pairs_with_labels)

    total_files = len(valid_pairs_with_labels)
    train_size = int(total_files * TRAIN_RATIO)
    validate_size = int(total_files * VALIDATE_RATIO)
    test_size = total_files - train_size - validate_size

    if test_size < 0:
        print(f"train_ratio ({TRAIN_RATIO}) and validate_ratio ({VALIDATE_RATIO}) greater than 1")
        return

    train_pairs = valid_pairs_with_labels[:train_size]
    validate_pairs = valid_pairs_with_labels[train_size : train_size + validate_size]
    test_pairs = valid_pairs_with_labels[train_size + validate_size :]

    train_dir = os.path.join(OUTPUT_PATH, "train")
    validate_dir = os.path.join(OUTPUT_PATH, "validate")
    test_dir = os.path.join(OUTPUT_PATH, "test")

    os.makedirs(os.path.join(train_dir, "png"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "ply"), exist_ok=True)
    os.makedirs(os.path.join(validate_dir, "png"), exist_ok=True)
    os.makedirs(os.path.join(validate_dir, "ply"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "png"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "ply"), exist_ok=True)

    train_data_list = []
    validate_data_list = []
    test_data_list = []

    for file_pair in train_pairs + validate_pairs + test_pairs:
        png_file = file_pair['png_file']
        ply_file = file_pair['ply_file']
        label = file_pair['label']

        src_png_path = os.path.join(GREYSCALE_DATA_FOLDER, png_file)
        src_ply_path = os.path.join(POINTCLOUD_DATA_FOLDER, ply_file)

        min_p, max_p, diff_p = get_image_pixel_stats(src_png_path)
        if min_p is None:
            print(f"Skipping pair {png_file} due to image loading error for {png_file}")
            continue

        file_data = {
            'File_Name_PNG': png_file, 
            'File_Name_PLY': ply_file, 
            'label': label,
            'Min_Pixel_Value': min_p,
            'Max_Pixel_Value': max_p,
            'Pixel_Value_Difference': diff_p
        }

        if file_pair in train_pairs:
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
            print(f"files {png_file} and {ply_file} not assigned to any split")

    train_df = pd.DataFrame(train_data_list)
    validate_df = pd.DataFrame(validate_data_list)
    test_df = pd.DataFrame(test_data_list)

    train_excel_path = os.path.join(train_dir, "train_labels.xlsx")
    validate_excel_path = os.path.join(validate_dir, "validate_labels.xlsx")
    test_excel_path = os.path.join(test_dir, "test_labels.xlsx")

    train_df.to_excel(train_excel_path, index=False)
    validate_df.to_excel(validate_excel_path, index=False)
    test_df.to_excel(test_excel_path, index=False)

    print(f"Datasplit complete")
    print(f"Train: {len(train_df)} file pairs saved to {train_dir}")
    print(f"Validate: {len(validate_df)} file pairs saved to {validate_dir}")
    print(f"Test: {len(test_df)} file pairs saved to {test_dir}")


def create_empty_description_file():
    """
    Creates an empty Excel file named 'description_greyscale.xlsm' at the predefined
    DESCRIPTION_FILE path. This file is used as a template for users 
    It includes a set of required columns that the dataset splitting logic uses
    """
    required_columns = ["Original_Image_ID", "Segment_ID_In_Original", 
                        "Row_Min", "Col_Min", "Row_Max", "Col_Max", "Segment_Width",
                        "Segment_Height", "Min_Pixel_Value", "Max_Pixel_Value",
                        "Pixel_Value_Difference", "Label (0 = no defect, 1 = defect)", "Perfect layer", "Ditch", "Crater", "Waves"]
    empty_df = pd.DataFrame(columns=required_columns)
    try:
        empty_df.to_excel(DESCRIPTION_FILE, index=False)
        print(f"Created empty description file at: {DESCRIPTION_FILE}")
        
    except Exception as e:
        print(f"Error creating empty description file at {DESCRIPTION_FILE}: {e}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(DESCRIPTION_FILE), exist_ok=True)
    if not os.path.exists(DESCRIPTION_FILE):
        create_empty_description_file()
        print("\nNo description file found")
        exit()

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    split_dataset()