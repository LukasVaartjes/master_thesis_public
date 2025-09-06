import os
import shutil
import random
import pandas as pd
from PIL import Image
import numpy as np
from collections import defaultdict

DATASET_DIR = "./dataset_agreed/"
#Ratio of train and validate, so that the test set is the remaining part
TRAIN_RATIO = 0.7
VALIDATE_RATIO = 0.15
GREYSCALE_DATA_FOLDER = f"{DATASET_DIR}greyscale"
POINTCLOUD_DATA_FOLDER = f"{DATASET_DIR}pointcloud_filtered"
DESCRIPTION_FILE = f"{DATASET_DIR}agreed_data.xlsx"
OUTPUT_PATH = f"{DATASET_DIR}split_output"

def get_image_pixel_stats(filepath):
    """
    Calculates the minimum, maximum, and difference in pixel values for a given greyscale image.
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
    Splits the dataset into train/validate/test.
    Ensures that all segments from the same Original_Image_ID stay in the same split.
    Keeps the 'ID' column from the original Excel file.
    """
    if not os.path.exists(DESCRIPTION_FILE):
        print(f"{DESCRIPTION_FILE} not found.")
        return

    df = pd.read_excel(DESCRIPTION_FILE)

    required_cols = [
        "ID", "Original_Image_ID", "Segment_ID_In_Original",
        "Min_Pixel_Value", "Max_Pixel_Value", "Pixel_Value_Difference",
        "Good_layer", "Ditch", "Crater", "Waves"
    ]
    for col in required_cols:
        if col not in df.columns:
            print(f"Required column '{col}' not found in the Excel file")
            return

    # Clean up IDs
    df["Original_Image_ID"] = pd.to_numeric(df["Original_Image_ID"], errors="coerce")
    df["Segment_ID_In_Original"] = pd.to_numeric(df["Segment_ID_In_Original"], errors="coerce")
    df.dropna(subset=["Original_Image_ID", "Segment_ID_In_Original"], inplace=True)

    df["Original_Image_ID"] = df["Original_Image_ID"].astype(int)
    df["Segment_ID_In_Original"] = df["Segment_ID_In_Original"].astype(int)

    # Filenames 
    df['png_file_name'] = df.apply(
        lambda row: f"{int(row['ID']):02d}_{int(row['Original_Image_ID']):02d}_box_{int(row['Segment_ID_In_Original'])}.png",
        axis=1
    )
    df['ply_file_name'] = df.apply(
        lambda row: f"{int(row['ID']):02d}_{int(row['Original_Image_ID']):02d}_box_{int(row['Segment_ID_In_Original'])}.ply",
        axis=1
    )

    all_png_files = set(os.listdir(GREYSCALE_DATA_FOLDER))
    all_ply_files = set(os.listdir(POINTCLOUD_DATA_FOLDER))

    valid_pairs = []
    for _, row in df.iterrows():
        if row["png_file_name"] not in all_png_files or row["ply_file_name"] not in all_ply_files:
            continue
        valid_pairs.append({
            "ID": row["ID"],
            "Original_Image_ID": row["Original_Image_ID"],
            "Segment_ID_In_Original": row["Segment_ID_In_Original"],
            "png_file": row["png_file_name"],
            "ply_file": row["ply_file_name"],
            "Min_Pixel_Value": row["Min_Pixel_Value"],
            "Max_Pixel_Value": row["Max_Pixel_Value"],
            "Pixel_Value_Difference": row["Pixel_Value_Difference"],
            "Good_layer": row["Good_layer"],
            "Ditch": row["Ditch"],
            "Crater": row["Crater"],
            "Waves": row["Waves"]
        })

    if not valid_pairs:
        print("No valid pairs found")
        return

    # Group by Original_Image_ID so that all segments of a sample are in same set
    grouped = defaultdict(list)
    for entry in valid_pairs:
        grouped[entry["Original_Image_ID"]].append(entry)

    all_groups = list(grouped.values())
    random.shuffle(all_groups)

    total_groups = len(all_groups)
    train_size = int(total_groups * TRAIN_RATIO)
    validate_size = int(total_groups * VALIDATE_RATIO)

    train_groups = all_groups[:train_size]
    validate_groups = all_groups[train_size:train_size+validate_size]
    test_groups = all_groups[train_size+validate_size:]

    train_pairs = [item for group in train_groups for item in group]
    validate_pairs = [item for group in validate_groups for item in group]
    test_pairs = [item for group in test_groups for item in group]

    train_dir = os.path.join(OUTPUT_PATH, "train")
    validate_dir = os.path.join(OUTPUT_PATH, "validate")
    test_dir = os.path.join(OUTPUT_PATH, "test")
    for split_dir in [train_dir, validate_dir, test_dir]:
        os.makedirs(os.path.join(split_dir, "png"), exist_ok=True)
        os.makedirs(os.path.join(split_dir, "ply"), exist_ok=True)

    def process_split(pairs, split_dir):
        data_list = []
        for file_pair in pairs:
            shutil.copy(os.path.join(GREYSCALE_DATA_FOLDER, file_pair["png_file"]),
                        os.path.join(split_dir, "png", file_pair["png_file"]))
            shutil.copy(os.path.join(POINTCLOUD_DATA_FOLDER, file_pair["ply_file"]),
                        os.path.join(split_dir, "ply", file_pair["ply_file"]))
            data_list.append(file_pair)
        return pd.DataFrame(data_list)

    train_df = process_split(train_pairs, train_dir)
    validate_df = process_split(validate_pairs, validate_dir)
    test_df = process_split(test_pairs, test_dir)

    train_df.to_excel(os.path.join(train_dir, "train_labels.xlsx"), index=False)
    validate_df.to_excel(os.path.join(validate_dir, "validate_labels.xlsx"), index=False)
    test_df.to_excel(os.path.join(test_dir, "test_labels.xlsx"), index=False)

    print("Datasplit complete")
    print(f"Train: {len(train_df)} segments from {len(train_groups)} images")
    print(f"Validate: {len(validate_df)} segments from {len(validate_groups)} images")
    print(f"Test: {len(test_df)} segments from {len(test_groups)} images")
    print_class_distribution(validate_df, "Validate")
    print_class_distribution(test_df, "Test")
    print_class_distribution(train_df, "Train")

#Print class distributions over all the sets
def print_class_distribution(df, split_name):
    print(f"\nClass distribution in {split_name}:")
    for col in ["Good_layer", "Ditch", "Crater", "Waves"]:
        count = df[col].sum()
        print(f"  {col}: {count} segments")


def create_empty_description_file():
    """
    Creates an empty Excel file named 'description_greyscale.xlsm' at the predefined
    DESCRIPTION_FILE path. This file is used as a template for users
    It includes a set of required columns that the dataset splitting logic uses
    """
    required_columns = ["Original_Image_ID", "Segment_ID_In_Original",
                        "Row_Min", "Col_Min", "Row_Max", "Col_Max", "Segment_Width",
                        "Segment_Height", "Min_Pixel_Value", "Max_Pixel_Value",
                        "Pixel_Value_Difference", "Good_layer", "Ditch", "Crater", "Waves"]
    empty_df = pd.DataFrame(columns=required_columns)
    try:
        empty_df.to_excel(DESCRIPTION_FILE, index=False)
        print(f"Created empty description file at: {DESCRIPTION_FILE}")

    except Exception as e:
        print(f"Creating empty description file at {DESCRIPTION_FILE}: {e}")

if __name__ == "__main__":
    os.makedirs(os.path.dirname(DESCRIPTION_FILE), exist_ok=True)
    if not os.path.exists(DESCRIPTION_FILE):
        create_empty_description_file()
        print("\nNo description file found")
        exit()

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    split_dataset()



