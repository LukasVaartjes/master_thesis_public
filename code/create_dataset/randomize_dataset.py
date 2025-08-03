import pandas as pd
import os
import glob

original_excel_file = "dataset/description_greyscale.xlsx"
new_excel_file = "randomized_and_labeled.xlsx"

base_dir = "dataset"
greyscale_dir = os.path.join(base_dir, "greyscale")
pointcloud_dir = os.path.join(base_dir, "pointcloud")

try:
    df = pd.read_excel(original_excel_file)
except FileNotFoundError:
    print(f"file '{original_excel_file}' not found")
    exit()

df_randomized = df.sample(frac=1, random_state=42).reset_index(drop=True)

#Add new id column
df_randomized.insert(0, 'ID', range(len(df_randomized)))
df_randomized['ID'] = df_randomized['ID'].astype(str).str.zfill(2)

df_randomized.to_excel(new_excel_file, index=False)
print(f"randomized and labeled dataset saved to '{new_excel_file}'.")

# Rename files
print("\nStarting to rename image and pointcloud files...")
for index, row in df_randomized.iterrows():
    try:
        original_img_id = str(int(row['Original_Image_ID'])).zfill(2)
        segment_id = str(int(row['Segment_ID_In_Original']))
        original_filename_base = f"{original_img_id}_box_{segment_id}"

        new_id = row['ID']
        new_filename_base = f"{new_id}_{original_filename_base}"

        #rename greyscale images
        original_greyscale_path = os.path.join(greyscale_dir, f"{original_filename_base}.png")
        new_greyscale_path = os.path.join(greyscale_dir, f"{new_filename_base}.png")
        if os.path.exists(original_greyscale_path):
            os.rename(original_greyscale_path, new_greyscale_path)
            # print(f"renamed {os.path.basename(original_greyscale_path)} to {os.path.basename(new_greyscale_path)}")
        else:
            print(f"greyscale file not found {original_greyscale_path}")

        #rename point cloud files
        original_pointcloud_path = os.path.join(pointcloud_dir, f"{original_filename_base}.ply")
        new_pointcloud_path = os.path.join(pointcloud_dir, f"{new_filename_base}.ply")
        if os.path.exists(original_pointcloud_path):
            os.rename(original_pointcloud_path, new_pointcloud_path)
            # print(f"renamed {os.path.basename(original_pointcloud_path)} to {os.path.basename(new_pointcloud_path)}")
        else:
            print(f"pointcloud file not found {original_pointcloud_path}")
            
    except Exception as e:
        print(f"error while processing row {index}: {e}")

print("\nrandomizing done")