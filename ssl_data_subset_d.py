

import os
import shutil

# Paths
source_root = "/home/krschap/rabina/data/s2a"
destination_root="/home/krschap/rabina/data/s2a_subset"
#source_root = "/Users/rabinatwayana/1_Rabina/3_CDE_III/ICPR_competition/ICPR-Contest-2026/data/s2a"          # original data folder
#destination_root = "/Users/rabinatwayana/1_Rabina/3_CDE_III/ICPR_competition/ICPR-Contest-2026/data/s2a_selected"   # where new selected folders will go

# Make destination folder if it doesn't exist
os.makedirs(destination_root, exist_ok=True)

# Iterate over each folder_id
for folder_id in os.listdir(source_root):
    folder_path = os.path.join(source_root, folder_id)
    if not os.path.isdir(folder_path):
        continue

    # Get list of timestamp folders and sort
    timestamps = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    if not timestamps:
        continue

    timestamps.sort()  # sort alphabetically or numerically
    mid_index = len(timestamps) // 2  # middle folder
    mid_timestamp = timestamps[mid_index]

    # Source and destination paths
    src_timestamp_path = os.path.join(folder_path, mid_timestamp)
    dst_folder_id_path = os.path.join(destination_root, folder_id)
    dst_timestamp_path = os.path.join(dst_folder_id_path, mid_timestamp)

    # Make destination folder
    os.makedirs(dst_timestamp_path, exist_ok=True)

    # Copy all files from selected timestamp folder
    for item in os.listdir(src_timestamp_path):
        src_item = os.path.join(src_timestamp_path, item)
        dst_item = os.path.join(dst_timestamp_path, item)
        if os.path.isdir(src_item):
            shutil.copytree(src_item, dst_item)
        else:
            shutil.copy2(src_item, dst_item)

    print(f"Copied {mid_timestamp} from {folder_id} to new folder.")

print("Done!")
