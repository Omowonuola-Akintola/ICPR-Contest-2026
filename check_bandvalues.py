import os
import rasterio
import numpy as np

# root_dir = "/Users/rabinatwayana/1_Rabina/3_CDE_III/ICPR_competition/ICPR-Contest-2026/data/s2a_selected"
root_dir = "/home/krschap/rabina/data/s2a_subset"

max_global = -np.inf
files_over_10000 = []
files_with_nan = []

for scene in os.listdir(root_dir):
    scene_path = os.path.join(root_dir, scene)

    if not os.path.isdir(scene_path):
        continue

    # go inside nested folder
    for sub in os.listdir(scene_path):
        band_folder = os.path.join(scene_path, sub)

        if not os.path.isdir(band_folder):
            continue

        for file in os.listdir(band_folder):
            if file.endswith(".tif") or file.endswith(".tiff"):
                filepath = os.path.join(band_folder, file)
                with rasterio.open(filepath) as src:
                    img = src.read(1)
                max_val = np.nanmax(img)
                min_val = np.nanmin(img)

                # # track global max
                # if max_val > max_global:
                #     max_global = max_val

                # check >8000
                if max_val > 10000:
                    print(f" MAX value {max_val:.2f} in file: {filepath}")
                    # files_over_10000.append((filepath, max_val))
                if min_val < 0:
                    print(f"MIN value {min_val:.2f} in file: {filepath}")
                    # files_over_10000.append((filepath, max_val))
                # check NaN
                if np.isnan(img).any():
                    print(f"NaN values in file: {filepath}")
                    files_with_nan.append(filepath)

