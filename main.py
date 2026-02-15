# tailscale ssh krschap@tuxedo
# nvtop
# htop
# uv venv rabina_venv
# source rabina_venv/bin/activate
# uv init
# uv add torchgeo lightning prettytable

import torch
torch.cuda.empty_cache()

# ========================
# Imports
# ========================
import os
import random
import time
from datetime import datetime
import glob
import argparse

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import rasterio
from rasterio.enums import Resampling

# ========================
# GPU & CPU setup
# ========================


# ========================
# Reproducibility
# ========================
seed = 42
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Timestamp for logging / checkpoints
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(seed)

class SSLDataset(Dataset):
    def __init__(self, scenes, bands, transforms=None, patch_size=264):
        """
        Args:
            scenes (list): List of scene folder paths.
            bands (list): List of band names (e.g., ["B1","B2"]).
            patch_size (tuple): Size of random crop (H, W).
            transforms (callable, optional): Optional transform to apply to patches.
        """
        self.scenes = scenes
        self.bands = bands
        self.patch_size = patch_size
        self.transforms = transforms
        # Precompute all timestamp paths to treat each timestamp as a sample
        self.samples = []
        for scene_path in scenes:
            timestamps = sorted([
                d for d in os.listdir(scene_path)
                if os.path.isdir(os.path.join(scene_path, d))
            ])
            for ts in timestamps:
                self.samples.append(os.path.join(scene_path, ts))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        ts_path = self.samples[idx]

        band_arrays = []

        for b in self.bands:
            path = os.path.join(ts_path, f"{b}.tif")
            with rasterio.open(path) as src:
                if src.height == self.patch_size and src.width == self.patch_size:
                    arr = src.read(1).astype(np.float32)
                else:
                    arr = src.read(
                        1,
                        out_shape=(self.patch_size, self.patch_size),
                        resampling=Resampling.bilinear
                    ).astype(np.float32)

            band_arrays.append(arr)

        # Insert fake B10
        insert_idx = 10
        b10_pad = np.zeros((self.patch_size, self.patch_size), dtype=np.float32)
        band_arrays.insert(insert_idx, b10_pad)

        img = np.stack(band_arrays, axis=0)

        # img_patch = self._random_crop(img)

        patch_tensor = torch.tensor(img, dtype=torch.float32)

        if self.transforms:
            patch_tensor = self.transforms(patch_tensor)

        return {"image": patch_tensor}

def calculate_stats_parallel(dataset, n_samples=None, batch_size=16, num_workers=4):

    total = len(dataset)
    print(f"Total samples in dataset: {total}")

    if n_samples is not None:
        n = min(total, n_samples)
        print(f"Calculating stats on {n} randomly selected samples...")
        # Randomly select a subset of indices for efficiency
        np.random.seed(seed)
        indices = np.random.choice(total, size=n, replace=False)
        subset = Subset(dataset, indices)
    else:
        print(f"Calculating stats on the entire dataset...")
        subset = dataset
        n_samples=total

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )

    channel_sum = 0.0
    channel_sum_sq = 0.0
    num_pixels = 0

    total_batches = len(loader)
    print(f"Total batches to process: {total_batches}")
    for batch_idx, batch in enumerate(loader, start=1):
        imgs = batch["image"]
        b, c, h, w = imgs.shape
        channel_sum += imgs.sum(dim=(0, 2, 3))
        channel_sum_sq += (imgs**2).sum(dim=(0, 2, 3))
        num_pixels += b * h * w

        if batch_idx % 500 == 0 or batch_idx == total_batches:
            print(f"Processed batch {batch_idx}/{total_batches} ")
                # f"≈ {batch_idx * b}/{n_samples} samples")

    mean = channel_sum / num_pixels
    # std  = torch.sqrt(channel_sum_sq / num_pixels - mean**2)
    variance = channel_sum_sq / num_pixels - mean**2
    variance = torch.clamp(variance, min=0)  # Avoid negative values
    # Add small epsilon to avoid sqrt(0) issues
    std = torch.sqrt(variance + 1e-8)
    return mean, std

def main(data_root_dir, n_samples,  batch_size, patch_size,num_workers):
    if not os.path.exists(data_root_dir):
        raise FileNotFoundError(f"Data root directory does not exist: {data_root_dir}")
    
    start_time = time.time()
    scenes = sorted(glob.glob(os.path.join(data_root_dir, "*/")))
    end_time = time.time()

    print(f"Found {len(scenes)} scenes in {end_time-start_time:.2f} seconds")

    bands = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
    temp_dataset = SSLDataset(scenes, bands, patch_size=patch_size)
    start_time = time.time()
    # mean, std = calculate_stats(temp_dataset, n_samples=10000)
    mean, std = calculate_stats_parallel(temp_dataset, n_samples=n_samples, batch_size=batch_size, num_workers=num_workers)
    end_time = time.time()
    print(f"calculate_stats time: {(end_time-start_time)/60:.2f} min")
    print("Mean:", mean)
    print("Std:", std)

if __name__ == "__main__":
    device =  "cpu"
    target_num_workers = int(os.cpu_count()*0.75)  # Use 75% of available CPU cores
    print("Using device:", device)
    print("CPU cores available:", os.cpu_count())
    print("CPU cores using:", target_num_workers)

    # target_num_workers = 0 # Use 75% of available CPU cores

    target_batch_size = 64   # prefer 256 or 128 depending on GPU memory
    target_patch_size = 264  # Not used in current code but can be added for random cropping

    parser = argparse.ArgumentParser(description="Calculate dataset statistics")
    parser.add_argument(
        "--data_root_dir",
        type=str,
        default="/Users/rabinatwayana/1_Rabina/3_CDE_III/ICPR_competition/ICPR-Contest-2026/data/ICPR_SSL_S2A_3k_sample",
        help="Path to the root directory of scenes"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None, # Set to None to use the entire dataset
        help="Number of samples to calculate statistics on"
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=target_num_workers,
        help="Number of workers for data loading"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=target_batch_size,
        help="Batch size for data loading"
    )

    parser.add_argument(
        "--patch_size",
        type=int,
        default=target_patch_size,
        help="Target patch size for resizing (if needed)"
    )

    args = parser.parse_args()
    main(args.data_root_dir, args.n_samples, args.batch_size, args.patch_size, args.num_workers)
