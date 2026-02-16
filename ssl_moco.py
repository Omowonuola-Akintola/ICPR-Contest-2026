# tailscale ssh krschap@tuxedo
# nvtop
# htop
# uv venv rabina_venv
# source rabina_venv/bin/activate
# uv init
# uv add torchgeo lightning prettytable
# uv run ssl_moco.py --data_root_dir /home/krschap/rabina/data/s2a

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
from prettytable import PrettyTable
from lightning.pytorch.loggers import CSVLogger
from torchvision import transforms
#from torchvision.models import ResNet50_Weights
from lightning.pytorch import Trainer
import kornia.augmentation as K
from dataclasses import dataclass
from torchgeo.trainers.moco import MoCoTask
from torchgeo.models.resnet import ResNet50_Weights
from typing import Optional
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

@dataclass
class DataConfig:
    data_root_dir: str = "/home/krschap/rabina/data/s2a"
    compute_stats: bool = True
    n_samples: int = None
    batch_size: int = 64
    patch_size: int = 264
    num_workers: int = 1

@dataclass
class TrainingConfig:
    experiment_out_dir: str = "ssl_moco_"
    model: str = "resnet50"
    #weights: ResNet50_Weights = ResNet50_Weights.SENTINEL2_ALL_MOCO
    in_channels: int = 13
    version: int = 2
    lr: float = 1e-4
    use_peft: bool = False
    temperature: float = 0.15
    memory_bank_size: int = 2048
    target_size: int = 224
    max_epochs: int = 100
    batch_size: int =32
    ckpt_path: Optional[str] = None
    #devices = []

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
    torch.set_printoptions(sci_mode=False, precision=4)
    # std  = torch.sqrt(channel_sum_sq / num_pixels - mean**2)
    variance = channel_sum_sq / num_pixels - mean**2
    variance = torch.clamp(variance, min=0)  # Avoid negative values
    # Add small epsilon to avoid sqrt(0) issues
    std = torch.sqrt(variance + 1e-8)
    return mean, std

def summary_trainable(model):
    table = PrettyTable()
    table.field_names = ["Module", "Type", "Trainable Params", "Total Params"]

    for name, module in model.named_children():
        total_params = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        table.add_row([name, type(module).__name__, f"{trainable_params:,}", f"{total_params:,}"])

    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(table)
    print(f"Total trainable parameters: {total_trainable:,} ({total_trainable / 1e6:.2f} M)")
    print(f"Total parameters: {total_params:,} ({total_params / 1e6:.2f} M)")


# def main(data_root_dir, n_samples,  batch_size, patch_size, num_workers):
def main(data_cfg, training_cfg):
    print("Data root directory:", data_cfg.data_root_dir)
    print("========================")
    print("Dataset config:", data_cfg)
    print("========================")
    print("Training config:", training_cfg)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(training_cfg.experiment_out_dir, exist_ok=True)
    logger = CSVLogger("logs", name=f"{training_cfg.experiment_out_dir}/metrics_{timestamp}")

    aug = K.AugmentationSequential(
        K.RandomResizedCrop(size=(training_cfg.target_size, training_cfg.target_size), scale=(0.4, 1.0)),
        K.RandomHorizontalFlip(),
        K.RandomVerticalFlip(),
        K.RandomGaussianBlur(kernel_size=(7,7), sigma=(0.1, 1.5), p=0.3),
        K.RandomBrightness(brightness=(0.85, 1.15), p=0.5),
        data_keys=['input'],
    )

    if not os.path.exists(data_cfg.data_root_dir):
        raise FileNotFoundError(f"Data root directory does not exist: {data_cfg.data_root_dir}")
    scenes = sorted(glob.glob(os.path.join(data_cfg.data_root_dir, "*/")))
    bands = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
    # ========================
    # Compute dataset statistics (mean, std)
    # ========================
    if data_cfg.compute_stats:
        start_time = time.time()
        #scenes = sorted(glob.glob(os.path.join(data_cfg.data_root_dir, "*/")))
        end_time = time.time()

        print(f"Found {len(scenes)} scenes in {end_time-start_time:.2f} seconds")

        # bands = ["B1","B2","B3","B4","B5","B6","B7","B8","B8A","B9","B11","B12"]
        temp_dataset = SSLDataset(scenes, bands, patch_size=data_cfg.patch_size)
        start_time = time.time()
        # mean, std = calculate_stats(temp_dataset, n_samples=10000)
        mean, std = calculate_stats_parallel(temp_dataset, n_samples=data_cfg.n_samples, batch_size=data_cfg.batch_size, num_workers=data_cfg.num_workers)
        end_time = time.time()
        print(f"calculate_stats time: {(end_time-start_time)/60:.2f} min")
        print("Mean:", mean)
        print("Std:", std)
        mean = mean.tolist()
        std = std.tolist()
    else:
        print("Using pre-computed mean and std")
        mean =[1278.6852, 1434.4397, 1694.8578, 1932.4095, 2269.2026, 2789.2571,
            3019.3687, 3144.4609, 3180.2004, 3289.7437,    0.0000, 2666.2808,
            2098.4189]
        
        std= [2148.8049, 2148.8838, 2038.1532, 2097.1721, 2066.5100, 1890.2559,      
            1842.6375, 1898.1923, 1762.8573, 1992.9287,    0.0001, 1247.9540,
            1132.2416]

    # ========================
    # Train MoCo model
    # ========================
    transform = transforms.Compose([
        transforms.Resize((training_cfg.target_size, training_cfg.target_size)),
        transforms.Normalize(mean=mean, std=std)
    ])
        
    dataset = SSLDataset(scenes, bands, transforms=transform, patch_size=training_cfg.target_size)
    print(len(dataset))
    print(dataset[0]['image'].shape)

    data_loader = DataLoader(
        dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        num_workers=data_cfg.num_workers,
        worker_init_fn=seed_worker,
        generator=g
    )
    num_batches = len(data_loader)
    print("Number of batches:", num_batches)

    import time
    task = MoCoTask(
        model=training_cfg.model,      
        weights= ResNet50_Weights.SENTINEL2_ALL_MOCO,
        in_channels=training_cfg.in_channels,       
        version=training_cfg.version,             # MoCo v2
        size=training_cfg.target_size,          
        augmentation1=aug,
        augmentation2=aug,
        lr=training_cfg.lr,
        memory_bank_size=training_cfg.memory_bank_size,
        temperature=training_cfg.temperature,
    )

    # -----------------------------
    # PEFT / Full Fine-Tuning Logic
    # -----------------------------
    if training_cfg.use_peft:
        print("Using PEFT: freezing backbone except last block, training projection head...")
        for name, param in task.backbone.named_parameters():
            if "layer4" in name:      # optionally fine-tune last residual block
                param.requires_grad = True
            else:
                param.requires_grad = False
    else:
        print("Full fine-tuning: backbone and projection head trainable...")
        for param in task.backbone.parameters():
            param.requires_grad = True

    # Momentum backbone always frozen
    for param in task.backbone_momentum.parameters():
        param.requires_grad = False

    # Projection head always trainable
    for param in task.projection_head.parameters():
        param.requires_grad = True

    # Example usage for your task
    summary_trainable(task)

    from lightning.pytorch.callbacks import ModelCheckpoint

    checkpoint_callback = ModelCheckpoint(
      dirpath=training_cfg.experiment_out_dir,
      filename="ssl-best-{epoch:02d}",
      monitor="train_loss",
      mode="min",
      save_top_k=1,
      save_last=True
    )

    trainer = Trainer(
        max_epochs=training_cfg.max_epochs,
        enable_progress_bar=True, 
        log_every_n_steps=num_batches,
        precision=16,
        accelerator="gpu", # if torch.cuda.is_available() else "cpu",
        #devices = [0], # training_cfg.devices,
	deterministic=True,
        callbacks=[checkpoint_callback],
        logger=logger)
    
    print("USING DEVICE CONFIRMATION", next(task.parameters()).device)
    start_time=time.time()
    trainer.fit(task, data_loader, ckpt_path=training_cfg.ckpt_path)
    end_time=time.time()
    print("After fit device:", next(task.parameters()).device)
    print(f"Training time: {(end_time-start_time)/60} min")

    torch.save(task.backbone.state_dict(),f"{training_cfg.experiment_out_dir}/ssl_backbone_{timestamp}.pth")
    torch.save(task.projection_head.state_dict(), f"{training_cfg.experiment_out_dir}/projection_head_{timestamp}.pth")
    trainer.save_checkpoint(f"{training_cfg.experiment_out_dir}/ssl_ckpt_{timestamp}.ckpt")


if __name__ == "__main__":
    device =  "cpu"
    target_num_workers = int(os.cpu_count()*0.75)  # Use 75% of available CPU cores
    print("Using device:", device)
    print("CPU cores available:", os.cpu_count())
    print("CPU cores using:", target_num_workers)

    # target_num_workers = 0 # Use 75% of available CPU cores

    # target_batch_size = 64   # prefer 256 or 128 depending on GPU memory
    # target_patch_size = 264  # Not used in current code but can be added for random cropping

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
    
    args = parser.parse_args()

    # Training configuration
    
    
    data_cfg = DataConfig(
        data_root_dir=args.data_root_dir,
        compute_stats=False,
        n_samples=args.n_samples, # only used for stats calculation, not training
        num_workers=args.num_workers,
        batch_size=64, # only used for stats calculation, not training
        patch_size=264, # only used for stats calculation, not training
    )

    training_cfg = TrainingConfig(
        experiment_out_dir=f"output/ssl_v2_e50_100_b96_mem_16k",
        model="resnet50",
        # weights= ResNet50_Weights.SENTINEL2_ALL_MOCO,
        in_channels=13,
        version=2,
        lr=1e-4,
        use_peft=False,
        temperature=0.15,
        memory_bank_size=16000, #4096, #2048
        target_size=224,
        batch_size=96,
        max_epochs=100,
        #ckpt_path="/home/krschap/rabina/ICPR-Contest-2026/output/ssl_v1_e20_b96_mem_16k/ssl_ckpt_20260215_042511.ckpt",
        ckpt_path = "/home/krschap/rabina/ICPR-Contest-2026/output/ssl_v1_e20_50_b96_mem_16k/ssl_ckpt_20260215_104921.ckpt"
        #devices=[0]
    )
    main(data_cfg, training_cfg)


   


    
    # main(args.data_root_dir, args.n_samples, args.batch_size, args.patch_size, args.num_workers, training_cfg)
