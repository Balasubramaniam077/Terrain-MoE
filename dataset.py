"""
DEM Dataset Loader & Synthetic Data Generator Module
=====================================================

Provides:
1. DEMDataset: Paired LR / HR DEM reader (.npy arrays or .tif GeoTIFFs)
2. generate_synthetic_dem_dataset: Automated synthetic terrain tile generator
   (using multi-octave Perlin-like spatial harmonics) for reviewer zero-config testing.
"""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DEMDataset(Dataset):
    """
    Dataset loader for paired LR and HR elevation tiles.
    """

    _VALID_EXT = ('.npy', '.tif', '.tiff', '.TIF', '.TIFF')

    def __init__(self, lr_dir: str, hr_dir: str):
        self.lr_dir = lr_dir
        self.hr_dir = hr_dir

        if os.path.exists(lr_dir) and os.path.exists(hr_dir):
            self.lr_files = sorted([f for f in os.listdir(lr_dir) if f.endswith(self._VALID_EXT)])
            self.hr_files = sorted([f for f in os.listdir(hr_dir) if f.endswith(self._VALID_EXT)])
        else:
            self.lr_files = []
            self.hr_files = []

    def __len__(self) -> int:
        return len(self.lr_files)

    def _load_file(self, path: str) -> np.ndarray:
        if path.lower().endswith('.npy'):
            arr = np.load(path)
        else:
            try:
                import rasterio
                with rasterio.open(path) as src:
                    arr = src.read(1)
            except ImportError:
                from PIL import Image
                arr = np.array(Image.open(path))
        return arr.astype(np.float32)

    def __getitem__(self, idx: int):
        lr_path = os.path.join(self.lr_dir, self.lr_files[idx])
        hr_path = os.path.join(self.hr_dir, self.hr_files[idx])

        lr_arr = self._load_file(lr_path)
        hr_arr = self._load_file(hr_path)

        # Per-sample LR normalization parameters
        lr_min = float(lr_arr.min())
        lr_max = float(lr_arr.max())
        denom = (lr_max - lr_min) if (lr_max - lr_min) > 1e-6 else 1.0

        lr_norm = (lr_arr - lr_min) / denom
        hr_norm = (hr_arr - lr_min) / denom

        lr_tensor = torch.from_numpy(lr_norm).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(hr_norm).unsqueeze(0).float()

        return {
            'lr': lr_tensor,
            'hr': hr_tensor,
            'lr_min': torch.tensor(lr_min, dtype=torch.float32),
            'lr_max': torch.tensor(lr_max, dtype=torch.float32),
            'filename': self.lr_files[idx]
        }


class SyntheticDEMDataset(Dataset):
    """
    On-the-fly synthetic terrain tile generator for instant testing & benchmarking.
    """

    def __init__(self, num_samples: int = 16, lr_shape=(64, 64), scale: int = 2):
        self.num_samples = num_samples
        self.lr_shape = lr_shape
        self.scale = scale
        self.hr_shape = (lr_shape[0] * scale, lr_shape[1] * scale)

    def __len__(self) -> int:
        return self.num_samples

    def _generate_terrain(self, shape, seed: int) -> np.ndarray:
        np.random.seed(seed)
        h, w = shape
        y, x = np.ogrid[:h, :w]

        # Multi-scale sinusoidal elevation harmonic synthesis
        elevation = (
            500.0 * np.sin(x / 15.0) * np.cos(y / 15.0) +
            200.0 * np.sin(x / 5.0 + y / 7.0) +
            50.0 * np.random.randn(h, w)
        )
        return elevation.astype(np.float32)

    def __getitem__(self, idx: int):
        hr_arr = self._generate_terrain(self.hr_shape, seed=idx + 100)
        
        # Downsample HR to LR via block mean pooling
        h_lr, w_lr = self.lr_shape
        lr_arr = hr_arr.reshape(h_lr, self.scale, w_lr, self.scale).mean(axis=(1, 3))

        lr_min = float(lr_arr.min())
        lr_max = float(lr_arr.max())
        denom = (lr_max - lr_min) if (lr_max - lr_min) > 1e-6 else 1.0

        lr_norm = (lr_arr - lr_min) / denom
        hr_norm = (hr_arr - lr_min) / denom

        return {
            'lr': torch.from_numpy(lr_norm).unsqueeze(0).float(),
            'hr': torch.from_numpy(hr_norm).unsqueeze(0).float(),
            'lr_min': torch.tensor(lr_min, dtype=torch.float32),
            'lr_max': torch.tensor(lr_max, dtype=torch.float32),
            'filename': f"synthetic_tile_{idx:03d}.npy"
        }


def get_dataloaders(config, scale: int = 2):
    """
    Returns train and val DataLoaders. Falls back to SyntheticDEMDataset if paths are empty.
    """
    train_dataset = DEMDataset(config.train_lr_path, config.train_hr_path)
    val_dataset = DEMDataset(config.val_lr_path, config.val_hr_path)

    if len(train_dataset) == 0:
        print("[Dataset] Real dataset paths empty or not found. Initializing Synthetic DEM dataset for test run.")
        train_dataset = SyntheticDEMDataset(num_samples=16, scale=scale)
        val_dataset = SyntheticDEMDataset(num_samples=8, scale=scale)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                              num_workers=config.num_workers, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                            num_workers=config.num_workers)

    return train_loader, val_loader
