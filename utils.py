"""
Utility Helpers for Checkpointing, Logging, and Routing Visualizations
========================================================================
"""
import os
import csv
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save_checkpoint(state: dict, path: str):
    """Saves PyTorch model state dictionary and optimizer checkpoints."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, scaler=None):
    """Loads PyTorch model state dictionary."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scaler and 'scaler' in ckpt:
        scaler.load_state_dict(ckpt['scaler'])
    return ckpt.get('epoch', 0), ckpt.get('best_val_loss', math.inf)


class CSVLogger:
    """CSV Logger for tracking training loss curves and validation metrics."""

    def __init__(self, filepath: str, fieldnames: list):
        self.filepath = filepath
        self.fieldnames = fieldnames
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if not os.path.exists(filepath):
            with open(filepath, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()

    def log(self, row: dict):
        with open(self.filepath, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


def save_routing_heatmaps(gates: torch.Tensor, save_path: str, tile_name: str = "tile"):
    """
    Renders 4-expert spatial activation heatmaps.

    Args:
        gates : [1, 4, H, W] Softmax expert probabilities
        save_path : Output directory or file path
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    gates_np = gates.detach().squeeze(0).cpu().numpy()  # [4, H, W]
    expert_names = ["E0: Ridge/Edge", "E1: Valley/Drainage", "E2: Flat/Smooth", "E3: Rough/Dissected"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for k in range(4):
        im = axes[k].imshow(gates_np[k], cmap='inferno', vmin=0.0, vmax=1.0)
        axes[k].set_title(expert_names[k], fontsize=10)
        axes[k].axis('off')
        fig.colorbar(im, ax=axes[k], fraction=0.046, pad=0.04)

    plt.suptitle(f"TerraMoE-v2 Routing Activation Heatmap — {tile_name}", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_dem_error_map(pred: torch.Tensor, gt: torch.Tensor, save_path: str, tile_name: str = "tile"):
    """Renders absolute elevation error map."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    err = torch.abs(pred - gt).detach().squeeze().cpu().numpy()

    plt.figure(figsize=(6, 5))
    plt.imshow(err, cmap='magma')
    plt.title(f"Absolute Elevation Error Map (m) — {tile_name}")
    plt.colorbar(label='Absolute Error (m)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
