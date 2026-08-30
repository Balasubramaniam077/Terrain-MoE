"""
TerraMoE-v2 Model Training Executable
=====================================

Usage:
    python train.py --epochs 75 --scale 2
    python train.py --scale 4 --epochs 100
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from config import TerraMoEv2Config
from model import build_model
from dataset import get_dataloaders
from losses import TerraMoEv2Loss
from metrics import calculate_dem_metrics
from utils import save_checkpoint, CSVLogger


def parse_args():
    parser = argparse.ArgumentParser(description="TerraMoE-v2 Model Training CLI")
    parser.add_argument('--scale', type=int, default=2, choices=[2, 4], help="Upscaling scale factor (2 or 4)")
    parser.add_argument('--epochs', type=int, default=75, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size")
    parser.add_argument('--lr', type=float, default=2e-4, help="Learning rate")
    parser.add_argument('--data_root', type=str, default="./data", help="Root directory for dataset")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Save directory for checkpoints")
    return parser.parse_args()


def main():
    args = parse_args()

    # Initialize Configuration
    config = TerraMoEv2Config()
    config.up_scale = args.scale
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.data_root = args.data_root
    config.save_dir = args.checkpoint_dir
    config.update_paths_by_scale()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================")
    print(f" TerraMoE-v2 Model Training Pipeline")
    print(f" Scale Factor : {config.up_scale}x")
    print(f" Target Device: {device}")
    print(f" Epochs       : {config.epochs}")
    print(f" Batch Size   : {config.batch_size}")
    print(f"==================================================")

    # Dataloaders
    train_loader, val_loader = get_dataloaders(config, scale=config.up_scale)

    # Model & Loss Evaluator
    model = build_model(config).to(device)
    loss_evaluator = TerraMoEv2Loss(config).to(device)

    # Optimizer & GradScaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scaler = GradScaler('cuda', enabled=config.amp and torch.cuda.is_available())

    # Logger
    csv_fields = ['epoch', 'train_loss', 'val_psnr', 'val_ssim', 'val_rmse_ele']
    logger = CSVLogger(os.path.join(config.log_dir, f"train_log_x{config.up_scale}.csv"), csv_fields)

    best_val_psnr = -1.0

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_epoch_loss = 0.0

        for batch in train_loader:
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)

            optimizer.zero_grad()

            if config.amp and torch.cuda.is_available():
                with autocast('cuda'):
                    pred, pred_2x, gates, aux = model(lr)
                    loss, loss_dict = loss_evaluator(pred, hr, lr, pred_2x, aux)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred, pred_2x, gates, aux = model(lr)
                loss, loss_dict = loss_evaluator(pred, hr, lr, pred_2x, aux)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                optimizer.step()

            total_epoch_loss += loss.item()

        avg_train_loss = total_epoch_loss / len(train_loader)

        # Validation Step
        model.eval()
        val_psnr_list, val_ssim_list, val_rmse_list = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                lr = batch['lr'].to(device)
                hr = batch['hr'].to(device)
                lr_min = batch['lr_min'].item()
                lr_max = batch['lr_max'].item()

                pred, _, _, _ = model(lr)
                m = calculate_dem_metrics(pred, hr, lr_min, lr_max, config.pixel_size)

                val_psnr_list.append(m['PSNR'])
                val_ssim_list.append(m['SSIM'])
                val_rmse_list.append(m['RMSE_Elevation'])

        mean_psnr = float(np.mean(val_psnr_list))
        mean_ssim = float(np.mean(val_ssim_list))
        mean_rmse = float(np.mean(val_rmse_list))

        print(f"Epoch [{epoch:03d}/{config.epochs:03d}] | Train Loss: {avg_train_loss:.4f} | Val PSNR: {mean_psnr:.2f} dB | Val SSIM: {mean_ssim:.4f} | Val RMSE Ele: {mean_rmse:.3f} m")

        logger.log({
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'val_psnr': mean_psnr,
            'val_ssim': mean_ssim,
            'val_rmse_ele': mean_rmse
        })

        if mean_psnr > best_val_psnr:
            best_val_psnr = mean_psnr
            save_path = os.path.join(config.save_dir, f"TerraMoE_v2_x{config.up_scale}_Best.pth")
            save_checkpoint({'epoch': epoch, 'model': model.state_dict(), 'best_psnr': best_val_psnr}, save_path)

    print(f"\nTraining Complete! Best Val PSNR: {best_val_psnr:.2f} dB")


if __name__ == '__main__':
    main()
