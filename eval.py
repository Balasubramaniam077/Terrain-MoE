"""
TerraMoE-v2 Model Benchmark Evaluation Executable
=================================================

Usage:
    python eval.py --checkpoint checkpoints/TerraMoE_v2_x2_Best.pth --scale 2 --save_maps
"""
import os
import argparse
import numpy as np
import torch

from config import TerraMoEv2Config
from model import build_model
from dataset import get_dataloaders, SyntheticDEMDataset
from metrics import calculate_dem_metrics
from utils import load_checkpoint, save_routing_heatmaps, save_dem_error_map


def parse_args():
    parser = argparse.ArgumentParser(description="TerraMoE-v2 Benchmark Evaluation CLI")
    parser.add_argument('--checkpoint', type=str, default=None, help="Path to checkpoint .pth file")
    parser.add_argument('--scale', type=int, default=2, choices=[2, 4], help="Upscaling factor")
    parser.add_argument('--save_maps', action='store_true', help="Save expert gate heatmaps & error maps")
    parser.add_argument('--quick_test', action='store_true', help="Run quick automated test on synthetic data")
    return parser.parse_args()


def main():
    args = parse_args()

    config = TerraMoEv2Config()
    config.up_scale = args.scale
    config.update_paths_by_scale()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"==================================================")
    print(f" TerraMoE-v2 Evaluation Suite")
    print(f" Scale Factor : {config.up_scale}x")
    print(f" Device       : {device}")
    print(f"==================================================")

    model = build_model(config).to(device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint weights from: {args.checkpoint}")
        load_checkpoint(args.checkpoint, model)
    else:
        print("Notice: Running evaluation with initialized/random weights for test verification.")

    model.eval()

    if args.quick_test or not os.path.exists(config.test_lr_path):
        print("Using Synthetic DEM Dataset for test execution...")
        from torch.utils.data import DataLoader
        test_dataset = SyntheticDEMDataset(num_samples=4, scale=config.up_scale)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    else:
        _, test_loader = get_dataloaders(config, scale=config.up_scale)

    psnr_vals, ssim_vals, rmse_ele_vals, rmse_slope_vals, rmse_aspect_vals = [], [], [], [], []

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            lr = batch['lr'].to(device)
            hr = batch['hr'].to(device)
            lr_min = batch['lr_min'].item()
            lr_max = batch['lr_max'].item()
            fname = batch['filename'][0] if isinstance(batch['filename'], list) else f"tile_{idx:03d}"

            pred, _, gates, _ = model(lr)
            m = calculate_dem_metrics(pred, hr, lr_min, lr_max, config.pixel_size)

            psnr_vals.append(m['PSNR'])
            ssim_vals.append(m['SSIM'])
            rmse_ele_vals.append(m['RMSE_Elevation'])
            rmse_slope_vals.append(m['RMSE_Slope'])
            rmse_aspect_vals.append(m['RMSE_Aspect'])

            print(f"Tile [{idx+1:02d}/{len(test_loader):02d}] {fname} | PSNR: {m['PSNR']:.2f} dB | SSIM: {m['SSIM']:.4f} | RMSE Ele: {m['RMSE_Elevation']:.3f}m | Slope: {m['RMSE_Slope']:.2f}°")

            if args.save_maps or args.quick_test:
                out_gate_path = os.path.join("results", "routing", f"gate_map_{idx:02d}.png")
                out_err_path = os.path.join("results", "error_maps", f"error_map_{idx:02d}.png")
                save_routing_heatmaps(gates, out_gate_path, tile_name=fname)
                save_dem_error_map(pred, hr, out_err_path, tile_name=fname)

    print("\n==================================================")
    print(" BENCHMARK PERFORMANCE SUMMARY")
    print(f" PSNR           : {np.mean(psnr_vals):.2f} ± {np.std(psnr_vals):.2f} dB")
    print(f" SSIM           : {np.mean(ssim_vals):.4f} ± {np.std(ssim_vals):.4f}")
    print(f" RMSE Elevation : {np.mean(rmse_ele_vals):.3f} ± {np.std(rmse_ele_vals):.3f} m")
    print(f" RMSE Slope     : {np.mean(rmse_slope_vals):.2f} ± {np.std(rmse_slope_vals):.2f} deg")
    print(f" RMSE Aspect    : {np.mean(rmse_aspect_vals):.2f} ± {np.std(rmse_aspect_vals):.2f} deg")
    print("==================================================")


if __name__ == '__main__':
    main()
