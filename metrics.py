"""
Geoscientific DEM Super-Resolution Evaluation Metrics
=====================================================

Calculates quantitative metrics for DEM quality assessment:
1. PSNR (Peak Signal-to-Noise Ratio)
2. SSIM (Structural Similarity Index)
3. RMSE_Elevation (Real-world elevation error in meters)
4. RMSE_Slope (Topographic slope error in degrees)
5. RMSE_Aspect (Directional aspect error in degrees)
"""
import math
import numpy as np
import torch
import torch.nn.functional as F


def calculate_psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Calculates PSNR (Peak Signal-to-Noise Ratio) in dB."""
    mse = F.mse_loss(pred, gt).item()
    if mse == 0:
        return 100.0
    data_range = max(pred.max().item(), gt.max().item()) - min(pred.min().item(), gt.min().item())
    if data_range <= 0:
        return 0.0
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def calculate_ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Calculates mean SSIM score."""
    mu_x = pred.mean()
    mu_y = gt.mean()
    var_x = pred.var()
    var_y = gt.var()
    cov_xy = torch.mean((pred - mu_x) * (gt - mu_y))

    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return float(ssim.item())


def calculate_dem_metrics(pred_norm: torch.Tensor, gt_norm: torch.Tensor,
                          lr_min: float, lr_max: float, pixel_size: float = 30.0) -> dict:
    """
    Computes real-meter elevation and topographic gradient metrics.

    Args:
        pred_norm : [1, 1, H, W] Normalized predicted DEM tensor
        gt_norm   : [1, 1, H, W] Normalized ground truth DEM tensor
        lr_min    : Minimum elevation scalar for denormalization
        lr_max    : Maximum elevation scalar for denormalization
        pixel_size: Spatial grid resolution in meters

    Returns:
        metrics_dict : Dictionary containing PSNR, SSIM, RMSE_Ele, RMSE_Slope, RMSE_Aspect
    """
    denom = (lr_max - lr_min) if (lr_max - lr_min) > 1e-6 else 1.0

    # Denormalize tensors to real elevation values (meters)
    pred_m = pred_norm * denom + lr_min
    gt_m = gt_norm * denom + lr_min

    # 1. Pixel Metrics
    psnr_val = calculate_psnr(pred_norm, gt_norm)
    ssim_val = calculate_ssim(pred_norm, gt_norm)

    # 2. Real-World Elevation RMSE (meters)
    rmse_ele = float(torch.sqrt(F.mse_loss(pred_m, gt_m)).item())

    # 3. Slope & Aspect Computation (Real-World Space)
    device, dtype = pred_m.device, pred_m.dtype
    kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=device, dtype=dtype).view(1, 1, 3, 3) / (8.0 * pixel_size)
    ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]], device=device, dtype=dtype).view(1, 1, 3, 3) / (8.0 * pixel_size)

    p_dzdx = F.conv2d(pred_m, kx, padding=1)
    p_dzdy = F.conv2d(pred_m, ky, padding=1)
    g_dzdx = F.conv2d(gt_m, kx, padding=1)
    g_dzdy = F.conv2d(gt_m, ky, padding=1)

    # Slope in degrees
    p_slope_deg = torch.rad2deg(torch.atan(torch.sqrt(p_dzdx ** 2 + p_dzdy ** 2 + 1e-8)))
    g_slope_deg = torch.rad2deg(torch.atan(torch.sqrt(g_dzdx ** 2 + g_dzdy ** 2 + 1e-8)))
    rmse_slope = float(torch.sqrt(F.mse_loss(p_slope_deg, g_slope_deg)).item())

    # Aspect in degrees
    p_aspect_deg = torch.rad2deg(torch.atan2(-p_dzdy, p_dzdx + 1e-8)) % 360.0
    g_aspect_deg = torch.rad2deg(torch.atan2(-g_dzdy, g_dzdx + 1e-8)) % 360.0
    diff_aspect = torch.abs(p_aspect_deg - g_aspect_deg)
    diff_aspect = torch.minimum(diff_aspect, 360.0 - diff_aspect)
    rmse_aspect = float(torch.sqrt(torch.mean(diff_aspect ** 2)).item())

    return {
        'PSNR': psnr_val,
        'SSIM': ssim_val,
        'RMSE_Elevation': rmse_ele,
        'RMSE_Slope': rmse_slope,
        'RMSE_Aspect': rmse_aspect
    }
