"""
Differential Terrain Operators & Spatial Decomposition Module
===============================================================

Provides differentiable differential operators for computing terrain descriptors:
1. Slope (radians / magnitude)
2. Aspect (Sine and Cosine directional decomposition)
3. Plan Curvature (horizontal contour curvature)
4. Profile Curvature (vertical slope gradient curvature)
5. Low-pass Gaussian Blur for Trend-Residual Elevation Decomposition
"""
import math
import torch
import torch.nn.functional as F

# Fixed Sobel spatial gradient kernels
_SOBEL_X = torch.tensor([[-1., 0., 1.],
                         [-2., 0., 2.],
                         [-1., 0., 1.]]).view(1, 1, 3, 3)

_SOBEL_Y = torch.tensor([[-1., -2., -1.],
                         [ 0.,  0.,  0.],
                         [ 1.,  2.,  1.]]).view(1, 1, 3, 3)


def compute_terrain_descriptors(dem: torch.Tensor, pixel_size: float = 30.0) -> torch.Tensor:
    """
    Computes 5 physically-based terrain descriptors from normalized or real DEM tiles.

    Args:
        dem : [B, 1, H, W] Elevation tensor
        pixel_size : Grid cell size in meters (default 30.0m)

    Returns:
        descriptors : [B, 5, H, W]
                      Channel 0: Normalized Slope
                      Channel 1: Aspect Sine (sin A)
                      Channel 2: Aspect Cosine (cos A)
                      Channel 3: Plan Curvature
                      Channel 4: Profile Curvature
    """
    device = dem.device
    dtype = dem.dtype

    kx = _SOBEL_X.to(device=device, dtype=dtype) / (8.0 * pixel_size)
    ky = _SOBEL_Y.to(device=device, dtype=dtype) / (8.0 * pixel_size)

    # First derivatives
    dz_dx = F.conv2d(dem, kx, padding=1)
    dz_dy = F.conv2d(dem, ky, padding=1)

    # 1. Slope
    slope = torch.atan(torch.sqrt(dz_dx ** 2 + dz_dy ** 2 + 1e-8))

    # 2 & 3. Aspect decomposition (Sine and Cosine)
    aspect_rad = torch.atan2(-dz_dy, dz_dx + 1e-8)
    aspect_sin = torch.sin(aspect_rad)
    aspect_cos = torch.cos(aspect_rad)

    # Second derivatives for curvature
    d2z_dx2 = F.conv2d(dz_dx, kx, padding=1)
    d2z_dy2 = F.conv2d(dz_dy, ky, padding=1)
    d2z_dxy = F.conv2d(dz_dx, ky, padding=1)

    p = dz_dx
    q = dz_dy
    p2_q2 = p ** 2 + q ** 2 + 1e-8

    # 4. Plan Curvature
    plan_curve = -(q ** 2 * d2z_dx2 - 2 * p * q * d2z_dxy + p ** 2 * d2z_dy2) / (p2_q2 ** 1.5)

    # 5. Profile Curvature
    prof_curve = -(p ** 2 * d2z_dx2 + 2 * p * q * d2z_dxy + q ** 2 * d2z_dy2) / (p2_q2 * torch.sqrt(1 + p2_q2))

    # Normalize curvature for gradient stability
    plan_curve = torch.clamp(plan_curve, -5.0, 5.0) / 5.0
    prof_curve = torch.clamp(prof_curve, -5.0, 5.0) / 5.0

    descriptors = torch.cat([slope, aspect_sin, aspect_cos, plan_curve, prof_curve], dim=1)
    return descriptors


def gaussian_blur(dem: torch.Tensor, kernel_size: int = 9, sigma: float = 2.0) -> torch.Tensor:
    """
    Extracts the smooth low-frequency trend component via 2D Gaussian filtering.

    Args:
        dem : [B, 1, H, W] Elevation tensor
        kernel_size : Size of Gaussian filter window
        sigma : Standard deviation of Gaussian distribution

    Returns:
        trend : [B, 1, H, W] Smooth macro-topography trend
    """
    device = dem.device
    dtype = dem.dtype
    radius = kernel_size // 2

    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k1d = torch.exp(-x ** 2 / (2 * sigma ** 2))
    k1d = k1d / k1d.sum()
    k2d = k1d.unsqueeze(1) @ k1d.unsqueeze(0)
    k2d = k2d.view(1, 1, kernel_size, kernel_size)

    trend = F.conv2d(dem, k2d, padding=radius)
    return trend


def upsample_terrain(dem: torch.Tensor, scale_factor: int = 2) -> torch.Tensor:
    """Bilinear upsampling helper for low-resolution DEM inputs."""
    return F.interpolate(dem, scale_factor=scale_factor, mode='bilinear', align_corners=False)
