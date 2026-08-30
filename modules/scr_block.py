"""
Spatially-Conditioned Residual Block (SCRBlock) & Terrain Modulation
=====================================================================

Implements terrain-conditioned feature transformation using dynamic scale
and shift parameters (FiLM style) derived from local physical descriptors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class TerrainModulation(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) driven by 5 physical terrain descriptors.
    Computes spatially adaptive scale (gamma) and shift (beta) maps.
    """

    def __init__(self, num_features: int, descriptor_channels: int = 5):
        super().__init__()
        self.param_gen = nn.Sequential(
            nn.Conv2d(descriptor_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, num_features * 2, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, terrain_desc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, C, H, W] Input feature map
            terrain_desc : [B, 5, H, W] Terrain descriptors

        Returns:
            modulated : [B, C, H, W] Modulated feature map
        """
        params = self.param_gen(terrain_desc)
        gamma, beta = torch.chunk(params, 2, dim=1)
        return x * (1.0 + gamma) + beta


class SCRBlock(nn.Module):
    """
    Spatially-Conditioned Residual Block (SCRBlock).
    Combines 3x3 depthwise-separable convolutions, GELU activation, and TerrainModulation.
    """

    def __init__(self, channels: int, descriptor_channels: int = 5):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.pw1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.act = nn.GELU()
        self.modulation = TerrainModulation(channels, descriptor_channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, terrain_desc: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.pw1(self.conv1(x))
        out = self.act(out)
        out = self.modulation(out, terrain_desc)
        out = self.conv2(out)
        return residual + out
