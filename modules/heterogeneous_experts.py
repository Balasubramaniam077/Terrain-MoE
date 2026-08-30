"""
Heterogeneous Expert Pool Architecture
=======================================

Implements 4 terrain-specialized neural network experts:
- Expert 0 (Ridge/Edge): Directional aspect-aligned SAConv filtering for crisp crest line restoration.
- Expert 1 (Valley/Drainage): Flow-aligned curvature attention for channel and stream continuity.
- Expert 2 (Flat/Smooth): Wide receptive field low-pass smoothing for plateau and basin stability.
- Expert 3 (High-Variance/Rough): Multi-scale dilated convolutions for complex, dissected topography.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .sa_conv import SAConv
from .scr_block import SCRBlock


class RidgeExpert(nn.Module):
    """Expert 0: Ridge & Edge Specialist."""

    def __init__(self, channels: int):
        super().__init__()
        self.sa_conv = SAConv(channels, channels)
        self.scr = SCRBlock(channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, terrain_desc: torch.Tensor) -> torch.Tensor:
        out = self.sa_conv(x, terrain_desc)
        out = self.scr(out, terrain_desc)
        out = self.conv(out)
        return x + out


class ValleyExpert(nn.Module):
    """Expert 1: Valley & Drainage Channel Specialist."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, terrain_desc: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.conv1(x))
        out = out * self.attn(out)
        out = self.conv2(out)
        return x + out


class FlatExpert(nn.Module):
    """Expert 2: Flat Plains & Plateau Specialist."""

    def __init__(self, channels: int):
        super().__init__()
        self.wide_conv = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.pw = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, terrain_desc: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.wide_conv(x))
        out = self.pw(out)
        return x + out


class RoughExpert(nn.Module):
    """Expert 3: Rough & Dissected Terrain Specialist (Multi-Scale Dilated)."""

    def __init__(self, channels: int):
        super().__init__()
        self.d1 = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(channels, channels // 2, kernel_size=3, padding=2, dilation=2)
        self.out_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, terrain_desc: torch.Tensor) -> torch.Tensor:
        out1 = F.gelu(self.d1(x))
        out2 = F.gelu(self.d2(x))
        out = torch.cat([out1, out2], dim=1)
        out = self.out_conv(out)
        return x + out


class ExpertPool(nn.Module):
    """
    Parallel Expert Pool executing heterogeneous expert networks.
    """

    def __init__(self, channels: int, num_experts: int = 4):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            RidgeExpert(channels),
            ValleyExpert(channels),
            FlatExpert(channels),
            RoughExpert(channels)
        ])

    def forward(self, x: torch.Tensor, terrain_desc: torch.Tensor, routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, C, H, W] Input feature tensor
            terrain_desc : [B, 5, H, W] Terrain descriptors
            routing_weights : [B, K, H, W] Routing softmax weights

        Returns:
            fused_output : [B, C, H, W] Expert-fused feature tensor
        """
        fused = torch.zeros_like(x)

        for k in range(self.num_experts):
            w_k = routing_weights[:, k:k+1, :, :]  # [B, 1, H, W]
            expert_out = self.experts[k](x, terrain_desc)
            fused = fused + w_k * expert_out

        return fused
