"""
TerraMoE Upsampling Decoder Module
==================================

Upsamples latent feature representations to high spatial resolution (2x or 4x)
using Sub-Pixel Convolution (PixelShuffle) and conditions output features
with high-resolution terrain descriptors.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .scr_block import SCRBlock


class TerraMoEDecoder(nn.Module):
    """
    Decoder network with PixelShuffle sub-pixel upscaling.
    """

    def __init__(self, embed_channels: int = 48, up_scale: int = 2, num_blocks: int = 3,
                 descriptor_channels: int = 5, grad_checkpoint: bool = True):
        super().__init__()
        self.up_scale = up_scale
        self.grad_checkpoint = grad_checkpoint

        self.blocks = nn.ModuleList([
            SCRBlock(embed_channels, descriptor_channels) for _ in range(num_blocks)
        ])

        if up_scale == 2:
            self.upsampler = nn.Sequential(
                nn.Conv2d(embed_channels, embed_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU()
            )
        elif up_scale == 4:
            self.upsampler = nn.Sequential(
                nn.Conv2d(embed_channels, embed_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU(),
                nn.Conv2d(embed_channels, embed_channels * 4, kernel_size=3, padding=1),
                nn.PixelShuffle(2),
                nn.GELU()
            )

        self.tail = nn.Sequential(
            nn.Conv2d(embed_channels, embed_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_channels // 2, 1, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor, terrain_desc_hr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, C, H, W] Expert-fused features at LR scale
            terrain_desc_hr : [B, 5, sH, sW] High-resolution terrain descriptors

        Returns:
            res_hr : [B, 1, sH, sW] Reconstructed high-resolution residual map
        """
        feat = self.upsampler(x)  # [B, C, sH, sW]

        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                feat = checkpoint(block, feat, terrain_desc_hr, use_reentrant=False)
            else:
                feat = block(feat, terrain_desc_hr)

        res_hr = self.tail(feat)
        return res_hr
