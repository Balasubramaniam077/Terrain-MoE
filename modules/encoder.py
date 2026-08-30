"""
Terrain Feature Encoder Backbone Module
=======================================

Extracts multi-scale representations from elevation input and fuses them with
physical terrain descriptors via stacked SCRBlocks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .scr_block import SCRBlock


class TerrainEncoder(nn.Module):
    """
    Encoder backbone that maps elevation residuals and descriptors into latent feature space C.
    """

    def __init__(self, in_channels: int = 1, embed_channels: int = 48, num_blocks: int = 3,
                 descriptor_channels: int = 5, grad_checkpoint: bool = True):
        super().__init__()
        self.grad_checkpoint = grad_checkpoint

        self.head = nn.Sequential(
            nn.Conv2d(in_channels + descriptor_channels, embed_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(embed_channels, embed_channels, kernel_size=3, padding=1)
        )

        self.blocks = nn.ModuleList([
            SCRBlock(embed_channels, descriptor_channels) for _ in range(num_blocks)
        ])

        self.tail = nn.Conv2d(embed_channels, embed_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, terrain_desc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, 1, H, W] Input elevation residual
            terrain_desc : [B, 5, H, W] Terrain descriptors

        Returns:
            features : [B, embed_channels, H, W] Latent feature representations
        """
        inp = torch.cat([x, terrain_desc], dim=1)
        feat = self.head(inp)
        res = feat

        for block in self.blocks:
            if self.grad_checkpoint and self.training:
                feat = checkpoint(block, feat, terrain_desc, use_reentrant=False)
            else:
                feat = block(feat, terrain_desc)

        feat = self.tail(feat) + res
        return feat
