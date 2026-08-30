"""
TerraMoE-v2 — Top-Level PyTorch Architecture Assembly
=====================================================

Connects:
1. Low-frequency trend & high-frequency residual decomposition
2. Differential physical terrain descriptor extraction
3. Multi-scale TerrainEncoder backbone
4. Expert-Choice Router (ECR) for pixel token gating
5. Heterogeneous ExpertPool (Ridge, Valley, Flat, Rough)
6. TerraMoEDecoder upsampling & residual recombination
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.terrain_ops import compute_terrain_descriptors, gaussian_blur, upsample_terrain
from modules.encoder import TerrainEncoder
from modules.heterogeneous_experts import ExpertPool
from modules.expert_choice_router import ExpertChoiceRouter
from modules.decoder import TerraMoEDecoder
from config import TerraMoEv2Config


class TerraMoEv2(nn.Module):
    """
    TerraMoE-v2 end-to-end network for DEM Super-Resolution.
    """

    def __init__(self, config: TerraMoEv2Config = None):
        super().__init__()
        if config is None:
            config = TerraMoEv2Config()
        self.config = config
        self.scale = config.up_scale
        C = config.embed_channels
        K = config.num_experts

        self.trend_k = config.trend_kernel_size
        self.trend_sigma = config.trend_sigma
        self.pixel_size = config.pixel_size

        # 1. Feature Encoder Backbone
        self.encoder = TerrainEncoder(
            in_channels=1,
            embed_channels=C,
            num_blocks=config.encoder_blocks,
            descriptor_channels=config.descriptor_channels,
            grad_checkpoint=config.grad_checkpoint
        )

        # 2. Expert Choice Router
        self.router = ExpertChoiceRouter(
            descriptor_channels=config.descriptor_channels,
            num_experts=K,
            capacity_factor=config.capacity_factor,
            z_weight=config.router_z_weight,
            imp_weight=config.router_imp_weight,
            route_weight=config.router_route_weight,
            noise_std=config.router_noise_std
        )

        # 3. Heterogeneous Expert Pool
        self.expert_pool = ExpertPool(channels=C, num_experts=K)

        # 4. Upsampling Decoder
        self.decoder = TerraMoEDecoder(
            embed_channels=C,
            up_scale=config.up_scale,
            num_blocks=config.decoder_blocks,
            descriptor_channels=config.descriptor_channels,
            grad_checkpoint=config.grad_checkpoint
        )

    def forward(self, x_lr: torch.Tensor):
        """
        Args:
            x_lr : [B, 1, H, W] Low-resolution input DEM tensor

        Returns:
            y      : [B, 1, sH, sW] Predicted High-Resolution DEM
            y_2x   : [B, 1, 2H, 2W] Intermediate 2x output (4x scale only) or None
            gates  : [B, K, H, W] Softmax expert routing weights map
            aux    : dict of auxiliary router losses ('z_loss', 'imp_loss', 'route_loss')
        """
        # 1. Macro-Trend & High-Frequency Residual Decomposition
        trend_lr = gaussian_blur(x_lr, self.trend_k, self.trend_sigma)
        res_lr = x_lr - trend_lr

        # 2. Extract Low-Resolution Terrain Descriptors
        desc_lr = compute_terrain_descriptors(x_lr, pixel_size=self.pixel_size)

        # 3. Encode Features
        feat_lr = self.encoder(res_lr, desc_lr)

        # 4. Route Features to Experts
        routing_weights, aux_losses = self.router(desc_lr)

        # 5. Process through Heterogeneous Expert Pool
        expert_feat_lr = self.expert_pool(feat_lr, desc_lr, routing_weights)

        # 6. Upsample Low-Res Trend & Descriptors
        trend_hr = upsample_terrain(trend_lr, scale_factor=self.scale)
        desc_hr = compute_terrain_descriptors(trend_hr, pixel_size=self.pixel_size / self.scale)

        # 7. Decode Residual Map
        res_hr = self.decoder(expert_feat_lr, desc_hr)

        # 8. Final Elevation Recombination
        y = trend_hr + res_hr

        y_2x = None
        if self.scale == 4:
            trend_2x = upsample_terrain(trend_lr, scale_factor=2)
            y_2x = trend_2x + F.interpolate(res_hr, scale_factor=0.5, mode='bilinear', align_corners=False)

        return y, y_2x, routing_weights, aux_losses


def build_model(config: TerraMoEv2Config = None) -> TerraMoEv2:
    """Helper factory function to construct TerraMoEv2 model instance."""
    return TerraMoEv2(config=config)
