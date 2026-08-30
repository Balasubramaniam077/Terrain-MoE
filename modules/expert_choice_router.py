"""
Expert-Choice Router (ECR) with Physics-Supervised Load Balancing
==================================================================

Implements Top-K / Expert-Choice routing over spatial DEM tokens.
Includes:
- ST-MoE z-loss to prevent logit magnitude explosion
- Global Coefficient of Variation (CV^2) importance loss to prevent expert collapse
- Physics-guided KL loss for terrain alignment supervision
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExpertChoiceRouter(nn.Module):
    """
    Expert-Choice Router (ECR) for spatial DEM token assignment.
    """

    def __init__(self, descriptor_channels: int = 5, num_experts: int = 4,
                 capacity_factor: float = 2.0, z_weight: float = 1e-3,
                 imp_weight: float = 0.10, route_weight: float = 0.05,
                 noise_std: float = 0.3):
        super().__init__()
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.z_weight = z_weight
        self.imp_weight = imp_weight
        self.route_weight = route_weight
        self.noise_std = noise_std

        # Gating network mapping terrain descriptors to expert logits
        self.gate_net = nn.Sequential(
            nn.Conv2d(descriptor_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, num_experts, kernel_size=1)
        )

    def forward(self, terrain_desc: torch.Tensor):
        """
        Args:
            terrain_desc : [B, 5, H, W] Terrain descriptors

        Returns:
            routing_weights : [B, K, H, W] Softmax expert weights per pixel
            aux_losses : dict containing 'z_loss', 'imp_loss', and 'route_loss'
        """
        B, C_desc, H, W = terrain_desc.shape
        K = self.num_experts
        N = H * W

        # 1. Compute raw gating logits
        logits = self.gate_net(terrain_desc)  # [B, K, H, W]

        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Softmax probability distribution over experts
        probs = F.softmax(logits, dim=1)  # [B, K, H, W]

        # 2. Compute Auxiliary Losses
        # A. ST-MoE Logit Z-Loss: prevents logit scale drift
        z_loss = self.z_weight * torch.mean(torch.logsumexp(logits, dim=1) ** 2)

        # B. Coefficient of Variation CV^2 Importance Loss (fights dead experts)
        mean_probs = probs.mean(dim=[0, 2, 3])  # Mean probability per expert [K]
        var_probs = torch.var(mean_probs, unbiased=False)
        mean_sq = torch.mean(mean_probs) ** 2 + 1e-8
        imp_loss = self.imp_weight * (var_probs / mean_sq)

        # C. Supervised Physics-Guided Routing Alignment Loss
        # Target prior: Expert 0 (Ridge), Expert 1 (Valley), Expert 2 (Flat), Expert 3 (Rough)
        slope = terrain_desc[:, 0:1]
        plan_curr = terrain_desc[:, 3:4]

        p_ridge = torch.clamp(slope * torch.clamp(plan_curr, min=0), 0, 1)
        p_valley = torch.clamp(slope * torch.clamp(-plan_curr, min=0), 0, 1)
        p_flat = torch.clamp(1.0 - slope, 0, 1)
        p_rough = torch.clamp(slope * (1.0 - torch.abs(plan_curr)), 0, 1)

        target_prior = torch.cat([p_ridge, p_valley, p_flat, p_rough], dim=1)
        target_prior = F.softmax(target_prior, dim=1)

        log_probs = F.log_softmax(logits, dim=1)
        route_loss = self.route_weight * F.kl_div(log_probs, target_prior, reduction='batchmean')

        aux_losses = {
            'z_loss': z_loss,
            'imp_loss': imp_loss,
            'route_loss': route_loss
        }

        return probs, aux_losses
