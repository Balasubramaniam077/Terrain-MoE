"""
TerraMoE-v2 — Unified Configuration Module for Computers & Geosciences
========================================================================

Defines configuration parameters for model architecture, physics-informed losses,
expert-choice routing, and training dataset paths.
"""
import os
from dataclasses import dataclass


@dataclass
class TerraMoEv2Config:
    """Configuration dataclass for TerraMoE-v2 Super-Resolution network."""

    # ── Architecture Specifications ───────────────────────────────────────
    up_scale: int = 2                       # Upscaling factor (2x or 4x)
    embed_channels: int = 48                # Embedding dimension C (48 low-compute, 64 full)
    num_experts: int = 4                    # Number of terrain-specialized experts
    encoder_blocks: int = 3                 # Number of SCRBlocks in TerrainEncoder
    decoder_blocks: int = 3                 # Number of SCRBlocks in TerraMoEDecoder
    descriptor_channels: int = 5            # Number of physical terrain descriptors (Slope, Aspect Sin/Cos, Plan/Profile Curvature)

    # ── Trend-Residual Elevation Decomposition ──────────────────────────
    trend_kernel_size: int = 9              # Gaussian blur kernel size for trend extraction
    trend_sigma: float = 2.0                # Gaussian blur standard deviation

    # ── Expert-Choice Router (ECR) Hyperparameters ────────────────────────
    capacity_factor: float = 2.0            # Capacity ratio c (Token capacity T = c * N / K)
    router_z_weight: float = 1e-3           # ST-MoE z-loss weight for logit stability
    router_imp_weight: float = 0.10         # Global importance CV^2 coefficient for load balance
    router_route_weight: float = 0.05       # Supervised terrain-routing KL loss weight
    router_noise_std: float = 0.3           # Exploration Gumbel/Gaussian noise on router logits

    # ── Training & Optimizer Setup ────────────────────────────────────────
    epochs: int = 75                        # Total training epochs
    batch_size: int = 16                    # Mini-batch size
    learning_rate: float = 2e-4             # AdamW initial learning rate
    weight_decay: float = 1e-4              # Weight decay coefficient
    grad_clip: float = 1.0                  # Maximum gradient norm for clipping
    warmup_epochs: int = 5                  # Learning rate warmup epochs
    num_workers: int = 0                    # DataLoader worker threads
    seed: int = 42                          # Random seed for reproducibility
    amp: bool = True                        # Mixed precision training (torch.cuda.amp)
    grad_checkpoint: bool = True            # Gradient checkpointing for memory efficiency

    # ── Multi-Term Physics-Informed Loss Weights ─────────────────────────
    lambda_pix: float = 0.40                # Charbonnier elevation pixel loss
    lambda_ssim: float = 0.15               # Structural similarity loss
    lambda_slope: float = 0.10              # Topographic slope error loss
    lambda_curve: float = 0.08              # Laplacian curvature loss
    lambda_drain: float = 0.05              # Flow accumulation / drainage KL loss
    lambda_freq: float = 0.10               # High-frequency magnitude FFT loss
    lambda_cycle: float = 0.05              # Downsampling cycle-consistency loss
    lambda_itcc: float = 0.15               # Intermediate terrain consistency loss (4x scale only)

    # ── Data Directories (Relative Defaults for Portability) ──────────────
    data_root: str = "./data"
    train_lr_path: str = os.path.join("data", "train", "x2", "lr")
    train_hr_path: str = os.path.join("data", "train", "x2", "hr")
    val_lr_path: str = os.path.join("data", "val", "x2", "lr")
    val_hr_path: str = os.path.join("data", "val", "x2", "hr")
    test_lr_path: str = os.path.join("data", "test", "x2", "lr")
    test_hr_path: str = os.path.join("data", "test", "x2", "hr")

    # ── Experiment Output Directories ─────────────────────────────────────
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"
    activation_maps_dir: str = "results/activation_maps"

    # ── Validation & Evaluation Settings ──────────────────────────────────
    val_freq: int = 1
    pixel_size: float = 30.0                # DEM spatial resolution in meters
    save_best_only: bool = True
    save_name: str = "TerraMoE_v2_Best.pth"

    def __post_init__(self):
        self.update_paths_by_scale()

    def update_paths_by_scale(self):
        """Update dataset relative paths dynamically based on upscaling scale factor."""
        scale = f"x{self.up_scale}"
        self.train_lr_path = os.path.join(self.data_root, "train", scale, "lr")
        self.train_hr_path = os.path.join(self.data_root, "train", scale, "hr")
        self.val_lr_path   = os.path.join(self.data_root, "val",   scale, "lr")
        self.val_hr_path   = os.path.join(self.data_root, "val",   scale, "hr")
        self.test_lr_path  = os.path.join(self.data_root, "test",  scale, "lr")
        self.test_hr_path  = os.path.join(self.data_root, "test",  scale, "hr")
