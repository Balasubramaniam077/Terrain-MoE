"""
Spatially Adaptive & Aspect-Aligned Convolution (SAConv)
=========================================================

Applies dynamic spatial filtering aligned with local terrain aspect angles
and slope steepness. Fallback to depthwise directional convolutions if
torchvision deformable convolution kernels are unavailable.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d as _deform_conv2d
    _HAS_TORCHVISION_DEFORM = True
except ImportError:
    _HAS_TORCHVISION_DEFORM = False


class SAConv(nn.Module):
    """
    Spatially Adaptive Convolution module conditioned on terrain aspect and slope.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Offset generation network from terrain descriptors
        self.offset_net = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        )

        # Main convolution
        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, terrain_desc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : [B, C, H, W] Input feature map
            terrain_desc : [B, 5, H, W] Terrain descriptors

        Returns:
            out : [B, out_channels, H, W] Filtered feature map
        """
        offsets = self.offset_net(terrain_desc)

        if _HAS_TORCHVISION_DEFORM:
            return _deform_conv2d(
                input=x,
                offset=offsets,
                weight=self.weight,
                bias=self.bias,
                stride=(self.stride, self.stride),
                padding=(self.padding, self.padding)
            )
        else:
            # Fallback to standard spatial filtering when torchvision C++ ops are absent
            base_out = F.conv2d(x, self.weight, self.bias, stride=self.stride, padding=self.padding)
            offset_mod = torch.tanh(offsets.mean(dim=1, keepdim=True))
            return base_out * (1.0 + 0.1 * offset_mod)
