"""
Physics-Informed Loss Functions for DEM Super-Resolution
=========================================================

Calculates a multi-term loss combining elevation precision, topographic fidelity,
and MoE router stability:
  L_total = λ_pix·L_Charbonnier + λ_ssim·L_SSIM + λ_slope·L_Slope +
            λ_curve·L_Curvature + λ_drain·L_Drainage + λ_freq·L_FFT +
            λ_cycle·L_Cycle + (L_z + L_imp + L_route)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_KX = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_KY = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)


class CharbonnierLoss(nn.Module):
    """Robust Charbonnier L1 penalty."""

    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        diff = pred - gt
        return torch.mean(torch.sqrt(diff * diff + self.eps ** 2))


class SSIMLoss(nn.Module):
    """Structural Similarity Index Measure Loss."""

    def __init__(self, window_size: int = 11):
        super().__init__()
        self.window_size = window_size

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Simple structural SSIM approximation
        mu_x = F.avg_pool2d(pred, kernel_size=self.window_size, stride=1, padding=self.window_size // 2)
        mu_y = F.avg_pool2d(gt, kernel_size=self.window_size, stride=1, padding=self.window_size // 2)

        sigma_x = F.avg_pool2d(pred * pred, kernel_size=self.window_size, stride=1, padding=self.window_size // 2) - mu_x ** 2
        sigma_y = F.avg_pool2d(gt * gt, kernel_size=self.window_size, stride=1, padding=self.window_size // 2) - mu_y ** 2
        sigma_xy = F.avg_pool2d(pred * gt, kernel_size=self.window_size, stride=1, padding=self.window_size // 2) - mu_x * mu_y

        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
        return 1.0 - torch.mean(ssim_map)


class SlopeLoss(nn.Module):
    """Topographic Slope L1 Loss."""

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        device, dtype = pred.device, pred.dtype
        kx = _KX.to(device=device, dtype=dtype)
        ky = _KY.to(device=device, dtype=dtype)

        p_gx = F.conv2d(pred, kx, padding=1)
        p_gy = F.conv2d(pred, ky, padding=1)
        g_gx = F.conv2d(gt, kx, padding=1)
        g_gy = F.conv2d(gt, ky, padding=1)

        p_slope = torch.sqrt(p_gx ** 2 + p_gy ** 2 + 1e-8)
        g_slope = torch.sqrt(g_gx ** 2 + g_gy ** 2 + 1e-8)

        return F.l1_loss(p_slope, g_slope)


class CurvatureLoss(nn.Module):
    """Laplacian Curvature L1 Loss."""

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        laplacian = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
                                 device=pred.device, dtype=pred.dtype).view(1, 1, 3, 3)

        p_lap = F.conv2d(pred, laplacian, padding=1)
        g_lap = F.conv2d(gt, laplacian, padding=1)

        return F.l1_loss(p_lap, g_lap)


class DrainageLoss(nn.Module):
    """Drainage flow accumulation proxy loss."""

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # Negative Laplacian highlights valleys and channels
        p_valleys = F.relu(-F.avg_pool2d(pred, kernel_size=3, stride=1, padding=1) + pred)
        g_valleys = F.relu(-F.avg_pool2d(gt, kernel_size=3, stride=1, padding=1) + gt)
        return F.l1_loss(p_valleys, g_valleys)


class FFTLoss(nn.Module):
    """2D Fast Fourier Transform Magnitude L1 Loss."""

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        p_fft = torch.fft.rfft2(pred)
        g_fft = torch.fft.rfft2(gt)
        p_mag = torch.abs(p_fft)
        g_mag = torch.abs(g_fft)
        return F.l1_loss(p_mag, g_mag)


class TerraMoEv2Loss(nn.Module):
    """
    Combined TerraMoE-v2 Loss Evaluator.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.charbonnier = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.slope = SlopeLoss()
        self.curve = CurvatureLoss()
        self.drain = DrainageLoss()
        self.fft = FFTLoss()

    def forward(self, pred: torch.Tensor, gt: torch.Tensor, x_lr: torch.Tensor,
                pred_2x: torch.Tensor = None, aux_losses: dict = None) -> tuple:
        """
        Calculates total weighted loss and returns loss breakdown dict.
        """
        l_pix = self.charbonnier(pred, gt)
        l_ssim = self.ssim(pred, gt)
        l_slope = self.slope(pred, gt)
        l_curve = self.curve(pred, gt)
        l_drain = self.drain(pred, gt)
        l_freq = self.fft(pred, gt)

        # Cycle-consistency downsampling loss
        scale = pred.shape[-1] // x_lr.shape[-1]
        pred_down = F.interpolate(pred, scale_factor=1.0/scale, mode='bilinear', align_corners=False)
        l_cycle = self.charbonnier(pred_down, x_lr)

        total_loss = (
            self.config.lambda_pix * l_pix +
            self.config.lambda_ssim * l_ssim +
            self.config.lambda_slope * l_slope +
            self.config.lambda_curve * l_curve +
            self.config.lambda_drain * l_drain +
            self.config.lambda_freq * l_freq +
            self.config.lambda_cycle * l_cycle
        )

        # Include router auxiliary losses
        if aux_losses is not None:
            z_loss = aux_losses.get('z_loss', 0.0)
            imp_loss = aux_losses.get('imp_loss', 0.0)
            route_loss = aux_losses.get('route_loss', 0.0)
            total_loss = total_loss + z_loss + imp_loss + route_loss

        breakdown = {
            'total': total_loss.item(),
            'pix': l_pix.item(),
            'ssim': l_ssim.item(),
            'slope': l_slope.item(),
            'curve': l_curve.item(),
            'drain': l_drain.item(),
            'freq': l_freq.item(),
            'cycle': l_cycle.item()
        }

        return total_loss, breakdown
