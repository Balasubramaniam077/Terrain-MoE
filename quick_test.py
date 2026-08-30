"""
TerraMoE-v2 Reviewer Quick Verification & Unit Test Suite
==========================================================

Runs an end-to-end verification of the TerraMoE-v2 codebase in < 10 seconds:
- Tests differential terrain operators (Slope, Aspect, Curvature)
- Verifies model forward pass and shape contracts (2x and 4x scale)
- Validates Expert-Choice Router (ECR) gating & auxiliary load balancing losses
- Tests multi-term physics-informed loss computations
- Runs mini 1-epoch training and evaluation pass on synthetic DEM data

Usage:
    python quick_test.py
"""
import sys
import argparse
import torch
import torch.nn.functional as F

from config import TerraMoEv2Config
from modules.terrain_ops import compute_terrain_descriptors, gaussian_blur
from model import build_model
from losses import TerraMoEv2Loss
from metrics import calculate_dem_metrics
from dataset import SyntheticDEMDataset
from utils import save_routing_heatmaps


def run_tests():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help="Target device (cpu or cuda)")
    args, _ = parser.parse_known_args()

    print("===============================================================")
    print(" TerraMoE-v2 — End-to-End Reviewer Automated Verification Suite")
    print("===============================================================")

    if args.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(f"[TEST 1] Environment setup | PyTorch version: {torch.__version__} | Target Device: {device}")

    # ------------------------------------------------------------------
    # Test 2: Terrain Operators
    # ------------------------------------------------------------------
    print("\n[TEST 2] Testing Differential Terrain Operators (Slope, Aspect, Curvature)...")
    dummy_dem = torch.randn(2, 1, 64, 64, device=device)
    descriptors = compute_terrain_descriptors(dummy_dem, pixel_size=30.0)
    assert descriptors.shape == (2, 5, 64, 64), f"Expected shape (2, 5, 64, 64), got {descriptors.shape}"
    print("  -> Terrain Descriptors shape [B=2, C=5, H=64, W=64]: PASS")

    trend = gaussian_blur(dummy_dem, kernel_size=9, sigma=2.0)
    assert trend.shape == (2, 1, 64, 64), f"Expected shape (2, 1, 64, 64), got {trend.shape}"
    print("  -> Trend-Residual Gaussian Blur shape [B=2, C=1, H=64, W=64]: PASS")

    # ------------------------------------------------------------------
    # Test 3: Model Architecture Forward Pass (2x & 4x)
    # ------------------------------------------------------------------
    print("\n[TEST 3] Testing TerraMoEv2 Neural Network Forward Pass...")
    config_2x = TerraMoEv2Config(up_scale=2, embed_channels=32, encoder_blocks=2, decoder_blocks=2)
    model_2x = build_model(config_2x).to(device)
    model_2x.eval()

    with torch.no_grad():
        y_2x, y_intermediate, gates_2x, aux_2x = model_2x(dummy_dem)

    assert y_2x.shape == (2, 1, 128, 128), f"Expected 2x SR shape (2, 1, 128, 128), got {y_2x.shape}"
    assert gates_2x.shape == (2, 4, 64, 64), f"Expected Gates shape (2, 4, 64, 64), got {gates_2x.shape}"
    assert 'z_loss' in aux_2x and 'imp_loss' in aux_2x and 'route_loss' in aux_2x
    print("  -> 2x Scale Model Forward Pass: PASS (Output shape: [2, 1, 128, 128])")
    print("  -> Expert-Choice Routing Gate map shape: PASS (Routing shape: [2, 4, 64, 64])")

    config_4x = TerraMoEv2Config(up_scale=4, embed_channels=32, encoder_blocks=2, decoder_blocks=2)
    model_4x = build_model(config_4x).to(device)
    model_4x.eval()

    with torch.no_grad():
        y_4x, y_2x_inter, gates_4x, aux_4x = model_4x(dummy_dem)

    assert y_4x.shape == (2, 1, 256, 256), f"Expected 4x SR shape (2, 1, 256, 256), got {y_4x.shape}"
    assert y_2x_inter.shape == (2, 1, 128, 128), f"Expected intermediate 2x shape (2, 1, 128, 128), got {y_2x_inter.shape}"
    print("  -> 4x Scale Model Forward Pass: PASS (Output shape: [2, 1, 256, 256])")

    # ------------------------------------------------------------------
    # Test 4: Physics-Informed Multi-Term Loss Computation
    # ------------------------------------------------------------------
    print("\n[TEST 4] Testing Physics-Informed Multi-Term Loss Function...")
    gt_2x = torch.randn(2, 1, 128, 128, device=device)
    loss_evaluator = TerraMoEv2Loss(config_2x).to(device)
    total_loss, breakdown = loss_evaluator(y_2x, gt_2x, dummy_dem, aux_losses=aux_2x)
    assert total_loss.item() > 0
    print(f"  -> Total Loss: {total_loss.item():.4f} | Charbonnier: {breakdown['pix']:.4f} | Slope: {breakdown['slope']:.4f}: PASS")

    # ------------------------------------------------------------------
    # Test 5: Geoscientific Evaluation Metrics
    # ------------------------------------------------------------------
    print("\n[TEST 5] Testing Geoscientific Metrics Evaluation...")
    metrics = calculate_dem_metrics(y_2x[0:1], gt_2x[0:1], lr_min=100.0, lr_max=1500.0, pixel_size=30.0)
    assert 'PSNR' in metrics and 'RMSE_Elevation' in metrics and 'RMSE_Slope' in metrics
    print(f"  -> PSNR: {metrics['PSNR']:.2f} dB | SSIM: {metrics['SSIM']:.4f} | RMSE Ele: {metrics['RMSE_Elevation']:.3f}m | Slope: {metrics['RMSE_Slope']:.2f}°: PASS")

    # ------------------------------------------------------------------
    # Test 6: Mini End-to-End Training Step on Synthetic Terrain Data
    # ------------------------------------------------------------------
    print("\n[TEST 6] Running End-to-End Synthetic Training & Artifact Generation Pass...")
    syn_dataset = SyntheticDEMDataset(num_samples=2, lr_shape=(32, 32), scale=2)
    syn_loader = torch.utils.data.DataLoader(syn_dataset, batch_size=2)

    optimizer = torch.optim.AdamW(model_2x.parameters(), lr=1e-4)
    model_2x.train()

    for batch in syn_loader:
        lr_t = batch['lr'].to(device)
        hr_t = batch['hr'].to(device)

        optimizer.zero_grad()
        pred, _, g_weights, aux_l = model_2x(lr_t)
        loss, _ = loss_evaluator(pred, hr_t, lr_t, aux_losses=aux_l)
        loss.backward()
        optimizer.step()

    print("  -> Backpropagation & Optimizer Gradient Update: PASS")

    save_routing_heatmaps(g_weights[0:1], "results/quick_test_gate_heatmap.png", tile_name="Synthetic_Verification_Tile")
    print("  -> Expert Routing Activation Heatmap Generation: PASS (Saved to results/quick_test_gate_heatmap.png)")

    print("\n===============================================================")
    print(" ALL VERIFICATION TESTS PASSED SUCCESSFULLY! CODEBASE IS READY.")
    print("===============================================================")


if __name__ == '__main__':
    run_tests()
