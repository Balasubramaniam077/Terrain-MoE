# TerraMoE-v2: Physics-Informed Terrain-Aware Mixture of Experts for DEM Super-Resolution

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official Python implementation accompanying the manuscript submitted to *Computers & Geosciences*:

> **"TerraMoE-v2: Physics-Informed Terrain-Aware Mixture-of-Experts for High-Fidelity Digital Elevation Model Super-Resolution"**

---

## 📂 Repository Directory Layout

```
com&geo/
├── README.md                      # Journal submission documentation & execution guide
├── requirements.txt                # Python environment dependencies
├── config.py                       # Dataclass setup for hyperparameters & relative paths
├── dataset.py                      # Paired DEM data loader + synthetic terrain generator
├── losses.py                       # Physics-informed multi-term DEM loss functions
├── metrics.py                      # Geoscientific evaluation metrics (PSNR, SSIM, RMSE)
├── model.py                        # Top-level TerraMoEv2 PyTorch module assembly
├── train.py                        # Executable training pipeline script
├── eval.py                         # Executable evaluation suite & heatmap renderer
├── utils.py                        # Checkpointing, CSV logging, and plotting utilities
└── modules/                        # Modular neural network building blocks
    ├── __init__.py                 # Package exports
    ├── terrain_ops.py              # Differential operators (Slope, Aspect, Curvature)
    ├── sa_conv.py                  # Aspect-Aligned Spatially Adaptive Convolution
    ├── scr_block.py                # Spatially-Conditioned Residual Block (SCRBlock)
    ├── encoder.py                  # Multi-scale Terrain Features Encoder
    ├── expert_choice_router.py     # Expert-Choice Router (ECR) with load balancing
    ├── heterogeneous_experts.py    # 4 Terrain-specialized experts (Ridge, Valley, Flat, Rough)
    └── decoder.py                  # Sub-pixel PixelShuffle upsampling decoder
```

---

## ⚙️ Installation & Requirements

### 1. Prerequisites
- Python $\ge 3.8$
- PyTorch $\ge 2.0.0$
- CUDA $\ge 11.8$ (Optional, CPU execution supported)

### 2. Environment Setup
Clone the repository and install required packages:

```bash
git clone https://github.com/your-username/TerraMoE-v2.git
cd com&geo
pip install -r requirements.txt
```

---

## 🏋️ Training & Benchmark Evaluation

### 1. Training on Custom DEM Datasets
Organize your paired terrain elevation tiles in NumPy `.npy` or GeoTIFF `.tif` format:
```
data/
├── train/
│   └── x2/
│       ├── lr/  (tile_001.npy, tile_002.npy, ...)
│       └── hr/  (tile_001.npy, tile_002.npy, ...)
└── val/
    └── x2/
        ├── lr/
        └── hr/
```

Launch training for **2× Scale**:
```bash
python train.py --scale 2 --epochs 75 --batch_size 16 --lr 2e-4
```

Launch training for **4× Scale**:
```bash
python train.py --scale 4 --epochs 100 --batch_size 16
```

### 2. Evaluating Model Performance
Evaluate a trained model checkpoint and generate routing heatmaps + elevation error maps:
```bash
python eval.py --checkpoint checkpoints/TerraMoE_v2_x2_Best.pth --scale 2 --save_maps
```

---

## 📊 Geoscientific Evaluation Metrics

Model performance is evaluated across both spatial domain metrics and real-world topographic metrics:

| Metric | Symbol | Description | Unit |
| :--- | :--- | :--- | :--- |
| **Peak Signal-to-Noise Ratio** | PSNR | Reconstruction signal fidelity | dB ($\uparrow$) |
| **Structural Similarity** | SSIM | Macro-topography structural coherence | $[0, 1]$ ($\uparrow$) |
| **Elevation RMSE** | $\text{RMSE}_{\text{ele}}$ | Real-world absolute elevation error | meters ($\downarrow$) |
| **Slope RMSE** | $\text{RMSE}_{\text{slope}}$ | Topographic surface gradient error | degrees ($^\circ$) ($\downarrow$) |
| **Aspect RMSE** | $\text{RMSE}_{\text{aspect}}$ | Surface orientation directional error | degrees ($^\circ$) ($\downarrow$) |


