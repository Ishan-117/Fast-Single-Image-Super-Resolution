# Fast Single Image Super-Resolution

A deep learning model for 4× image super-resolution, upsampling 32×32 patches to 128×128 using a CNN with sub-pixel convolution. Built with PyTorch as part of the EEU4C16/EEP5C16 Deep Learning module at Trinity College Dublin.

## Overview

Super-resolution recovers high-frequency detail from low-resolution images. This project trains a compact CNN (under 5 million parameters) that balances reconstruction quality against inference speed, targeting near real-time performance (~30 fps).

The model takes a 32×32 RGB patch as input and outputs a 128×128 reconstruction, using learned upsampling via PixelShuffle rather than naive interpolation.

## Model architecture

The network follows a feature extraction → upsampling design:

**Feature extraction**: 6 convolutional blocks with ReLU activations, progressively reducing channel depth from 256 → 128 → 64. All convolutions use 3×3 kernels (first layer uses 5×5) with same-padding to preserve spatial dimensions.

**Upsampling**: A final 3×3 convolution produces 48 channels, followed by PixelShuffle with factor 4 — this rearranges a (B, 48, 32, 32) tensor into (B, 3, 128, 128), performing the 4× upscaling in a single learned operation.

```
Input: (B, 3, 32, 32)
    → Conv 5×5 → 256 channels → ReLU
    → 5× Conv 3×3 blocks (256→256→128→128→64) → ReLU
    → Conv 3×3 → 48 channels
    → PixelShuffle(4)
Output: (B, 3, 128, 128)
```

Parameter count stays under the 5 million limit.

## Training pipeline

**Data**: High-resolution 128×128 image patches provided as a compressed NumPy archive. Low-resolution 32×32 inputs are generated via bicubic downsampling with anti-aliasing using scikit-image.

**Preprocessing**: Pixel values normalised to [0, 1]. Data is memory-mapped from disk for efficiency.

**Augmentation**: Random horizontal/vertical flips and 90° rotations applied to LR–HR pairs jointly to preserve alignment.

**Split**: 80/20 train/validation split with fixed random seed.

**Training**:
- Loss: MSE (L2)
- Optimiser: Adam (lr=0.001)
- Batch size: 16
- Epochs: 30
- Hardware: Google Colab GPU

**Metrics**: PSNR (peak signal-to-noise ratio) and SSIM (structural similarity index) tracked on the validation set each epoch with live plotting.

## Project structure

```
├── models.py          ← SuperResolutionModel class definition
├── model.pth          ← Trained weights (state_dict)
├── Lab_7.ipynb        ← Data processing, training loop, evaluation
└── README.md
```

## How to run

### Prerequisites

Python 3.9+, PyTorch, scikit-image, scikit-learn, matplotlib, tqdm.

### Setup

```bash
pip install torch torchvision scikit-image scikit-learn matplotlib tqdm
```

### Training (Google Colab)

Open `Lab_7.ipynb` in Colab and run cells sequentially. The notebook downloads the dataset (~617 MB), generates LR–HR pairs, and trains the model. Trained weights are saved as `model.pth`.

### Inference

```python
import torch
from models import SuperResolutionModel

model = SuperResolutionModel()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# lr_image: torch.Tensor of shape (1, 3, 32, 32), values in [0, 1]
with torch.no_grad():
    sr_image = model(lr_image)  # (1, 3, 128, 128)
```

## Evaluation metrics

- **PSNR** — measures pixel-level reconstruction accuracy in dB. Higher is better.
- **SSIM** — measures perceptual structural similarity (0 to 1). Higher is better.

Both are computed between the model's super-resolved output and the ground-truth high-resolution image.

## Key design choices

**PixelShuffle over transposed convolution**: PixelShuffle avoids checkerboard artifacts common with `ConvTranspose2d` and is computationally cheaper — a single rearrangement step replaces multiple learned upsampling layers.

**Progressive channel reduction**: Starting wide (256 channels) captures rich features from the small 32×32 input, then narrowing (→128→64) reduces computation before the upsampling stage.

**No batch normalisation**: Omitted to preserve pixel-level detail — batch norm can wash out fine texture information that matters for super-resolution quality.

## References

- Shi, W. et al. "Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network." CVPR 2016 — introduced PixelShuffle
- Google Research. "Enhance! RAISR Sharp Images with Machine Learning." 2016
- Berger et al. "QuickSRNet: Plain Single-Image Super-Resolution Architecture for Faster Inference on Mobile Platforms." CVPR Workshop 2023
