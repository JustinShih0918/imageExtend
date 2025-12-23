# ImageExtend: Deep Learning Image Outpainting

## Project Overview

ImageExtend is a deep learning project that performs **image outpainting** (image extension beyond original boundaries) using a Generative Adversarial Network (GAN) architecture. The system employs a UNet Generator and Patch Discriminator to realistically extend images in all directions.

---

## Quick Start - Reproducing Results

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/JustinShih0918/imageExtend.git
cd imageExtend

# Setup conda environment (automatically installs all dependencies)
chmod +x env_setup.sh
./env_setup.sh

# Activate the environment
conda activate imgext
```

### 2. Download Pre-trained Model

The pre-trained model is too large for GitHub. Download it from Google Drive:

**Download Link**: [Pre-trained Model (G_epoch_063.pt)](https://drive.google.com/file/d/1bRubCe_ZZlu8Vu95C4BUnEU45e_mm0FO/view?usp=sharing)

After downloading, place the file in the `checkpoints/` directory:
```bash
# Create checkpoints directory if it doesn't exist
mkdir -p checkpoints
# Move the downloaded model to checkpoints/
mv ~/Downloads/G_epoch_063.pt checkpoints/
```

### 3. Test with Pre-trained Model (Inference Only)

Once the model is in place, test immediately:

```bash
# Test on images
python test.py --test_dir data/test --output_dir results_comparison --extend 64 --restore_size

# Test on video
python test_video.py --input test_video.mp4 --output_dir results_video --extend 64
```

Results will be saved in `results_comparison/` (for images) or `results_video/` (for videos).

### 4. Training from Scratch (Optional)

```bash
# Prepare your training data in data/train/ directory
python train.py --data_dir data/train --epochs 20 --batch_size 8
```

---

## Repository Structure

```
imageExtend/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── env_setup.sh                 # Environment setup script
├── env_active.sh                # Quick activation script
│
├── models/                      # Model architectures
│   ├── generator.py             # UNet Generator
│   ├── discriminator.py         # Patch Discriminator
│   └── __init__.py
│
├── datasets/                    # Data loading and processing
│   ├── image_dataset.py         # Image dataset loader
│   ├── inpainting_dataset.py   # Inpainting dataset with masking
│   └── __init__.py
│
├── utils/                       # Utility functions
│   ├── losses.py                # Loss functions (GAN, L1, perceptual)
│   ├── mask_utils.py            # Mask generation utilities
│   ├── metrics.py               # Evaluation metrics (PSNR, SSIM)
│   └── resize_utils.py          # Image resizing utilities
│
├── train.py                     # Training script
├── test.py                      # Testing script (images)
├── test_video.py                # Testing script (videos)
├── resize.py                    # Image resizing utility
│
├── checkpoints/                 # Model checkpoints
│   └── G_epoch_063.pt          # Pre-trained generator (download from Google Drive)
│
├── data/                        # Data directory
│   ├── train/                   # Training images (user-provided)
│   ├── test/                    # Test images (user-provided)
│   └── val/                     # Validation images (optional)
│
├── outputs/                     # Training outputs (sample images per epoch)
├── results_comparison/          # Test results (images)
└── results_video/               # Test results (videos)
    ├── orig_frames/             # Original video frames
    └── pred_frames/             # Extended video frames
```

---

## Dataset Access

### Training Data
- **Required Format**: JPEG/PNG images
- **Location**: Place training images in `data/train/` directory
- **Recommendations**: 
  - Use high-quality images (minimum 256x256 pixels)
  - Diverse content for better generalization
  - Suggested datasets: Places365, [COCO 2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset/data), or custom images

### Test Data
- **Included**: Sample test images are located in `data/test/`
- **Custom Testing**: Add your own images to `data/test/` directory

### Pre-trained Model
- **Download**: [Google Drive Link](https://drive.google.com/file/d/1bRubCe_ZZlu8Vu95C4BUnEU45e_mm0FO/view?usp=sharing)
- **Installation**: Download and place in `checkpoints/G_epoch_063.pt`
- **Training Details**: Trained for 63 epochs on diverse image dataset

---

## Detailed Usage Instructions

### Training

**Basic Training:**
```bash
python train.py --data_dir data/train --epochs 20 --batch_size 8
```

**Training Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--data_dir` | `data/train` | Directory containing training images |
| `--out_dir` | `outputs` | Directory for visualized training outputs |
| `--checkpoint` | `checkpoints` | Directory to save model checkpoints |
| `--epochs` | `20` | Number of training epochs |
| `--batch_size` | `8` | Batch size (reduce if GPU memory is limited) |
| `--image_size` | `256` | Image resize dimension (256x256) |
| `--lr` | `2e-4` | Learning rate for Adam optimizer |
| `--lambda_gan` | `1.0` | Weight for adversarial loss |
| `--lambda_l1` | `100.0` | Weight for L1 reconstruction loss (higher = more fidelity) |

**Training Outputs:**
- Checkpoints saved to: `checkpoints/G_epoch_XXX.pt`, `checkpoints/D_epoch_XXX.pt`
- Sample images per epoch: `outputs/`

### Testing (Images)

**Basic Testing:**
```bash
python test.py --test_dir data/test --output_dir results_comparison --extend 64 --restore_size
```

**Testing Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--test_dir` | `data/test` | Folder containing test images |
| `--output_dir` | `results_comparison` | Where outputs are saved |
| `--image_size` | `192` | Resize dimension before inference |
| `--checkpoint` | `checkpoints/G_epoch_010.pt` | Generator checkpoint path (auto-selects newest if missing) |
| `--extend` | `64` | Padding size for extension |
| `--restore_size` | `False` | Flag: restore to original aspect ratio |

### Testing (Videos)

**Basic Video Testing:**
```bash
python test_video.py --input test_video.mp4 --output_dir results_video --extend 64
```

**Video Testing Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | *Required* | Input video filename |
| `--output_dir` | `results_video` | Output directory for video and frames |
| `--image_size` | `192` | Resize dimension before inference |
| `--checkpoint` | `checkpoints/G_epoch_010.pt` | Generator checkpoint path |
| `--extend` | `64` | Padding size for extension |
| `--frames_count` | `1` | Frame sampling rate (1 = every frame) |
| `--restore_size` | `False` | Flag: restore to original aspect ratio |

**Video Outputs:**
- Extended video: `results_video/output.mp4`
- Original frames: `results_video/orig_frames/`
- Extended frames: `results_video/pred_frames/`

---

## System Requirements

### Dependencies
- **Python**: 3.10+
- **PyTorch**: GPU-accelerated (CUDA recommended)
- **Key Libraries**:
  - torchvision
  - Pillow
  - numpy
  - matplotlib
  - pytorch_msssim (for SSIM metric)
  - opencv-python (for video processing)
  - moviepy (for video I/O)

All dependencies are automatically installed via `env_setup.sh` or can be manually installed:
```bash
pip install -r requirements.txt
```

### Hardware Recommendations
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for training)
- **CPU**: Multi-core processor (for inference)
- **RAM**: 16GB+ recommended
- **Storage**: 10GB+ free space for datasets and outputs

---

## Results and Examples

### Expected Outputs
- **Training Progress**: Visualized sample outputs saved per epoch in `outputs/`
- **Test Results**: Side-by-side comparisons (input vs. extended) in `results_comparison/`
- **Video Results**: Extended videos with original and predicted frames in `results_video/`

### Model Performance
- **Architecture**: UNet Generator with skip connections, Patch Discriminator
- **Loss Functions**: Adversarial loss + L1 reconstruction loss
- **Evaluation Metrics**: PSNR, SSIM (implemented in `utils/metrics.py`)

---

## Key Features

- **Realistic Image Extension**: Extends images in all directions seamlessly
- **Video Support**: Process videos frame-by-frame with temporal consistency
- **Pre-trained Model Available**: Ready-to-use checkpoint downloadable from Google Drive
- **Flexible Training**: Customizable hyperparameters and loss weights
- **GPU Acceleration**: Optimized for CUDA-enabled GPUs
- **Comprehensive Utilities**: Masking, metrics, and loss functions included

---

## Environment Management

```bash
# Activate environment
conda activate imgext

# Deactivate environment
conda deactivate

# Quick activation (if env_active.sh is configured)
source env_active.sh
```

---

## Additional Scripts

- **`resize.py`**: Utility for batch image resizing
- **`PY.py`**: Additional Python utilities

---

## Contributing

This is a course project for ML Final Project. For questions or issues, please contact the repository owner.

---

## License

This project is for educational purposes as part of a Machine Learning course final project.
