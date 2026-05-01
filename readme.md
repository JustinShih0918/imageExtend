# ImageExtend: GAN Image and Video Outpainting

ImageExtend is a PyTorch project for image outpainting: extending visual content beyond the original image boundaries with a GAN-based model. The repository contains the model architecture, training loop, dataset preprocessing, image inference, video inference, and evaluation utilities.

The trained generator is used by the companion web application: [video_expend_web](https://github.com/JustinShih0918/video_expend_web).

## Overview

This repository focuses on the model pipeline:

- model architecture;
- training loop;
- loss functions;
- dataset preprocessing;
- checkpointing;
- image and video inference scripts;
- quantitative and visual evaluation.

## Features

- **Gated U-Net generator**: encoder-decoder model with skip connections, gated convolutions, dilated bottleneck layers, and RGB output.
- **PatchGAN discriminator**: spectral-normalized discriminator that returns logits and intermediate features.
- **Mask-conditioned outpainting**: model input combines masked RGB content and a binary mask channel.
- **GAN training objective**: combines hinge adversarial loss, masked L1 reconstruction loss, and feature matching loss.
- **Preprocessing pipeline**: optional offline resizing to 256x256 for faster training.
- **Image inference**: extends images from a test directory and saves comparison results.
- **Video inference**: samples video frames, applies outpainting frame by frame, and writes extended output video plus frame folders.
- **Metrics**: PSNR and SSIM utilities for evaluation.
- **Pretrained checkpoint**: downloadable generator weights for inference.

## Architecture

```text
Training images
  -> Resize / dataset preprocessing
  -> Random mask generation
  -> Condition tensor: masked RGB + mask
  -> Gated U-Net generator
  -> PatchGAN discriminator
  -> Hinge GAN loss + masked L1 loss + feature matching loss
  -> Checkpointed generator and discriminator

Inference image or video frame
  -> Resize to input square
  -> Center on larger canvas
  -> Build outpainting mask
  -> Generator predicts missing border content
  -> Merge generated border with original center
  -> Save image/video output
```

## Model Details

### Generator

The generator is implemented in [`models/generator.py`](models/generator.py):

- 4-channel input: RGB image plus mask;
- gated convolution blocks for mask-aware feature learning;
- encoder downsampling stages;
- dilated bottleneck layers for larger receptive fields;
- decoder upsampling with skip connections;
- 3-channel RGB output.

### Discriminator

The discriminator is implemented in [`models/discriminator.py`](models/discriminator.py):

- condition image and generated/real target are concatenated;
- spectral normalization is used in convolution layers;
- intermediate features are returned for feature matching loss;
- final output is a patch-level realism map.

## Tech Stack

| Area | Tools |
| --- | --- |
| Language | Python 3.10+ |
| Deep Learning | PyTorch, TorchVision, AMP mixed precision |
| Computer Vision | OpenCV, Pillow, NumPy |
| Training | GAN, gated U-Net, PatchGAN, hinge loss, feature matching |
| Metrics | PSNR, SSIM |
| Data | COCO 2017 or custom image folders |

## Quick Start

### 1. Environment Setup

```bash
git clone https://github.com/JustinShih0918/imageExtend.git
cd imageExtend
chmod +x env_setup.sh
./env_setup.sh
conda activate imgext
```

You can also install dependencies manually:

```bash
pip install -r requirements.txt
```

### 2. Download Pretrained Model

Download the pretrained generator checkpoint:

[Download G_epoch_063.pt](https://drive.google.com/file/d/1bRubCe_ZZlu8Vu95C4BUnEU45e_mm0FO/view?usp=sharing)

Place it under:

```text
checkpoints/G_epoch_063.pt
```

### 3. Image Inference

```bash
python test.py \
  --test_dir data/test \
  --output_dir results_comparison \
  --extend 64 \
  --restore_size
```

### 4. Video Inference

```bash
python test_video.py \
  --input test_video.mp4 \
  --output_dir results_video \
  --extend 64
```

## Training From Scratch

### 1. Prepare Data

Place training images under:

```text
data/train/
```

Recommended dataset: [COCO 2017](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset/data)

### 2. Preprocess Images

Offline resizing reduces per-epoch training overhead.

```bash
python resize_data.py
```

The processed images are saved to:

```text
data/train_256/
```

### 3. Train

```bash
python train.py \
  --data_dir data/train_256 \
  --epochs 50 \
  --batch_size 16
```

Training saves:

- generator checkpoints: `checkpoints/G_epoch_XXX.pt`
- discriminator checkpoints: `checkpoints/D_epoch_XXX.pt`
- visual samples: `outputs/epoch_XXX.png`

## Training Parameters

| Parameter | Default | Description |
| --- | --- | --- |
| `--data_dir` | `data/train_256` | Directory containing training images |
| `--out_dir` | `outputs` | Directory for visualized training outputs |
| `--epochs` | `50` | Number of training epochs |
| `--batch_size` | `16` | Batch size |
| `--image_size` | `256` | Training image size |
| `--lr` | `2e-4` | Adam learning rate |
| `--lambda_gan` | `1.0` | Adversarial loss weight |
| `--lambda_l1` | `10.0` | Masked L1 reconstruction loss weight |
| `--lambda_fm` | `40.0` | Feature matching loss weight |

## Repository Structure

```text
imageExtend/
├── readme.md
├── requirements.txt
├── env_setup.sh
├── models/
│   ├── generator.py             # Gated U-Net generator
│   └── discriminator.py         # PatchGAN discriminator
├── datasets/
│   ├── image_dataset.py
│   └── inpainting_dataset.py
├── utils/
│   ├── losses.py
│   ├── mask_utils.py
│   ├── metrics.py
│   └── resize_utils.py
├── train.py                     # GAN training loop
├── test.py                      # Image inference
├── test_video.py                # Video inference
├── resize_data.py               # Offline dataset preprocessing
├── checkpoints/                 # Model checkpoints, not committed
├── data/                        # Training/test data, user-provided
├── outputs/                     # Training visualizations
├── results_comparison/          # Image inference outputs
└── results_video/               # Video inference outputs
```

## System Requirements

- Python 3.10+
- PyTorch
- CUDA-capable GPU recommended for training
- 16GB+ RAM recommended
- OpenCV and MoviePy for video processing

## Results and Evaluation

The repository supports both qualitative and quantitative evaluation:

- qualitative samples saved during training;
- side-by-side image comparison outputs;
- video frame output folders;
- PSNR and SSIM metrics through `utils/metrics.py`.

## License

This project was developed as an educational machine learning final project.
