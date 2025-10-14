# ImageExtend: Deep Learning Image Outpainting

## Project Overview

ImageExtend is a deep learning project that performs image outpainting (also known as image extension), which extends images beyond their original boundaries in a realistic way. The project uses a Generative Adversarial Network (GAN) architecture with a UNet Generator and Patch Discriminator to generate high-quality extended image regions.

## Features

- Extend images beyond their original boundaries
- Train custom models on your own image datasets
- GPU acceleration support for faster training and inference

## Installation

### Prerequisites
- Anaconda or Miniconda

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd imageExtend
```

2. Run the environment setup script:
```bash
chmod +x env_setup.sh
./env_setup.sh
```

This will create a conda environment called `imgext` with all required dependencies.

## Usage

### Activate Environment
```bash
conda activate imgext
```

### Training

1. Place your training images in the `data/train` directory
2. Run the training script:
```bash
python train.py
```

Training progress will be displayed, and sample outputs will be saved in the `outputs` directory. Model checkpoints are saved in the `checkpoints` directory.

## Project Structure

- `models/` - Contains the generator and discriminator model architectures
- `datasets/` - Dataset loading and processing code
- `train.py` - Main training script
- `infer.py` - Script for inference on new images
- `env_setup.sh` - Environment setup script
- `requirements.txt` - Python package dependencies

## Requirements

- Python 3.10
- PyTorch
- torchvision
- Pillow
- numpy
- matplotlib
- pytorch_msssim

## Examples

Training outputs and model progress can be found in the `outputs` directory after training.