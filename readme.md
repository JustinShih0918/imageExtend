# ImageExtend: Deep Learning Image Outpainting

## Project Overview

ImageExtend is a deep learning project that performs image outpainting (also known as image extension), which extends images beyond their original boundaries in a realistic way. The project uses a Generative Adversarial Network (GAN) architecture with a UNet Generator and Patch Discriminator to generate high-quality extended image regions.

## Features

- Extend images beyond their original boundaries
- Train custom models on your own image datasets
- GPU acceleration support for faster training and inference

## Installation

### Setup

1. Clone the repository:
```bash
git clone https://github.com/JustinShih0918/imageExtend.git
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

#### To deactivate
```bash
conda deactivate
```

### Training

1. Place your training images in the `data/train` directory
2. Run the training script:
```bash
python train.py --data_dir data/train --out_dir outputs --epochs 20 --batch_size 8
```

You can change some parameter's value.

`--data_dir`（default: data/train）：Directory containing the training images. Each image will be used to generate random border masks during training.

`--out_dir`（default: outputs）：Directory where visualized training outputs (sample images) are saved for each epoch. The actual model checkpoints (G_epoch_xxx.pt, D_epoch_xxx.pt) are saved inside a separate checkpoints/ folder.

`--checkpoint` (default: checkpoints) : Directory where saves checkpoints.

`--epochs`（default: 20）：Number of training epochs (how many full passes over the dataset).

`--batch_size`（default: 8）：Number of images per training batch. If your GPU memory is limited, reduce this value.

`--image_size`（default: 256）：All images are resized to (image_size, image_size) before being fed to the network.

`--lr`（default: 2e-4）：Learning rate for both the generator (G) and discriminator (D) optimizers (Adam).

`--lambda_gan`（default: 1.0）：Weight for the adversarial (GAN) loss term in the generator’s total loss.

`--lambda_l1`（default: 100.0）：Weight for the L1 reconstruction loss, computed only on the masked (missing/outpainted) regions. A higher value emphasizes image fidelity; a lower value emphasizes realism.

Training progress will be displayed, and sample outputs will be saved in the `outputs` directory. Model checkpoints are saved in the `checkpoints` directory.

### Testing
#### output_picture

```bash
python test.py --test_dir data/test --output_dir results --extend 64 --restore_size
```

You can change some parameter's value.   
`--test_dir` (default: data/test)：Folder containing test images.

`--output_dir` (default: results)：Where the output would be saved.

`--image_size` (default: 192)：Images are resized to (image_size, image_size) before inference.

`--checkpoint` (default: checkpoints/G_epoch_010.pt; If missing, the newest in checkpoints/ is used)：Generator checkpoint path to load.

`--extend` (default : 64) : The size of padding to extend lets the result image become (image_size + extend, image_size + extend)

`--restore_size` (default : False) : A flag that if you wrote `--restore_size`, final results are restored to original image ratio, else, it will be (image_size + extend, image_size + extend)

#### output video

```bash
python test_video.py --input test_video.mp4 --output_dir results_video --image_size 192 --extend 64 --frames_count 1
```

You can change some parameter's value.   
`--input` (it is required)：The input video name

`--output_dir` (default: results_video)：Where the output video, original video frames and predicted video frames would be saved.

`--image_size` (default: 192)：Images are resized to (image_size, image_size) before inference.

`--checkpoint` (default: checkpoints/G_epoch_010.pt; If missing, the newest in checkpoints/ is used)：Generator checkpoint path to load.

`--extend` (default : 64) : The size of padding to extend lets the result image become (image_size + extend, image_size + extend)

`--frames_count` (default : 1) : Determines how many fps to take.

`--restore_size` (default : False) : A flag that if you wrote `--restore_size`, final results are restored to original image ratio, else, it will be (image_size + extend, image_size + extend)

## Project Structure

- `models/` - Contains the generator and discriminator model architectures
- `datasets/` - Dataset loading and processing code
- `train.py` - Main training script
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
- opencv-python
- moviepy

## Examples

Training outputs and model progress can be found in the `outputs` directory after training.