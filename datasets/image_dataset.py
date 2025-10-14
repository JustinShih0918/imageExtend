import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import random

class ImageExtendDataset(Dataset):
    def __init__(self, image_dir, size_full=(448, 256), size_crop=(400, 224)):
        """
        Args:
            image_dir (str): Path to the folder containing images
            size_full (tuple): target resize for full images (H, W)
            size_crop (tuple): random crop size to simulate missing area
        """
        self.image_dir = image_dir
        self.image_files = [
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        self.size_full = size_full
        self.size_crop = size_crop

        # Transforms: resize full image to fixed size and convert to tensor
        self.transform = T.Compose([
            T.Resize(size_full),
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        img = Image.open(img_path).convert("RGB")

        # Resize
        img = self.transform(img)  # Tensor: (C, H, W)
        _, H, W = img.shape

        crop_h, crop_w = self.size_crop

        # Random top-left corner for cropping
        top = random.randint(0, H - crop_h)
        left = random.randint(0, W - crop_w)

        # Crop the smaller region (simulate input)
        x = img[:, top:top + crop_h, left:left + crop_w]

        # Pad back to full size (centered in a larger zero tensor)
        padded = torch.zeros_like(img)
        padded[:, top:top + crop_h, left:left + crop_w] = x

        # (x = cropped/incomplete input, y = full image target)
        return padded, img
