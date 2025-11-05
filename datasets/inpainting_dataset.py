# datasets/inpainting_dataset.py
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile, UnidentifiedImageError

from utils.mask_utils import make_random_border_mask

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ImageFolderWithMask(Dataset):
    """
    Dataset for image inpainting with random border masks.
    
    This dataset loads images from a directory, validates them, and generates
    random border masks for training an inpainting model.
    
    Args:
        root: Path to the folder containing images
        image_size: Target size to resize images (default: 256)
        max_ratio: Maximum ratio of each side to mask (default: 0.25)
        min_bytes: Minimum file size in bytes to consider (default: 5KB)
    """
    def __init__(self, root, image_size=256, max_ratio=0.25, min_bytes=5*1024):
        self.paths = []
        self.max_ratio = max_ratio
        root = Path(root)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        
        print("Scanning images...")
        count = 0
        for p in root.rglob("*"):
            count += 1
            if count % 1000 == 0:
                print(f"Scanned {count} files...")
            
            if p.suffix.lower() in exts and p.is_file():
                try:
                    # Check file size
                    if p.stat().st_size < min_bytes:
                        continue
                    
                    # 1) Verify image structure
                    with Image.open(p) as im:
                        im.verify()
                    
                    # 2) Reopen and actually decode to ensure load/convert works
                    with Image.open(p) as im:
                        im.load()
                    
                    self.paths.append(p)
                except Exception:
                    # Skip broken images silently
                    continue

        if not self.paths:
            raise FileNotFoundError(f"No valid images found in {root.resolve()}")

        random.shuffle(self.paths)
        
        # Image preprocessing pipeline
        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size), 
                            interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        """
        Returns:
            cond: Tensor of shape (4, H, W) - masked RGB image (3ch) + mask (1ch)
            target: Tensor of shape (3, H, W) - original RGB image in [0, 1]
            mask: Tensor of shape (1, H, W) - binary mask where 1 = region to generate
        """
        tries = 3
        for _ in range(tries):
            p = self.paths[idx]
            try:
                # Reopen and decode to avoid deferred loading errors
                with Image.open(p) as im:
                    im.load()
                    img = im.convert("RGB")
                
                img = self.to_tensor(img)  # (3, H, W)
                _, H, W = img.shape

                # Generate random border mask
                mask = make_random_border_mask(H, W, max_ratio=self.max_ratio)  # (1, H, W)
                
                # Create masked input
                masked = img * (1 - mask)
                
                # Concatenate masked image and mask as condition
                cond = torch.cat([masked, mask], dim=0)  # (4, H, W)
                target = img
                
                return cond, target, mask
                
            except (OSError, UnidentifiedImageError, ValueError):
                # If this image fails, try the next one (circular)
                idx = (idx + 1) % len(self.paths)
                continue
        
        raise RuntimeError(f"Too many failures reading images; last tried: {p}")
