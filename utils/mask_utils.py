# utils/mask_utils.py
import random
import torch


def make_random_border_mask(h, w, max_ratio=0.25):
    """
    Create a random border mask for image inpainting.
    
    Returns a binary mask where 1 indicates the region to generate (outer border)
    and 0 indicates the known region.
    
    Args:
        h: Height of the mask
        w: Width of the mask
        max_ratio: Maximum ratio of each side to mask (default: 0.25)
    
    Returns:
        torch.Tensor: Binary mask of shape (1, H, W) with values in {0, 1}
    """
    # Random width for each border side
    t = random.uniform(0, max_ratio)
    b = random.uniform(0, max_ratio)
    l = random.uniform(0, max_ratio)
    r = random.uniform(0, max_ratio)

    top = int(h * t)
    bot = int(h * b)
    left = int(w * l)
    right = int(w * r)

    mask = torch.zeros(1, h, w)
    
    # Apply border mask
    if top > 0:
        mask[:, :top, :] = 1
    if bot > 0:
        mask[:, h - bot:, :] = 1
    if left > 0:
        mask[:, :, :left] = 1
    if right > 0:
        mask[:, :, w - right:] = 1

    # Guarantee not all-zero (fallback to top border)
    if mask.sum() == 0:
        border = max(1, min(h, w) // 32)
        mask[:, :border, :] = 1
    
    return mask


def denorm01_to_m11(x):
    """Convert tensor from [0, 1] range to [-1, 1] range."""
    return x * 2 - 1


def m11_to_01(x):
    """Convert tensor from [-1, 1] range to [0, 1] range."""
    return (x + 1) / 2
