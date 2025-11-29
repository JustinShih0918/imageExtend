# utils/metrics.py
import torch

def calculate_psnr(img1, img2):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two tensor images.
    Args:
        img1, img2: Tensors in range [-1, 1] (GAN output)
    Returns:
        psnr: scalar tensor
    """
    # 1. Convert [-1, 1] to [0, 1] for standard PSNR calculation
    img1 = (img1 + 1.0) / 2.0
    img2 = (img2 + 1.0) / 2.0
    
    # 2. Clamp to ensure range (numerical stability)
    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    # 3. Calculate MSE (Mean Squared Error)
    mse = torch.mean((img1 - img2) ** 2)
    
    # 4. Handle division by zero
    if mse == 0:
        return float('inf')
    
    # 5. PSNR formula: 20 * log10(MAX / sqrt(MSE)), here MAX is 1.0
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    return psnr