# models/discriminator.py
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

def conv_block(in_ch, out_ch, ks=4, stride=2, pad=1):
    """
    Helper function to create a convolution block with Spectral Normalization.
    
    We use Spectral Normalization instead of Batch Normalization here because 
    it provides better stability for GAN training, especially in the Discriminator.
    """
    return nn.Sequential(
        # Wrap Conv2d with spectral_norm to constrain the Lipschitz constant
        spectral_norm(nn.Conv2d(in_ch, out_ch, ks, stride, pad, bias=True)),
        nn.LeakyReLU(0.2, inplace=True)
    )

class PatchDiscriminator(nn.Module):
    """
    PatchGAN Discriminator that supports Feature Matching.
    
    It takes the concatenated condition (masked image + mask) and the target image
    (real or fake) and outputs a patch-wise validity map. It also returns
    intermediate feature maps for loss calculation.

    Args:
        in_ch (int): Input channels (default 7: 4 for condition + 3 for image).
        ndf (int): Base number of discriminator filters.

    Returns:
        tuple: (logits, feature_list)
    """
    def __init__(self, in_ch=7, ndf=64):
        super().__init__()

        # Block 1: (N, 7, 256, 256) -> (N, 64, 128, 128)
        # The first layer typically doesn't use Normalization, but Spectral Norm is fine.
        self.block1 = nn.Sequential(
            spectral_norm(nn.Conv2d(in_ch, ndf, 4, 2, 1, bias=True)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Block 2: (N, 64, 128, 128) -> (N, 128, 64, 64)
        self.block2 = conv_block(ndf, ndf*2)

        # Block 3: (N, 128, 64, 64) -> (N, 256, 32, 32)
        self.block3 = conv_block(ndf*2, ndf*4)

        # Block 4: (N, 256, 32, 32) -> (N, 512, 31, 31)
        # Stride=1 is used here to maintain the receptive field size without reducing resolution too much.
        self.block4 = conv_block(ndf*4, ndf*8, stride=1)

        # Output Convolution: (N, 512, 31, 31) -> (N, 1, 30, 30)
        # Produces a 1-channel map of logits (validity scores)
        self.out_conv = spectral_norm(nn.Conv2d(ndf*8, 1, 4, 1, 1, bias=True))

    def forward(self, cond, img):
        """
        Args:
            cond: Condition input (masked image + mask)
            img: Real or Generated image
        """
        # 1. Concatenate condition and image along the channel dimension
        x = torch.cat([cond, img], dim=1)
        
        # 2. Pass through layers and collect intermediate features
        f1 = self.block1(x)
        f2 = self.block2(f1)
        f3 = self.block3(f2)
        f4 = self.block4(f3)
        
        # 3. Final classification logits
        out = self.out_conv(f4)
        
        # [CRITICAL] Return both logits and the list of features.
        # The list [f1, f2, f3, f4] is required for calculating Feature Matching Loss in train.py.
        return out, [f1, f2, f3, f4]