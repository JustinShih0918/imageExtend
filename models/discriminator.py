# models/discriminator.py
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, ks=4, stride=2, pad=1, norm=True, leaky=True):
    """Convolution block with optional normalization and activation."""
    layers = [nn.Conv2d(in_ch, out_ch, ks, stride, pad, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class PatchDiscriminator(nn.Module):
    """
    Conditional PatchGAN Discriminator.
    
    Takes concatenated condition and image as input, outputs a patch-wise prediction map.
    
    Args:
        in_ch: Number of input channels (default: 7 for cond(4) + img(3))
        ndf: Number of discriminator filters in first conv layer (default: 64)
    
    Input:  concat([cond(4ch), img(3ch)]) -> (N, 7, H, W)
    Output: (N, 1, H', W') patch logits
    """
    def __init__(self, in_ch=7, ndf=64):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.c2 = conv_block(ndf, ndf*2)
        self.c3 = conv_block(ndf*2, ndf*4)
        self.c4 = conv_block(ndf*4, ndf*8, stride=1)  # keep receptive field
        self.out = nn.Conv2d(ndf*8, 1, 4, 1, 1)

    def forward(self, cond, img):
        """
        Args:
            cond: Condition input (N, 4, H, W) - masked image + mask
            img: Image input (N, 3, H, W) - real or generated image
        
        Returns:
            Patch-wise logits (N, 1, H', W')
        """
        x = torch.cat([cond, img], dim=1)
        x = self.c1(x)
        x = self.c2(x)
        x = self.c3(x)
        x = self.c4(x)
        return self.out(x)  # logits
