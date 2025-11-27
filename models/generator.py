# models/generator.py
import torch
import torch.nn as nn


def conv_block(in_ch, out_ch, ks=4, stride=2, pad=1, norm=True, leaky=True):
    """Convolution block with optional normalization and activation."""
    layers = [nn.Conv2d(in_ch, out_ch, ks, stride, pad, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=False) if leaky else nn.ReLU(inplace=False))
    return nn.Sequential(*layers)


def deconv_block(in_ch, out_ch, ks=4, stride=2, pad=1, dropout=False):
    """Transposed convolution block with batch normalization and optional dropout."""
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, ks, stride, pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=False),
    ]
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class DilatedBlock(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2, inplace=False)
        )
    def forward(self, x):
        return self.conv(x)

class UNetGenerator(nn.Module):
    """
    U-Net Generator for image inpainting.
    
    Input:  (N, 4, H, W) = [masked_rgb(3) + mask(1)] in [0,1] for RGB, {0,1} for mask
    Output: (N, 3, H, W) in [-1, 1]
    
    Args:
        in_ch: Number of input channels (default: 4 for RGB + mask)
        out_ch: Number of output channels (default: 3 for RGB)
        ngf: Number of generator filters in first conv layer (default: 64)
    """
    def __init__(self, in_ch=4, out_ch=3, ngf=64):
        super().__init__()
        # Encoder
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, ngf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=False))
        self.e2 = conv_block(ngf, ngf*2)
        self.e3 = conv_block(ngf*2, ngf*4)
        self.e4 = conv_block(ngf*4, ngf*8)
        self.e5 = conv_block(ngf*8, ngf*8)
        self.e6 = conv_block(ngf*8, ngf*8)
        self.e7 = conv_block(ngf*8, ngf*8)
        self.e8_bottleneck = nn.Sequential(
            nn.Conv2d(ngf*8, ngf*8, 4, 2, 1), 
            nn.LeakyReLU(0.2, inplace=True),
            DilatedBlock(ngf*8, dilation=2),
            DilatedBlock(ngf*8, dilation=4),
            DilatedBlock(ngf*8, dilation=8),
            
            nn.ReLU(inplace=True)
        )

        # Decoder with skip connections
        self.d1 = deconv_block(ngf*8, ngf*8, dropout=True)
        self.d2 = deconv_block(ngf*8*2, ngf*8, dropout=True)
        self.d3 = deconv_block(ngf*8*2, ngf*8, dropout=True)
        self.d4 = deconv_block(ngf*8*2, ngf*8)
        self.d5 = deconv_block(ngf*8*2, ngf*4)
        self.d6 = deconv_block(ngf*4*2, ngf*2)
        self.d7 = deconv_block(ngf*2*2, ngf)
        self.out = nn.Sequential(nn.ConvTranspose2d(ngf*2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8_bottleneck(e7)

        # Decoder with skip connections
        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], dim=1))
        d3 = self.d3(torch.cat([d2, e6], dim=1))
        d4 = self.d4(torch.cat([d3, e5], dim=1))
        d5 = self.d5(torch.cat([d4, e4], dim=1))
        d6 = self.d6(torch.cat([d5, e3], dim=1))
        d7 = self.d7(torch.cat([d6, e2], dim=1))
        out = self.out(torch.cat([d7, e1], dim=1))
        return out  # Output in [-1, 1]
