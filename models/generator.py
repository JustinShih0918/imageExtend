# models/generator.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedConv2d(nn.Module):
    """
    OPTIMIZED Gated Convolution:
    Uses a single convolution with 2x output channels and splits them.
    This is much faster than running two separate convolutions.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, dilation=1, activation='elu'):
        super().__init__()
        
        # [OPTIMIZATION] Combine Feature and Gate into one Conv layer
        # Output channels = out_ch * 2
        self.conv = nn.Conv2d(in_ch, out_ch * 2, kernel_size, stride, padding, 
                              dilation=dilation, padding_mode='reflect')
        
        # Instance Norm only acts on the feature part (out_ch)
        self.norm = nn.InstanceNorm2d(out_ch, track_running_stats=False, affine=False)
        
        if activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'leaky':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        else:
            self.activation = nn.Identity()
            
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1. Single Conv forward pass (Faster)
        x = self.conv(x)
        
        # 2. Split the output into Feature and Gate
        feature, gate = torch.chunk(x, 2, dim=1)
        
        # 3. Apply Norm and Activation to Feature only
        feature = self.norm(feature)
        feature = self.activation(feature)
        
        # 4. Apply Sigmoid to Gate
        gate = self.sigmoid(gate)
        
        # 5. Gating
        return feature * gate

class GatedDeconv(nn.Module):
    """
    Upsampling + Optimized Gated Convolution
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = GatedConv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.up(x)
        return self.conv(x)

class UNetGenerator(nn.Module):
    """
    Optimized Gated U-Net
    """
    # [OPTIMIZATION] Default ngf set to 32 for speed
    def __init__(self, in_ch=4, out_ch=3, ngf=32): 
        super().__init__()
        
        # --- Encoder ---
        # c1: Standard Conv is fine for first layer (fastest)
        self.c1 = nn.Conv2d(in_ch, ngf, 5, 2, 2, padding_mode='reflect')
        self.c1_act = nn.ELU()
        
        # c2: 128 -> 64 (if ngf=32)
        self.c2 = GatedConv2d(ngf, ngf*2, stride=2)
        # c3: 64 -> 32
        self.c3 = GatedConv2d(ngf*2, ngf*4, stride=2)
        # c4: 32 -> 16
        self.c4 = GatedConv2d(ngf*4, ngf*8, stride=2)
        
        # --- Bottleneck (Dilated) ---
        self.b1 = GatedConv2d(ngf*8, ngf*8, dilation=2, padding=2)
        self.b2 = GatedConv2d(ngf*8, ngf*8, dilation=4, padding=4)
        self.b3 = GatedConv2d(ngf*8, ngf*8, dilation=8, padding=8)
        self.b4 = GatedConv2d(ngf*8, ngf*8, dilation=4, padding=4)
        self.b5 = GatedConv2d(ngf*8, ngf*8, dilation=2, padding=2)
        
        # --- Decoder ---
        self.d1 = GatedDeconv(ngf*8*2, ngf*4)
        self.d2 = GatedDeconv(ngf*4 + ngf*4, ngf*2)
        self.d3 = GatedDeconv(ngf*2 + ngf*2, ngf)
        self.d4 = GatedDeconv(ngf + ngf, ngf//2)
        
        # Output
        self.out_conv = nn.Sequential(
            nn.Conv2d(ngf//2, out_ch, 3, 1, 1, padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x):
        # Encoder
        c1 = self.c1_act(self.c1(x))
        c2 = self.c2(c1)
        c3 = self.c3(c2)
        c4 = self.c4(c3)
        
        # Bottleneck
        b = self.b1(c4)
        b = self.b2(b)
        b = self.b3(b)
        b = self.b4(b)
        b = self.b5(b)
        
        # Decoder
        d1 = self.d1(torch.cat([b, c4], dim=1))
        d2 = self.d2(torch.cat([d1, c3], dim=1))
        d3 = self.d3(torch.cat([d2, c2], dim=1))
        d4 = self.d4(torch.cat([d3, c1], dim=1))
        
        out = self.out_conv(d4)
        return out