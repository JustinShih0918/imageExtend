import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class PatchDiscriminator(nn.Module):
    """
    Deep PatchGAN Discriminator.
    Optimized for structure consistency (Indoor scenes) and Feature Matching.
    """
    def __init__(self, in_ch=7, ndf=32, n_layers=4): # ndf=32 (Lightweight), n_layers=4 (Deep)
        super().__init__()
        
        self.n_layers = n_layers
        self.layers = nn.ModuleList()
        
        # 1. Initial Layer (No Norm, 4x4, stride 2)
        self.layers.append(
            nn.Sequential(
                spectral_norm(nn.Conv2d(in_ch, ndf, 4, 2, 1)),
                nn.LeakyReLU(0.2, inplace=False)
            )
        )
        
        # 2. Downsampling Layers
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            
            self.layers.append(
                nn.Sequential(
                    spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 2, 1)),
                    nn.LeakyReLU(0.2, inplace=False)
                )
            )
            
        # 3. Intermediate Layer (Stride=1, adjust channels)
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        self.layers.append(
            nn.Sequential(
                spectral_norm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, 4, 1, 1)),
                nn.LeakyReLU(0.2, inplace=False)
            )
        )
        
        # 4. Output Layer (1 channel prediction map)
        self.out_conv = spectral_norm(nn.Conv2d(ndf * nf_mult, 1, 4, 1, 1))

    def forward(self, cond, img):
        # Concatenate condition (masked image + mask) and input (real/fake)
        x = torch.cat([cond, img], dim=1)
        
        features = []
        
        # Pass through layers and collect features for Feature Matching Loss
        for layer in self.layers:
            x = layer(x)
            features.append(x)
            
        # Final output (Real/Fake score)
        out = self.out_conv(x)
        
        # Return [Logits, Feature_List]
        return out, features