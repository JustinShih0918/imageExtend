# models/discriminator.py
import torch.nn as nn
import torch

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features*2, 4, 2, 1),
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features*2, features*4, 4, 2, 1),
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features*4, 1, 4, 1, 1)
        )

    def forward(self, x, y):
        # concatenate input and output (or real target)
        return self.net(torch.cat([x, y], dim=1))
