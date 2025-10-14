import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super().__init__()

        def down(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

            # up: ConvTranspose -> BN -> ReLU
        def up(in_c, out_c, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # Encoder
        self.down1 = down(in_channels, features, norm=False)
        self.down2 = down(features, features * 2)
        self.down3 = down(features * 2, features * 4)
        self.down4 = down(features * 4, features * 8)
        self.down5 = down(features * 8, features * 8)
        self.down6 = down(features * 8, features * 8)

        # Decoder
        self.up1 = up(features * 8, features * 8, dropout=True)
        self.up2 = up(features * 16, features * 8, dropout=True)
        self.up3 = up(features * 16, features * 4)
        self.up4 = up(features * 8, features * 2)
        self.up5 = up(features * 4, features)
        self.up6 = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def crop_and_concat(self, upsampled, bypass):
        # crop bypass to match upsampled size (handles odd/non-square shapes)
        _, _, H, W = upsampled.shape
        bypass = TF.center_crop(bypass, [H, W])
        return torch.cat([upsampled, bypass], dim=1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)

        u1 = self.up1(d6)
        u2 = self.up2(self.crop_and_concat(u1, d5))
        u3 = self.up3(self.crop_and_concat(u2, d4))
        u4 = self.up4(self.crop_and_concat(u3, d3))
        u5 = self.up5(self.crop_and_concat(u4, d2))
        out = self.up6(self.crop_and_concat(u5, d1))
        return out
