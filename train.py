# train.py
import os
import math
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from PIL import Image, ImageFile, UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ----------------------------
# Utils: mask & transforms
# ----------------------------
def make_random_border_mask(h, w, max_ratio=0.25):
    """
    Return (1,H,W) 0/1 mask: 1 = region to generate (outer border).
    Each side's width is random up to max_ratio of that side.
    """
    t = random.uniform(0, max_ratio)
    b = random.uniform(0, max_ratio)
    l = random.uniform(0, max_ratio)
    r = random.uniform(0, max_ratio)

    top = int(h * t)
    bot = int(h * b)
    left = int(w * l)
    right = int(w * r)

    mask = torch.zeros(1, h, w)
    if top > 0:
        mask[:, :top, :] = 1
    if bot > 0:
        mask[:, h - bot :, :] = 1
    if left > 0:
        mask[:, :, :left] = 1
    if right > 0:
        mask[:, :, w - right :] = 1

    # Guarantee not all-zero
    if mask.sum() == 0:
        border = max(1, min(h, w) // 32)
        mask[:, :border, :] = 1
    return mask


class ImageFolderWithMask(Dataset):
    def __init__(self, root, image_size=256, max_ratio=0.25, min_bytes=5*1024):
        self.paths = []
        self.max_ratio = max_ratio
        root = Path(root)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        count = 0
        print("scanning images...")
        for p in root.rglob("*"):
            count += 1
            if count % 1000 == 0:
                print(f"Scanned {count} files...")
            if p.suffix.lower() in exts and p.is_file():
                try:
                    if p.stat().st_size < min_bytes:
                        continue
                    # 1) 結構驗證
                    with Image.open(p) as im:
                        im.verify()
                    # 2) 重新開啟並實際 decode 一次（確保 load/convert 不會炸）
                    with Image.open(p) as im:
                        im.load()
                except Exception:
                    # print(f"[skip] broken image: {p}")
                    continue
                self.paths.append(p)

        if not self.paths:
            raise FileNotFoundError(f"No valid images found in {root.resolve()}")

        random.shuffle(self.paths)
        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        tries = 3
        for _ in range(tries):
            p = self.paths[idx]
            try:
                # 重新開啟並解碼，避免延遲到 convert 時才炸
                with Image.open(p) as im:
                    im.load()
                    img = im.convert("RGB")
                img = self.to_tensor(img)  # (3,H,W)
                _, H, W = img.shape

                mask = make_random_border_mask(H, W, max_ratio=self.max_ratio)  # (1,H,W)
                masked = img * (1 - mask)
                cond = torch.cat([masked, mask], dim=0)
                target = img
                return cond, target, mask
            except (OSError, UnidentifiedImageError, ValueError):
                # 這張壞就換下一張（循環）
                idx = (idx + 1) % len(self.paths)
                continue
        raise RuntimeError(f"Too many failures reading images; last tried: {p}")

# ----------------------------
# Models: UNet & PatchGAN
# ----------------------------
def conv_block(in_ch, out_ch, ks=4, stride=2, pad=1, norm=True, leaky=True):
    layers = [nn.Conv2d(in_ch, out_ch, ks, stride, pad, bias=not norm)]
    if norm:
        layers.append(nn.BatchNorm2d(out_ch))
    layers.append(nn.LeakyReLU(0.2, inplace=True) if leaky else nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

def deconv_block(in_ch, out_ch, ks=4, stride=2, pad=1, dropout=False):
    layers = [
        nn.ConvTranspose2d(in_ch, out_ch, ks, stride, pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if dropout: layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)

class UNetGenerator(nn.Module):
    """
    Input:  (N,4,H,W) = [masked_rgb(3) + mask(1)] in [0,1] for rgb, {0,1} for mask
    Output: (N,3,H,W) in [-1,1]
    """
    def __init__(self, in_ch=4, out_ch=3, ngf=64):
        super().__init__()
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, ngf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.e2 = conv_block(ngf, ngf*2)
        self.e3 = conv_block(ngf*2, ngf*4)
        self.e4 = conv_block(ngf*4, ngf*8)
        self.e5 = conv_block(ngf*8, ngf*8)
        self.e6 = conv_block(ngf*8, ngf*8)
        self.e7 = conv_block(ngf*8, ngf*8)
        self.e8 = nn.Sequential(nn.Conv2d(ngf*8, ngf*8, 4, 2, 1), nn.ReLU(inplace=True))

        self.d1 = deconv_block(ngf*8, ngf*8, dropout=True)
        self.d2 = deconv_block(ngf*8*2, ngf*8, dropout=True)
        self.d3 = deconv_block(ngf*8*2, ngf*8, dropout=True)
        self.d4 = deconv_block(ngf*8*2, ngf*8)
        self.d5 = deconv_block(ngf*8*2, ngf*4)
        self.d6 = deconv_block(ngf*4*2, ngf*2)
        self.d7 = deconv_block(ngf*2*2, ngf)
        self.out = nn.Sequential(nn.ConvTranspose2d(ngf*2, out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x):
        e1 = self.e1(x);  e2 = self.e2(e1);  e3 = self.e3(e2);  e4 = self.e4(e3)
        e5 = self.e5(e4); e6 = self.e6(e5); e7 = self.e7(e6);  e8 = self.e8(e7)

        d1 = self.d1(e8)
        d2 = self.d2(torch.cat([d1, e7], dim=1))
        d3 = self.d3(torch.cat([d2, e6], dim=1))
        d4 = self.d4(torch.cat([d3, e5], dim=1))
        d5 = self.d5(torch.cat([d4, e4], dim=1))
        d6 = self.d6(torch.cat([d5, e3], dim=1))
        d7 = self.d7(torch.cat([d6, e2], dim=1))
        out = self.out(torch.cat([d7, e1], dim=1))
        return out  # [-1,1]

class PatchDiscriminator(nn.Module):
    """
    Conditional PatchGAN: input = concat([cond(4ch), img(3ch)]) -> (N,1,H',W') logits
    """
    def __init__(self, in_ch=7, ndf=64):
        super().__init__()
        self.c1 = nn.Sequential(nn.Conv2d(in_ch, ndf, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.c2 = conv_block(ndf, ndf*2)
        self.c3 = conv_block(ndf*2, ndf*4)
        self.c4 = conv_block(ndf*4, ndf*8, stride=1)  # keep receptive field
        self.out = nn.Conv2d(ndf*8, 1, 4, 1, 1)

    def forward(self, cond, img):
        x = torch.cat([cond, img], dim=1)
        x = self.c1(x); x = self.c2(x); x = self.c3(x); x = self.c4(x)
        return self.out(x)  # logits

# ----------------------------
# Helpers
# ----------------------------
def denorm01_to_m11(x):  # [0,1] -> [-1,1]
    return x * 2 - 1

def m11_to_01(x):        # [-1,1] -> [0,1]
    return (x + 1) / 2

# ----------------------------
# Training
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/train")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_gan", type=float, default=1.0)
    ap.add_argument("--lambda_l1", type=float, default=100.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    ds = ImageFolderWithMask(args.data_dir, image_size=args.image_size)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    G = UNetGenerator(in_ch=4, out_ch=3, ngf=64).to(device)
    D = PatchDiscriminator(in_ch=7, ndf=64).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss(reduction="none")

    print(f"Training on {device} with {len(ds)} images")

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        for i, (cond, target_01, mask) in enumerate(dl):
            cond = cond.to(device)                 # (N,4,H,W)
            target_01 = target_01.to(device)       # (N,3,H,W)
            mask = mask.to(device)                 # (N,1,H,W)
            target = denorm01_to_m11(target_01)    # [-1,1]

            # --- Train D ---
            with torch.no_grad():
                fake = G(cond)
            logits_real = D(cond, target)
            logits_fake = D(cond, fake.detach())
            valid = torch.ones_like(logits_real)
            fake_lbl = torch.zeros_like(logits_fake)
            d_loss = 0.5 * (bce(logits_real, valid) + bce(logits_fake, fake_lbl))
            optD.zero_grad(set_to_none=True)
            d_loss.backward()
            optD.step()

            # --- Train G ---
            fake = G(cond)
            logits_fake = D(cond, fake)
            g_adv = bce(logits_fake, valid)

            # masked L1 only on the border region (mask==1)
            l1_map = l1(fake, target)                 # (N,3,H,W), in [-1,1] space
            l1_masked = (l1_map * mask).sum() / (mask.sum() * 3 + 1e-6)

            g_loss = args.lambda_gan * g_adv + args.lambda_l1 * l1_masked
            optG.zero_grad(set_to_none=True)
            g_loss.backward()
            optG.step()

            if i % 10 == 0:
                print(f"Epoch[{epoch}/{args.epochs}] Batch[{i}] "
                      f"| L_D={d_loss.item():.3f} | G_adv={g_adv.item():.3f} | L1_mask={l1_masked.item():.3f}")

        # Visualization each epoch
        G.eval()
        with torch.no_grad():
            cond_v, target_v, mask_v = next(iter(dl))
            cond_v = cond_v.to(device); target_v = target_v.to(device)
            out_v = G(cond_v)
            vis = torch.cat([
                cond_v[:, :3],                         # masked RGB
                cond_v[:, 3:].repeat(1,3,1,1),         # mask as 3ch
                m11_to_01(out_v).clamp(0,1),           # prediction
                target_v                                # ground truth
            ], dim=0)
            save_image(vis, os.path.join(args.out_dir, f"epoch_{epoch:03d}.png"),
                       nrow=cond_v.shape[0], padding=2)

        # Save checkpoints each epoch
        torch.save(G.state_dict(), os.path.join("checkpoints", f"G_epoch_{epoch:03d}.pt"))
        torch.save(D.state_dict(), os.path.join("checkpoints", f"D_epoch_{epoch:03d}.pt"))

if __name__ == "__main__":
    main()
