# test.py
import os
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

# pip install pytorch-msssim
from pytorch_msssim import ssim as ssim_fn

# =============== Models (same as train.py) ===============
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
        return self.out(torch.cat([d7, e1], dim=1))  # [-1,1]

# =============== Mask makers (random & center-crop-like) ===============
def make_random_border_mask(h, w, max_ratio=0.25):
    import random
    t = int(h * random.uniform(0, max_ratio))
    b = int(h * random.uniform(0, max_ratio))
    l = int(w * random.uniform(0, max_ratio))
    r = int(w * random.uniform(0, max_ratio))
    m = torch.zeros(1, h, w)
    if t: m[:, :t, :] = 1
    if b: m[:, h-b:, :] = 1
    if l: m[:, :, :l] = 1
    if r: m[:, :, w-r:] = 1
    if m.sum() == 0:
        border = max(1, min(h,w)//32)
        m[:, :border, :] = 1
    return m

def make_center_crop_border_mask(h, w, crop_h, crop_w):
    """
    Build a mask that's 1 in the OUTER border (outside the central crop),
    and 0 inside the crop. This mirrors your old "crop and pad back" logic.
    """
    top  = (h - crop_h) // 2
    left = (w - crop_w) // 2
    m = torch.ones(1, h, w)
    m[:, top:top+crop_h, left:left+crop_w] = 0
    return m

# =============== Helpers ===============
def m11_to_01(x):
    return (x + 1) / 2

def evaluate(generated_01, gt_01, mask=None):
    """
    Both tensors in [0,1]. If mask provided (1 in border region),
    metrics are computed ONLY on masked region.
    """
    if mask is not None:
        # Avoid empty division if mask accidentally zero
        if mask.sum() < 1:
            mask = None

    if mask is not None:
        generated_01 = generated_01 * mask
        gt_01 = gt_01 * mask

    l1 = F.l1_loss(generated_01, gt_01).item()
    mse = F.mse_loss(generated_01, gt_01)
    psnr_val = 10 * torch.log10(torch.tensor(1.0, device=mse.device) / (mse + 1e-12)).item()
    ssim_val = ssim_fn(generated_01, gt_01, data_range=1.0).item()
    return l1, psnr_val, ssim_val

# =============== Main Test ===============
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", type=str, default="data/test")
    ap.add_argument("--output_dir", type=str, default="results_comparison")
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--checkpoint", type=str, default="checkpoints/G_epoch_010.pt")
    ap.add_argument("--mask_mode", type=str, choices=["center", "random"], default="center")
    ap.add_argument("--center_full_w", type=int, default=448)  # for compatibility with your old numbers
    ap.add_argument("--center_full_h", type=int, default=256)
    ap.add_argument("--center_crop_w", type=int, default=400)
    ap.add_argument("--center_crop_h", type=int, default=224)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # Model
    G = UNetGenerator(in_ch=4, out_ch=3, ngf=64).to(device)
    # Be generous: allow both .pt and .pth checkpoints
    ckpt_path = args.checkpoint
    if not os.path.isfile(ckpt_path):
        # try a few fallbacks
        cand = sorted(Path("checkpoints").glob("G_epoch_*.*"))
        if not cand:
            raise FileNotFoundError("No generator checkpoint found in --checkpoint or ./checkpoints/")
        ckpt_path = str(cand[-1])
    state = torch.load(ckpt_path, map_location=device)
    G.load_state_dict(state)
    G.eval()

    to_tensor = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    total_full = {"l1":0.0, "psnr":0.0, "ssim":0.0}
    total_mask = {"l1":0.0, "psnr":0.0, "ssim":0.0}
    count = 0

    for fname in os.listdir(args.test_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            continue
        path = os.path.join(args.test_dir, fname)
        img = Image.open(path).convert("RGB")
        img_t = to_tensor(img).unsqueeze(0).to(device)   # (1,3,H,W) in [0,1]
        _, _, H, W = img_t.shape

        if args.mask_mode == "center":
            # emulate your old center-crop numbers, scaled to current resolution
            # We compute a crop ratio from the old sizes and apply to current size.
            crop_ratio_h = args.center_crop_h / max(1, args.center_full_h)
            crop_ratio_w = args.center_crop_w / max(1, args.center_full_w)
            crop_h = max(1, int(round(H * crop_ratio_h)))
            crop_w = max(1, int(round(W * crop_ratio_w)))
            mask = make_center_crop_border_mask(H, W, crop_h, crop_w).unsqueeze(0).to(device)  # (1,1,H,W)
        else:
            mask = make_random_border_mask(H, W, max_ratio=0.25).unsqueeze(0).to(device)       # (1,1,H,W)

        # Condition: concatenate masked image + mask
        masked = img_t * (1 - mask)                     # (1,3,H,W)
        cond   = torch.cat([masked, mask], dim=1)       # (1,4,H,W)

        with torch.no_grad():
            pred_m11 = G(cond)                          # [-1,1]
        pred_01 = m11_to_01(pred_m11).clamp(0,1)

        # Compose only masked part from prediction to show realistic completion
        final = pred_01 * mask + img_t * (1 - mask)

        # Save grid: [input | mask | output | gt]
        vis = torch.cat([masked, mask.repeat(1,3,1,1), final, img_t], dim=0)
        save_image(vis, os.path.join(args.output_dir, fname), nrow=1, normalize=False)
        print(f"Saved comparison: {os.path.join(args.output_dir, fname)}")

        # Metrics: full image and masked region only
        l1_full, psnr_full, ssim_full = evaluate(final, img_t, mask=None)
        l1_mask, psnr_mask, ssim_mask = evaluate(final, img_t, mask=mask)

        print(f"[Full ] L1: {l1_full:.4f}, PSNR: {psnr_full:.2f} dB, SSIM: {ssim_full:.4f}")
        print(f"[Mask ] L1: {l1_mask:.4f}, PSNR: {psnr_mask:.2f} dB, SSIM: {ssim_mask:.4f}")

        total_full["l1"]  += l1_full
        total_full["psnr"]+= psnr_full
        total_full["ssim"]+= ssim_full

        total_mask["l1"]  += l1_mask
        total_mask["psnr"]+= psnr_mask
        total_mask["ssim"]+= ssim_mask

        count += 1

    if count > 0:
        print("\n--- Average Metrics ---")
        print(f"[Full ] L1: {total_full['l1']/count:.4f}, PSNR: {total_full['psnr']/count:.2f} dB, SSIM: {total_full['ssim']/count:.4f}")
        print(f"[Mask ] L1: {total_mask['l1']/count:.4f}, PSNR: {total_mask['psnr']/count:.2f} dB, SSIM: {total_mask['ssim']/count:.4f}")

if __name__ == "__main__":
    main()
