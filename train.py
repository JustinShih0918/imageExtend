# train.py
import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Import models
from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator

# Import dataset
from datasets.inpainting_dataset import ImageFolderWithMask

# Import utilities
from utils.mask_utils import denorm01_to_m11, m11_to_01


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
