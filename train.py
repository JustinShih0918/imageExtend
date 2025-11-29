# train.py
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.cuda.amp import autocast, GradScaler  

# Import models
from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator

# Import dataset
from datasets.inpainting_dataset import ImageFolderWithMask

# Import utilities
from utils.mask_utils import denorm01_to_m11, m11_to_01

from utils.metrics import calculate_psnr

import datetime

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data/train_256")
    ap.add_argument("--out_dir", type=str, default="outputs")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32) 
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lambda_gan", type=float, default=1.0)
    ap.add_argument("--lambda_l1", type=float, default=10.0)
    ap.add_argument("--lambda_fm", type=float, default=40.0) 
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Dataset & DataLoader
    ds = ImageFolderWithMask(args.data_dir, image_size=args.image_size)
    
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, 
                    num_workers=4, pin_memory=True, persistent_workers=True)

    # Initialize Models
    G = UNetGenerator(in_ch=4, out_ch=3, ngf=64).to(device)
    D = PatchDiscriminator(in_ch=7, ndf=64).to(device)

    # Optimizers
    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    optD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    
    # Losses
    l1 = nn.L1Loss(reduction="none")
    l1_loss_fn = nn.L1Loss() 


    scaler = GradScaler()

    print(f"Training on {device} with {len(ds)} images using Feature Matching...")

    for epoch in range(1, args.epochs + 1):
        G.train(); D.train()
        for i, (cond, target_01, mask) in enumerate(dl):
            # Move data to GPU
            cond = cond.to(device)          # (N,4,H,W)
            target_01 = target_01.to(device) # (N,3,H,W)
            mask = mask.to(device)          # (N,1,H,W)
            target = denorm01_to_m11(target_01) # [-1,1]

            #  Train Discriminator
  
            optD.zero_grad(set_to_none=True)
            
            with autocast():
                fake = G(cond)
                

                logits_real, _ = D(cond, target)
                logits_fake, _ = D(cond, fake.detach())
                
                # Hinge Loss
                d_loss_real = torch.mean(F.relu(1.0 - logits_real))
                d_loss_fake = torch.mean(F.relu(1.0 + logits_fake))
                d_loss = 0.5 * (d_loss_real + d_loss_fake)
            
            scaler.scale(d_loss).backward()
            scaler.step(optD)
            scaler.update()

            #  Train Generator
            optG.zero_grad(set_to_none=True)
            
            with autocast():
                fake = G(cond)
                
                
                logits_fake, feats_fake = D(cond, fake)
                g_adv = -torch.mean(logits_fake)

                l1_map = l1(fake, target)
                l1_masked = (l1_map * mask).sum() / (mask.sum() * 3 + 1e-6)

                with torch.no_grad():
                    _, feats_real = D(cond, target)
                
                fm_loss = 0.0

                for f_fake, f_real in zip(feats_fake, feats_real):
                    fm_loss += l1_loss_fn(f_fake, f_real)
                

                g_loss = (args.lambda_gan * g_adv) + \
                         (args.lambda_l1 * l1_masked) + \
                         (args.lambda_fm * fm_loss)  
                         
            scaler.scale(g_loss).backward()
            scaler.step(optG)
            scaler.update()
            
            with torch.no_grad():
                    # Calculate PSNR between Generated(fake) and Ground Truth(target)
                    # Note: We can calculate it on the whole image or just the mask region.
                    # Usually, calculating on the whole image is standard.
                    current_psnr = calculate_psnr(fake, target)

            # Log
            if i % 50 == 0: 
                print(f"Epoch[{epoch}/{args.epochs}] Batch[{i}] "
                      f"| L_D={d_loss.item():.3f} | G_adv={g_adv.item():.3f} "
                      f"| L1={l1_masked.item():.3f} | FM={fm_loss.item():.3f}"
                      f"| PSNR={current_psnr.item():.2f} dB") 

        
        # Visualization
        
        G.eval()
        with torch.no_grad():
            try:
                cond_v, target_v, mask_v = next(iter(dl))
            except StopIteration:
                cond_v, target_v, mask_v = next(iter(DataLoader(ds, batch_size=args.batch_size)))

            cond_v = cond_v.to(device); target_v = target_v.to(device)
            out_v = G(cond_v)
            
            vis = torch.cat([
                cond_v[:, :3],                         # Masked Input
                cond_v[:, 3:].repeat(1,3,1,1),         # Mask only
                m11_to_01(out_v).clamp(0,1),           # Output
                target_v                               # Ground Truth
            ], dim=0)
            
            save_image(vis, os.path.join(args.out_dir, f"epoch_{epoch:03d}.png"),
                       nrow=cond_v.shape[0], padding=2)

        # Save checkpoints
        torch.save(G.state_dict(), os.path.join("checkpoints", f"G_epoch_{epoch:03d}.pt"))
        torch.save(D.state_dict(), os.path.join("checkpoints", f"D_epoch_{epoch:03d}.pt"))

if __name__ == "__main__":
    main()