import torch
import os
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from datasets.image_dataset import ImageExtendDataset
from models.generator import UNetGenerator
from models.discriminator import PatchDiscriminator
import torch.nn.functional as F
import torch.optim as optim

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Data ---
    dataset = ImageExtendDataset("data/train")
    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    if len(dataset) == 0:
        print("No training images found in data/train/")
        return

    # --- Models ---
    G = UNetGenerator().to(device)
    D = PatchDiscriminator().to(device)

    opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    print(f"Training started on {device} with {len(dataset)} images")

    # --- Training ---
    for epoch in range(10):
        for i, batch in enumerate(loader):
            x, y = [b.to(device) for b in batch]
            fake_y = G(x)

            # Train D
            D_real = D(x, y)
            D_fake = D(x, fake_y.detach())
            loss_D = 0.5 * (F.mse_loss(D_real, torch.ones_like(D_real)) +
                            F.mse_loss(D_fake, torch.zeros_like(D_fake)))
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # Train G
            D_fake = D(x, fake_y)
            loss_G_GAN = F.mse_loss(D_fake, torch.ones_like(D_fake))
            loss_G_L1 = F.l1_loss(fake_y, y) * 100
            loss_G = loss_G_GAN + loss_G_L1
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            if i % 10 == 0:
                print(f"Epoch[{epoch+1}] Batch[{i}] | L_D={loss_D.item():.3f} | L_G={loss_G.item():.3f}")
            
            # --- save sample images every few epochs ---
            if (epoch + 1) % 5 == 0:
                os.makedirs("outputs", exist_ok=True)
                sample_fake = fake_y[:4]  # first few samples
                sample_real = y[:4]
                sample_in = x[:4]

                # Combine input, output, target side by side
                save_image(
                    torch.cat([sample_in, sample_fake, sample_real], dim=0),
                    f"outputs/epoch_{epoch+1:03d}.png",
                    nrow=4,
                    normalize=True
                )

            # --- save model checkpoints ---
            if (epoch + 1) % 10 == 0:
                torch.save(G.state_dict(), f"checkpoints/G_epoch_{epoch+1}.pth")
                torch.save(D.state_dict(), f"checkpoints/D_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    main()
