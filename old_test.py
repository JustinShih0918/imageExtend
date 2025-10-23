import os
import torch
from torchvision.utils import save_image
from torchvision import transforms as T
from PIL import Image
from models.generator import UNetGenerator
import torch.nn.functional as F

# Optional: install pytorch-msssim if not yet
# pip install pytorch-msssim
from pytorch_msssim import ssim

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load trained generator ---
G = UNetGenerator(in_channels=3, out_channels=3).to(device)
G.load_state_dict(torch.load("checkpoints/G_epoch_10.pth", map_location=device))
G.eval()

# --- Paths ---
test_dir = "data/test"        # folder with original full images
output_dir = "results_comparison"    # where outputs will be saved
os.makedirs(output_dir, exist_ok=True)

# --- Image transform ---
size_full = (448, 256)
size_crop = (400, 224)

transform = T.Compose([
    T.Resize(size_full),
    T.ToTensor()
])

# --- Function to generate input (simulate cropped / missing border) ---
def create_input(img_tensor, crop_size=size_crop):
    _, H, W = img_tensor.shape
    crop_h, crop_w = crop_size
    top = (H - crop_h) // 2
    left = (W - crop_w) // 2

    # crop and pad back to full size
    cropped = img_tensor[:, top:top + crop_h, left:left + crop_w]
    padded = torch.zeros_like(img_tensor)
    padded[:, top:top + crop_h, left:left + crop_w] = cropped
    return padded

# --- Evaluation function ---
def evaluate(generated, ground_truth, mask=None):
    """
    generated, ground_truth: [B, C, H, W] normalized [0,1]
    mask: optional binary mask for evaluating only extended region
    """
    if mask is not None:
        generated = generated * mask
        ground_truth = ground_truth * mask

    l1 = F.l1_loss(generated, ground_truth).item()
    mse = F.mse_loss(generated, ground_truth)
    psnr_val = 10 * torch.log10(1 / mse).item()
    ssim_val = ssim(generated, ground_truth, data_range=1.0).item()

    return l1, psnr_val, ssim_val

# --- Run testing ---
total_l1, total_psnr, total_ssim = 0, 0, 0
count = 0

for filename in os.listdir(test_dir):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(test_dir, filename)
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img)

    # create input for generator (simulate cropped input)
    input_tensor = create_input(img_tensor)

    # add batch dimension
    input_tensor = input_tensor.unsqueeze(0).to(device)
    img_tensor = img_tensor.unsqueeze(0).to(device)

    # generate output
    with torch.no_grad():
        output_tensor = G(input_tensor)

    # save comparison: [input | generated | ground truth]
    combined = torch.cat([input_tensor, output_tensor, img_tensor], dim=0)
    save_path = os.path.join(output_dir, filename)
    save_image(combined, save_path, nrow=3, normalize=True)
    print(f"Saved comparison: {save_path}")

    # --- Compute metrics ---
    # optionally, mask only the extended border region
    # mask = torch.zeros_like(img_tensor)
    # mask[:, :, top:top+crop_h, left: left+crop_w] = 0
    # mask[:, :, :, :] = 1 - mask  # evaluate only outside input
    # For simplicity, here we evaluate full image
    l1_val, psnr_val, ssim_val = evaluate(output_tensor, img_tensor)
    print(f"L1: {l1_val:.4f}, PSNR: {psnr_val:.2f} dB, SSIM: {ssim_val:.4f}")

    total_l1 += l1_val
    total_psnr += psnr_val
    total_ssim += ssim_val
    count += 1

# --- Print average metrics ---
if count > 0:
    print("\n--- Average Metrics ---")
    print(f"L1: {total_l1/count:.4f}")
    print(f"PSNR: {total_psnr/count:.2f} dB")
    print(f"SSIM: {total_ssim/count:.4f}")
