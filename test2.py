import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from models.generator import UNetGenerator
from utils.mask_utils import m11_to_01


def snap_up(x: int, mult: int) -> int:
    """Round x up to nearest multiple of mult."""
    return (x + mult - 1) // mult * mult


def safe_load_state(model, ckpt_path, device):
    """Load checkpoint safely; supports weights_only on newer PyTorch; tolerant to wrappers."""
    if not os.path.isfile(ckpt_path):
        cand = sorted(Path("checkpoints").glob("G_epoch_*.*"))
        if not cand:
            raise FileNotFoundError("No generator checkpoint found in --checkpoint or ./checkpoints/")
        ckpt_path = str(cand[-1])

    try:
        state = torch.load(ckpt_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print("[Warn] load_state_dict mismatches -> missing:", missing, " unexpected:", unexpected)


def build_center_canvas_and_mask(img_S, S, Hc, Wc, device):
    """
    Create (Hc,Wc) canvas with the SxS image CENTERED.
    Mask=1 on extended border, 0 on the centered SxS region.
    """
    # center coordinates
    top  = (Hc - S) // 2
    left = (Wc - S) // 2
    bot  = top + S
    right= left + S

    canvas = torch.zeros(1, 3, Hc, Wc, device=device)
    canvas[:, :, top:bot, left:right] = img_S  # center place

    mask = torch.ones(1, 1, Hc, Wc, device=device)
    mask[:, :, top:bot, left:right] = 0        # inner (original) region is kept
    return canvas, mask


def forward_with_auto_snap(G, canvas, mask, multiples=(64, 128, 256, 512), pad_mode_canvas="reflect"):
    """
    Temporarily pad (right/bottom) up to multiples for UNet, then crop back to (Hc,Wc).
    No resizing is performed; output size == target size.
    pad_mode_canvas: 'reflect' or 'constant' for canvas padding aesthetics.
    """
    _, _, Hc, Wc = canvas.shape
    device = canvas.device

    for mult in multiples:
        Hs = snap_up(Hc, mult)
        Ws = snap_up(Wc, mult)

        pad_right  = Ws - Wc
        pad_bottom = Hs - Hc

        try:
            # Pad canvas（通常用 reflect/replicate 讓邊界更自然）
            if pad_mode_canvas in ("reflect", "replicate"):
                canvas_big = F.pad(canvas, (0, pad_right, 0, pad_bottom), mode=pad_mode_canvas)
            else:
                canvas_big = F.pad(canvas, (0, pad_right, 0, pad_bottom), mode="constant", value=0)

            # Mask 外延的部位一律視為 1（需要生成）
            mask_big = F.pad(mask, (0, pad_right, 0, pad_bottom), mode="constant", value=1)

            cond = torch.cat([canvas_big * (1 - mask_big), mask_big], dim=1)  # (1,4,Hs,Ws)

            with torch.no_grad():
                pred_big_m11 = G(cond)  # (1,3,Hs,Ws)
            pred_big = m11_to_01(pred_big_m11).clamp(0, 1)

            final_big = pred_big * mask_big + canvas_big * (1 - mask_big)
            final = final_big[:, :, :Hc, :Wc]  # crop back to target size (no resize)
            print(f"[Outpaint] snapped multiple={mult} -> model_in=({Hs},{Ws}), crop_back=({Hc},{Wc})")
            return final
        except RuntimeError as e:
            if "Sizes of tensors must match" in str(e):
                print(f"[Retry] multiple={mult} failed due to size mismatch, trying bigger multiple...")
                continue
            raise
    raise RuntimeError(f"All multiples failed for target {Hc}x{Wc}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_dir", type=str, default="data/test")
    ap.add_argument("--output_dir", type=str, default="results_comparison2")
    ap.add_argument("--image_size", type=int, default=64, help="base S; the original gets resized to SxS before extension")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/G_epoch_010.pt")

    # Choose one: --extend n  (四周各延伸 n)；或分別用 --pad_h / --pad_w（上下/左右合計延伸量）
    ap.add_argument("--extend", type=int, default=None, help="extend n on both height and width (final=(S+n)x(S+n))")
    ap.add_argument("--pad_h", type=int, default=0, help="total vertical extension (final height = S + pad_h)")
    ap.add_argument("--pad_w", type=int, default=0, help="total horizontal extension (final width  = S + pad_w)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    # derive pads
    if args.extend is not None:
        pad_h = pad_w = max(0, int(args.extend))
    else:
        pad_h = max(0, int(args.pad_h))
        pad_w = max(0, int(args.pad_w))

    S = int(args.image_size)
    if S <= 0:
        raise ValueError("--image_size must be > 0")

    Hc = S + pad_h
    Wc = S + pad_w
    if Hc == S and Wc == S:
        raise ValueError("No extension set. Use --extend n or --pad_h/--pad_w.")

    # Model
    G = UNetGenerator(in_ch=4, out_ch=3, ngf=64).to(device)
    safe_load_state(G, args.checkpoint, device)
    G.eval()

    # transforms: resize original to SxS
    to_S = transforms.Compose([
        transforms.Resize((S, S), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    for fname in os.listdir(args.test_dir):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            continue
        path = os.path.join(args.test_dir, fname)
        img = Image.open(path).convert("RGB")
        base = Path(fname).stem

        # 1) 原圖縮到 SxS
        img_S = to_S(img).unsqueeze(0).to(device)  # (1,3,S,S)

        # 2) 建立 (S+pad_h, S+pad_w) 畫布，原圖「置中」，外框=mask=1
        canvas, mask = build_center_canvas_and_mask(img_S, S, Hc, Wc, device)

        # 3) 模型推論：臨時 pad 到 64/128/256/512 倍數，再裁回 (Hc,Wc)
        final = forward_with_auto_snap(G, canvas, mask, multiples=(64, 128, 256, 512), pad_mode_canvas="reflect")

        # 4) 輸出（最終尺寸就是 (S+pad_h) x (S+pad_w)）
        out_img = os.path.join(args.output_dir, f"{base}_outpaint_center_{Hc}x{Wc}.png")
        grid_img = os.path.join(args.output_dir, f"{base}_grid_{Hc}x{Wc}.png")

        masked_vis = canvas * (1 - mask)
        vis = torch.cat([masked_vis, mask.repeat(1, 3, 1, 1), final, canvas], dim=0)
        save_image(final, out_img)
        save_image(vis, grid_img, nrow=1, normalize=False)
        print(f"Saved: {out_img}")

if __name__ == "__main__":
    main()
