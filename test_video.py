# video_1fps.py (frames_count controls sampling fps & output fps)
import os
import argparse
import math
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path

from models.generator import UNetGenerator
from utils.mask_utils import m11_to_01


# ==== helpers from your test ====
def snap_up(x: int, mult: int) -> int:
    return (x + mult - 1) // mult * mult

def safe_load_state(model, ckpt_path, device):
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
    top  = (Hc - S) // 2
    left = (Wc - S) // 2
    bot  = top + S
    right= left + S
    canvas = torch.zeros(1, 3, Hc, Wc, device=device)
    canvas[:, :, top:bot, left:right] = img_S
    mask = torch.ones(1, 1, Hc, Wc, device=device)
    mask[:, :, top:bot, left:right] = 0
    return canvas, mask

def forward_with_auto_snap(G, canvas, mask, multiples=(64,128,256,512), pad_mode_canvas="reflect"):
    _, _, Hc, Wc = canvas.shape
    for mult in multiples:
        Hs = snap_up(Hc, mult); Ws = snap_up(Wc, mult)
        pad_r, pad_b = Ws - Wc, Hs - Hc
        try:
            if pad_mode_canvas in ("reflect","replicate"):
                canvas_big = F.pad(canvas, (0,pad_r,0,pad_b), mode=pad_mode_canvas)
            else:
                canvas_big = F.pad(canvas, (0,pad_r,0,pad_b), mode="constant", value=0)
            mask_big = F.pad(mask, (0,pad_r,0,pad_b), mode="constant", value=1)
            cond = torch.cat([canvas_big*(1-mask_big), mask_big], dim=1)
            with torch.no_grad():
                pred_big_m11 = G(cond)
            pred_big = m11_to_01(pred_big_m11).clamp(0,1)
            final_big = pred_big*mask_big + canvas_big*(1-mask_big)
            return final_big[:, :, :Hc, :Wc]
        except RuntimeError as e:
            if "Sizes of tensors must match" in str(e):
                continue
            raise
    raise RuntimeError(f"All multiples failed for target {Hc}x{Wc}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=str, required=True, help="input video path")
    ap.add_argument("--output_dir", type=str, default="results_video", help="output video (no audio)")
    ap.add_argument("--checkpoint", type=str, default="checkpoints/G_epoch_010.pt")
    ap.add_argument("--image_size", type=int, default=192)   # S
    ap.add_argument("--extend", type=int, default=64)        # n (四邊等寬)
    ap.add_argument("--frames_count", type=int, default=1, help="frames sampled per second AND output fps")
    ap.add_argument("--restore_size", action="store_true", help="restore to original aspect ratio size after outpainting")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # model
    G = UNetGenerator(in_ch=4, out_ch=3, ngf=64).to(device)
    safe_load_state(G, args.checkpoint, device)
    G.eval()

    S = int(args.image_size)
    n = int(args.extend)
    Hc = Wc = S + n

    # transforms: SxS
    to_S = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((S, S), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    # IO
    cap = cv2.VideoCapture("data/" + args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {args.input}")

    W0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_cnt = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration_ms = (frame_cnt / max(1e-6, src_fps)) * 1000.0  # 影片總毫秒（估計）
    duration_ms = max(0.0, duration_ms)

    # output fps = frames_count
    target_fps = max(1, int(args.frames_count))
    interval_ms = 1000.0 / target_fps

    # output size（依你原本非等比回放的邏輯）
    scale_h = H0 / S
    scale_w = W0 / S
    if args.restore_size:
        out_h = int(round((S + n) * scale_h))
        out_w = int(round((S + n) * scale_w))
    else:
        out_h = S + n
        out_w = S + n

    save_dir = args.output_dir
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(os.path.join(save_dir, "out_sampled.mp4"), fourcc, float(target_fps), (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to open VideoWriter")

    # save dirs
    if save_dir:
        orig_dir = os.path.join(save_dir, "orig_frames")
        pred_dir = os.path.join(save_dir, "pred_frames")
        os.makedirs(orig_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

    # sample per-second with frames_count frames (time-based seek)
    kept = 0
    num_slots = int(math.floor(duration_ms / interval_ms)) + 1
    for i in range(num_slots):
        t_ms = i * interval_ms
        cap.set(cv2.CAP_PROP_POS_MSEC, t_ms)
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        # (optional) save original frame
        if save_dir:
            cv2.imwrite(os.path.join(orig_dir, f"frame_{kept:06d}.png"), frame_bgr)

        # outpaint: SxS → (S+n)x(S+n)
        img_S = to_S(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)
        canvas, mask = build_center_canvas_and_mask(img_S, S, Hc, Wc, device)
        final = forward_with_auto_snap(G, canvas, mask, multiples=(64,128,256,512), pad_mode_canvas="reflect")

        # resize back to desired (out_h, out_w) — your original non-uniform scaling
        if args.restore_size:
            final_resized = F.interpolate(final, size=(out_h, out_w),
                                          mode='bicubic', align_corners=False, antialias=True)
        else:
            final_resized = final

        out_bgr = (final_resized.squeeze(0).permute(1,2,0).clamp(0,1).cpu().numpy() * 255.0).round().astype("uint8")
        out_bgr = cv2.cvtColor(out_bgr, cv2.COLOR_RGB2BGR)

        if save_dir:
            cv2.imwrite(os.path.join(pred_dir, f"frame_{kept:06d}.png"), out_bgr)

        writer.write(out_bgr)
        kept += 1

    cap.release()
    writer.release()
    print(f"[done] sampled {target_fps} fps → wrote {kept} frames at {target_fps} FPS to {os.path.join(save_dir, "out_sampled.mp4")}")

if __name__ == "__main__":
    main()
