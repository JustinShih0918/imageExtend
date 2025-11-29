import torch.nn.functional as F

def resize_like(x, ref):
    # x: 要被 resize 的 tensor (N, C, H, W)
    # ref: 目標尺寸的 tensor (N, C_ref, H_ref, W_ref)
    if x.size(2) != ref.size(2) or x.size(3) != ref.size(3):
        x = F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)
    return x