import math

import torch


def create_lpips_metric(device: str | torch.device | None = None):
    import pyiqa

    return pyiqa.create_metric("lpips", device=device).eval()


def calculate_lpips(
    original_rgb: torch.Tensor,
    recon_rgb: torch.Tensor,
    metric: torch.nn.Module,
) -> float:
    with torch.inference_mode():
        score = metric(recon_rgb, original_rgb).item()
    if not math.isfinite(score):
        raise RuntimeError("LPIPS returned a non-finite value.")
    return score
