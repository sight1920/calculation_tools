import math

import torch


def create_dists_metric(device: str | torch.device | None = None):
    import pyiqa

    return pyiqa.create_metric("dists", device=device).eval()


def calculate_dists(
    original_rgb: torch.Tensor,
    recon_rgb: torch.Tensor,
    metric: torch.nn.Module,
) -> float:
    with torch.inference_mode():
        score = metric(recon_rgb, original_rgb).item()
    if not math.isfinite(score):
        raise RuntimeError("DISTS returned a non-finite value.")
    return score
