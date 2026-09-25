import math

import numpy as np


def calculate_psnr(
    original_yuv: tuple[np.ndarray, np.ndarray, np.ndarray],
    recon_yuv: tuple[np.ndarray, np.ndarray, np.ndarray],
    peak: int,
) -> dict[str, float]:
    peak_squared = float(peak * peak)
    scores = {}
    names = ("PSNR_Y", "PSNR_U", "PSNR_V")
    for name, reference, reconstructed in zip(names, original_yuv, recon_yuv):
        squared_error = reference.astype(np.float64) - reconstructed.astype(np.float64)
        np.square(squared_error, out=squared_error)
        mse = float(squared_error.mean(dtype=np.float64))
        scores[name] = math.inf if mse == 0.0 else 10.0 * math.log10(peak_squared / mse)

    scores["PSNR_YUV"] = (
        6.0 * scores["PSNR_Y"] + scores["PSNR_U"] + scores["PSNR_V"]
    ) / 8.0
    return scores
