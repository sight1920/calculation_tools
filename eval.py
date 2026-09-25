import argparse
from contextlib import contextmanager
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import BinaryIO

import numpy as np
import torch
import torch.nn.functional as F

from metrics.dists import calculate_dists, create_dists_metric
from metrics.fdim import (
    SUPPORTED_BIT_DEPTHS, calculate_fdim, calculate_fdim_deep,
    calculate_fdim_vmaf, create_fdim_metric,
)
from metrics.lpips import calculate_lpips, create_lpips_metric
from metrics.psnr import calculate_psnr
from metrics.vmaf import VmafMetric


def read_yuv_frame(
    stream: BinaryIO,
    height: int,
    width: int,
    dtype: np.dtype,
    peak: int,
    frame_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_samples = height * width
    uv_samples = (height // 2) * (width // 2)
    samples_per_frame = y_samples + 2 * uv_samples
    frame = np.fromfile(stream, dtype=dtype, count=samples_per_frame)
    if frame.size != samples_per_frame:
        raise ValueError(f"Incomplete input at frame {frame_index} (zero-based).")
    if frame.max() > peak:
        raise ValueError(
            f"Frame {frame_index}: sample exceeds maximum {peak}; "
            "check bit depth, byte order and alignment."
        )

    y = frame[:y_samples].reshape(height, width)
    u = frame[y_samples:y_samples + uv_samples].reshape(height // 2, width // 2)
    v = frame[y_samples + uv_samples:].reshape(height // 2, width // 2)
    return y, u, v


def yuv420_to_rgb(
    frame: tuple[np.ndarray, np.ndarray, np.ndarray],
    peak: int,
    device: torch.device,
) -> torch.Tensor:
    y_plane, u_plane, v_plane = frame
    y = torch.from_numpy(y_plane.astype(np.float32)).to(device)[None, None] / peak
    uv_planes = np.stack((u_plane, v_plane)).astype(np.float32)
    uv = torch.from_numpy(uv_planes).to(device).unsqueeze(0) / peak
    uv = F.interpolate(uv, size=y_plane.shape, mode="nearest")
    cb, cr = uv.chunk(2, dim=1)

    kr, kg, kb = 0.2126, 0.7152, 0.0722
    r = y + (2 - 2 * kr) * (cr - 0.5)
    b = y + (2 - 2 * kb) * (cb - 0.5)
    g = (y - kr * r - kb * b) / kg
    return torch.cat((r, g, b), dim=1).clamp(0.0, 1.0)


@contextmanager
def fdim_rgb_stream(path: Path, height: int, width: int, bit_depth: int, frames: int):
    pixel_format = "yuv420p" if bit_depth == 8 else f"yuv420p{bit_depth}le"
    command = [
        "ffmpeg", "-s", f"{width}x{height}", "-pix_fmt", pixel_format,
        "-i", str(path.resolve()), "-f", "image2pipe", "-pix_fmt", "rgb24",
        "-vcodec", "rawvideo", "-frames:v", str(frames),
        "-nostdin", "-loglevel", "error", "-",
    ]
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors, bufsize=10**8)
        try:
            yield process.stdout
            process.stdout.close()
            try:
                returncode = process.wait(timeout=10)
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"FDIM FFmpeg did not finish: {path}") from error
            if returncode:
                errors.seek(0)
                detail = errors.read().decode("utf-8", errors="replace")[-2000:]
                raise RuntimeError(f"FDIM FFmpeg failed for {path}: {detail}")
        finally:
            process.stdout.close()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()


def read_fdim_rgb_frame(
    stream: BinaryIO, height: int, width: int, device: torch.device, frame_index: int,
) -> torch.Tensor:
    payload = stream.read(height * width * 3)
    if len(payload) != height * width * 3:
        raise RuntimeError(f"FDIM FFmpeg returned an incomplete RGB frame at index {frame_index}.")
    rgb = np.frombuffer(payload, dtype=np.uint8).reshape(height, width, 3).copy()
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).contiguous().to(dtype=torch.float32).div(255)
    return tensor.unsqueeze(0).to(device)


def evaluate_sequences(
    original_path: str | Path,
    recon_path: str | Path,
    height: int,
    width: int,
    frames: int,
    bit_depth: int,
    device: str | torch.device | None = None,
) -> dict[str, float]:
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise ValueError("YUV420 height and width must be positive even integers.")
    if frames <= 0:
        raise ValueError("frames must be a positive integer.")
    if bit_depth not in SUPPORTED_BIT_DEPTHS:
        raise ValueError("FDIM's official VMAF supports bit depths 8, 10, 12 and 16.")
    if height < 32 or width < 32:
        raise ValueError("AlexNet LPIPS requires height and width of at least 32.")

    original_path = Path(original_path)
    recon_path = Path(recon_path)
    dtype = np.dtype("u1" if bit_depth == 8 else "<u2")
    peak = (1 << bit_depth) - 1
    samples_per_frame = height * width + 2 * (height // 2) * (width // 2)
    bytes_per_frame = samples_per_frame * dtype.itemsize
    for path in (original_path, recon_path):
        file_size = path.stat().st_size
        if file_size % bytes_per_frame:
            raise ValueError(
                f"{path}: file size {file_size} is not a multiple of "
                f"the frame size {bytes_per_frame}; check HxW, bit depth "
                "and whether the file is truncated."
            )
        available_frames = file_size // bytes_per_frame
        if available_frames < frames:
            raise ValueError(
                f"{path}: requested {frames} frames, but only "
                f"{available_frames} complete frames are available."
            )

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("FDIM requires the FFmpeg executable on PATH.")

    lpips_metric = create_lpips_metric(device=device)
    dists_metric = create_dists_metric(device=lpips_metric.device)
    fdim_metric = create_fdim_metric(device=lpips_metric.device)
    fdim_deep_scores = []
    names = ("PSNR_Y", "PSNR_U", "PSNR_V", "PSNR_YUV", "LPIPS", "DISTS", "VMAF")
    totals = {name: 0.0 for name in names}
    with (
        VmafMetric(height, width, bit_depth) as vmaf_metric,
        original_path.open("rb") as original,
        recon_path.open("rb") as recon,
        fdim_rgb_stream(original_path, height, width, bit_depth, frames) as original_fdim,
        fdim_rgb_stream(recon_path, height, width, bit_depth, frames) as recon_fdim,
    ):
        for frame_index in range(frames):
            reference = read_yuv_frame(original, height, width, dtype, peak, frame_index)
            reconstructed = read_yuv_frame(recon, height, width, dtype, peak, frame_index)
            frame_scores = calculate_psnr(reference, reconstructed, peak)
            vmaf_metric.add_frame(reference, reconstructed)
            ref_rgb = yuv420_to_rgb(reference, peak, lpips_metric.device)
            rec_rgb = yuv420_to_rgb(reconstructed, peak, lpips_metric.device)
            frame_scores["LPIPS"] = calculate_lpips(ref_rgb, rec_rgb, lpips_metric)
            frame_scores["DISTS"] = calculate_dists(ref_rgb, rec_rgb, dists_metric)
            ref_fdim_rgb = read_fdim_rgb_frame(original_fdim, height, width, lpips_metric.device, frame_index)
            rec_fdim_rgb = read_fdim_rgb_frame(recon_fdim, height, width, lpips_metric.device, frame_index)
            fdim_deep_scores.append(calculate_fdim_deep(ref_fdim_rgb, rec_fdim_rgb, fdim_metric))
            for name, value in frame_scores.items():
                totals[name] += value
        totals["VMAF"] = sum(vmaf_metric.finish())

    scores = {name: total / frames for name, total in totals.items()}
    fdim_vmaf = calculate_fdim_vmaf(original_path, recon_path, height, width, frames, bit_depth)
    scores["FDIM"] = calculate_fdim(float(np.mean(fdim_deep_scores)), fdim_vmaf)
    return scores


def parse_resolution(value: str) -> tuple[int, int]:
    match = re.fullmatch(r"(\d+)\s*[xX×]\s*(\d+)", value.strip())
    if match is None:
        raise argparse.ArgumentTypeError("HxW must be HEIGHTxWIDTH, e.g. 324x576.")
    height, width = map(int, match.groups())
    if height <= 0 or width <= 0 or height % 2 or width % 2:
        raise argparse.ArgumentTypeError("YUV420 height and width must be positive and even.")
    return height, width


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate Y/U/V/YUV PSNR, LPIPS, DISTS, VMAF and FDIM following AVS4 CfE Annex C."
    )
    parser.add_argument(
        "--original_path", type=Path, required=True,
        help="Path to the original planar YUV420 sequence.",
    )
    parser.add_argument(
        "--recon_path", type=Path, required=True,
        help="Path to the reconstructed planar YUV420 sequence.",
    )
    parser.add_argument(
        "--HxW", type=parse_resolution, required=True, metavar="HEIGHTxWIDTH",
        help="Frame height x width, e.g. 324x576.",
    )
    parser.add_argument(
        "--frames", type=int, required=True,
        help="Number of frames to evaluate, starting at frame 0.",
    )
    parser.add_argument(
        "--bit_depth", type=int, required=True, choices=SUPPORTED_BIT_DEPTHS,
        help="Sample bit depth: 8 uses uint8; 10/12/16 use little-endian uint16.",
    )
    args = parser.parse_args()
    height, width = args.HxW
    try:
        scores = evaluate_sequences(
            original_path=args.original_path,
            recon_path=args.recon_path,
            height=height,
            width=width,
            frames=args.frames,
            bit_depth=args.bit_depth,
        )
    except (OSError, ValueError, ImportError, RuntimeError) as error:
        parser.error(str(error))

    print(f"Resolution (HxW): {height}x{width}")
    print(f"Frames: {args.frames}")
    print(f"Bit depth: {args.bit_depth}")
    for name, value in scores.items():
        unit = " dB" if name.startswith("PSNR_") else ""
        print(f"{name}: {value:.6f}{unit}")


if __name__ == "__main__":
    main()
