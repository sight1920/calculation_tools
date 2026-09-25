from functools import lru_cache
import hashlib
import importlib
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import ModuleType
import xml.etree.ElementTree as ET

import torch


COMMIT = "e01944a363089a636ed3ce092b1d20cf475d44f2"
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "third_party" / f"avs-cvqa-{COMMIT}"
SUPPORTED_BIT_DEPTHS = (8, 10, 12, 16)
PINNED_FILES = {
    "fdim/dist/deep_model.py": "9977d8d584a16389e7625c3ad505f4353cb07fa1766f965bf20c6b3467e76f02",
    "fdim/utils/utils.py": "c9ae47ee57643e6f2001b96a909a714fdb36767c7a6504107324e50ea84b487d",
    "fdim/utils/reader.py": "9a8571336c3fd4afd706059dd24eef33508beff0b1c1baf9dda35e24e5e8b040",
    "fdim/dist/checkpoints/dist_5.0.0.ckpt": "def2615203d1bc86645bda4657068f493169c9f3d0dbb7b8224f04b1046c7eb5",
    "fdim/vmaf/vmaf": "db9cf25254ff6a7b271574844389fef04df06d7892428e21f1b2fe50661ed86f",
}


def verify_installation() -> None:
    if sys.platform != "linux" or platform.machine().lower() not in ("x86_64", "amd64"):
        raise RuntimeError("The pinned FDIM VMAF executable requires Linux x86_64.")
    for relative, expected in PINNED_FILES.items():
        path = SOURCE_ROOT / relative
        if not path.is_file():
            raise RuntimeError(f"Missing FDIM file: {path}. Run code/setup_fdim.py first.")
        with path.open("rb") as stream:
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"FDIM official file SHA256 mismatch: {path}")


@lru_cache(maxsize=1)
def _official_modules():
    verify_installation()
    name = f"_avs_cvqa_{COMMIT}"
    package = ModuleType(name)
    package.__path__ = [str(SOURCE_ROOT / "fdim")]
    sys.modules[name] = package
    model_module = importlib.import_module(f"{name}.dist.deep_model")
    if sys.path[-1:] == ["./fdim/dist/"]:
        sys.path.pop()
    utils_module = importlib.import_module(f"{name}.utils.utils")
    return model_module, utils_module


def create_fdim_metric(device: str | torch.device | None = None) -> torch.nn.Module:
    model_module, _ = _official_modules()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model_module.Model()
    checkpoint = torch.load(
        SOURCE_ROOT / "fdim/dist/checkpoints/dist_5.0.0.ckpt",
        map_location=device, weights_only=True,
    )
    model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=True)
    return model.to(device).eval()


def calculate_fdim_deep(
    original_rgb: torch.Tensor,
    recon_rgb: torch.Tensor,
    metric: torch.nn.Module,
) -> float:
    with torch.no_grad():
        score = metric(original_rgb, recon_rgb).item()
    if not math.isfinite(score):
        raise RuntimeError("FDIM deep model returned a non-finite value.")
    return score


def calculate_fdim_vmaf(
    original_path: str | Path,
    recon_path: str | Path,
    height: int,
    width: int,
    frames: int,
    bit_depth: int,
) -> float:
    _official_modules()
    if bit_depth not in SUPPORTED_BIT_DEPTHS:
        raise ValueError("FDIM's official VMAF supports bit depths 8, 10, 12 and 16.")
    if frames <= 0 or min(height, width) <= 0 or height % 2 or width % 2:
        raise ValueError("FDIM requires positive frames and positive even dimensions.")
    model_version = "vmaf_4k_v0.6.1" if width * height >= 3840 * 2160 else "vmaf_v0.6.1"
    with tempfile.TemporaryDirectory(prefix="calculation-tools-fdim-") as directory:
        output = Path(directory) / "vmaf.xml"
        command = [
            str(SOURCE_ROOT / "fdim/vmaf/vmaf"),
            "--reference", str(Path(original_path).resolve()),
            "--distorted", str(Path(recon_path).resolve()),
            "--width", str(width), "--height", str(height),
            "--model", f"version={model_version}",
            "--pixel_format", "420", "--bitdepth", str(bit_depth),
            "--subsample", "1", "--frame_cnt", str(frames),
            "--threads", "32", "--output", str(output),
        ]
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode != 0:
            raise RuntimeError(f"FDIM VMAF failed: {(process.stderr or process.stdout)[-2000:]}")
        try:
            root = ET.parse(output).getroot()
            indices = [int(frame.attrib["frameNum"]) for frame in root.findall("frames/frame")]
            if indices != list(range(frames)):
                raise RuntimeError(f"FDIM VMAF returned {len(indices)} frames; expected {frames}.")
            pooled = root.find("pooled_metrics/metric[@name='vmaf']")
            if pooled is None:
                raise RuntimeError("FDIM VMAF output has no pooled VMAF score.")
            score = float(pooled.attrib["mean"])
        except (ET.ParseError, KeyError, ValueError, OSError) as error:
            raise RuntimeError(f"Invalid FDIM VMAF output: {error}") from error
    if not math.isfinite(score):
        raise RuntimeError("FDIM VMAF returned a non-finite value.")
    return score


def calculate_fdim(deep_mean: float, vmaf_mean: float) -> float:
    if not math.isfinite(deep_mean) or not math.isfinite(vmaf_mean):
        raise ValueError("FDIM requires finite sequence means.")
    _, utils_module = _official_modules()
    deep = utils_module.map_score(deep_mean, score_flag="reg")
    vmaf = utils_module.map_score(vmaf_mean, score_flag="VMAF")
    return float((deep + vmaf) / 2)
