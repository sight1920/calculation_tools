# YUV420 Objective Quality Metrics

## Introduction

This package evaluates an original/reconstructed pair of planar YUV420 sequences using the AVS4 CfE Annex C metric conventions implemented in this project.

One command reports eight values: **PSNR_Y, PSNR_U, PSNR_V, PSNR_YUV, LPIPS, DISTS, VMAF and FDIM**. The interface takes two file paths, resolution, frame count and bit depth. Frames are read from the beginning of both files.

| Metric | Backend | Better quality |
| --- | --- | --- |
| Y/U/V/YUV PSNR | NumPy | Higher |
| LPIPS | PyIQA 0.1.15.post2, AlexNet LPIPS v0.1 | Lower |
| DISTS | PyIQA 0.1.15.post2, VGG16 DISTS | Lower |
| VMAF | Official libvmaf v3.2.1, `vmaf_v0.6.1` | Higher |
| FDIM | Official avs-cvqa commit `e01944a363089a636ed3ce092b1d20cf475d44f2` | Higher |

## Installation

### Prerequisites

- **Linux x86_64**. The pinned FDIM executable requires this platform. Windows and macOS are not supported by the complete metric pipeline.
- **Python 3.10**, including the `venv` module. The release is tested with Python 3.10.9.
- A C/C++ compiler and `pkg-config`, used to build libvmaf.
- **FFmpeg on PATH**. A build with the libvmaf filter is not required. The recorded reference environment uses FFmpeg 4.3.
- An NVIDIA GPU is optional. PyIQA and FDIM use CUDA when available and otherwise use the CPU.
- Internet access for Python dependencies and initial asset preparation, or a previously downloaded asset cache. Allow roughly 15 GB of free disk space for the environment, downloads, model caches and build files.

For example, on Ubuntu 22.04, the system packages can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y python3.10 python3.10-venv build-essential pkg-config ffmpeg unzip
```

If these tools are already installed, use the existing installation. Check `python3.10 --version`, `ffmpeg -version` and `cc --version`. FFmpeg versions can affect FDIM RGB conversion rounding; compare results in the same environment when exact reproduction matters.

### 1. Extract the package and create an environment

```bash
cd calculation_tools
python3.10 -m venv .venv
source .venv/bin/activate
```

Run the remaining commands from this `calculation_tools` directory, with the environment activated. This environment does not inherit system Python packages.

To verify the extracted source files, run `sha256sum -c MANIFEST.sha256` from the package directory.

### 2. Install Python dependencies

```bash
python -m pip install -r requirements.txt
python -m pip check
```

The requirements pin the principal libraries, including Torch 2.6.0 and torchvision 0.21.0. PyTorch's default Linux wheels also install their CUDA runtime dependencies; CPU evaluation remains available. The GPU driver, if used, must support the installed PyTorch runtime.

### 3. Prepare model weights, source archives and the example

Use a project-local model cache:

```bash
export TORCH_HOME="$PWD/.cache/torch"
python prepare_assets.py
```

The preparation script verifies the size and SHA256 of every file against `assets.json`. It downloads:

- The official VMAF v3.2.1 source archive.
- The exact FDIM source archive, containing its checkpoint and bundled VMAF executable.
- AlexNet, VGG16, LPIPS and DISTS model weights.
- The official Netflix reference/distorted YUV sample pair.

The source files and reusable downloads are stored in `downloads/`; weights are copied into the cache selected by `TORCH_HOME`; example YUV files are copied into `samples/`. Existing verified files are reused. No weights or test media are embedded in this source ZIP.

Keep the same `TORCH_HOME` when evaluating. In a new shell, activate `.venv` and set it again before running the tool. Otherwise, PyTorch may look in a different cache and download weights again.

If a download endpoint is inaccessible, use the **Offline asset preparation** instructions below. The LPIPS/DISTS downloads have official GitHub and Hugging Face alternatives.

### 4. Install the native metric backends

```bash
python setup_vmaf.py --archive downloads/vmaf-v3.2.1.tar.gz
python setup_fdim.py --archive downloads/avs-cvqa-e01944a363089a636ed3ce092b1d20cf475d44f2.tar.gz
```

VMAF is compiled and installed under `third_party/vmaf-3.2.1`. Its setup runs the applicable upstream tests. Build files are kept under the system temporary directory; source archive checksums are verified before extraction.

FDIM is installed under `third_party/avs-cvqa-e01944a363089a636ed3ce092b1d20cf475d44f2`. Its installer verifies the source, checkpoint and executable and sets the executable permission. Use this installer with this package's dependencies; the upstream FDIM `install.sh` targets a different Torch environment.

## Instructions for Evaluation

### 1. Prepare the input sequences

Inputs must be **raw planar YUV420**, with the same resolution, bit depth and frame alignment. Each frame stores the complete Y plane, followed by U, then V, without a header or row padding.

| Argument | Meaning | Example |
| --- | --- | --- |
| `--original_path` | Original/reference YUV file | `samples/src01_hrc00_576x324.yuv` |
| `--recon_path` | Reconstructed/distorted YUV file | `samples/src01_hrc01_576x324.yuv` |
| `--HxW` | **Height x width** | `324x576` |
| `--frames` | Number of frames from frame 0 | `48` |
| `--bit_depth` | Sample bit depth: 8, 10, 12 or 16 | `8` |

Both dimensions must be even and at least 32. Both inputs must contain the requested number of complete frames. If more frames are present, only the requested prefix is evaluated.

For 8-bit input, samples are `uint8`. For 10/12/16-bit input, samples are little-endian `uint16`, with values in the low bits. NV12/P010, packed higher-bit-depth formats, images and compressed MP4 files are not accepted by this entry point.

### 2. Run the example

```bash
python eval.py \
  --original_path samples/src01_hrc00_576x324.yuv \
  --recon_path samples/src01_hrc01_576x324.yuv \
  --HxW 324x576 \
  --frames 48 \
  --bit_depth 8
```

The example is 576 pixels wide, 324 pixels high, 48 frames, 8-bit YUV420. The example sequences are `src01_hrc00_576x324.yuv` and `src01_hrc01_576x324.yuv`, taken from the reference examples at https://github.com/Netflix/vmaf/blob/v3.2.1/libvmaf/tools/README.md#example. Its reference output in the recorded environment is:

```text
Resolution (HxW): 324x576
Frames: 48
Bit depth: 8
PSNR_Y: 30.755064 dB
PSNR_U: 38.449441 dB
PSNR_V: 40.991910 dB
PSNR_YUV: 32.996467 dB
LPIPS: 0.143703
DISTS: 0.130336
VMAF: 76.667831
FDIM: 4.004700
```

Model-loading messages may appear before these results. `examples/expected_results.json` contains full-precision reference values and the tolerances used for the recorded environment. Neural-network floating-point results and FFmpeg conversion can vary across runtimes or devices; the file is not a guarantee of bitwise agreement on every machine.

### 3. Evaluate your own videos

Replace the paths and sequence properties, for example:

```bash
python eval.py \
  --original_path /path/to/original.yuv \
  --recon_path /path/to/reconstructed.yuv \
  --HxW 1080x1920 \
  --frames 100 \
  --bit_depth 10
```

Use `python eval.py --help` to display the interface. To select a GPU, set `CUDA_VISIBLE_DEVICES=0` before the command. To force CPU evaluation, set `CUDA_VISIBLE_DEVICES=""`. CPU evaluation is slower.

## Offline Asset Preparation

The asset cache is portable between machines. On a connected machine, run:

```bash
python prepare_assets.py --cache_dir /path/to/asset-cache
```

Alternatively, download the eight files listed in `assets.json` from their official URLs and put them directly in that directory using the listed filenames. Do not rename or recompress the archives.

Copy that directory to the target machine. After installing the Python dependencies, run:

```bash
export TORCH_HOME="$PWD/.cache/torch"
python prepare_assets.py --offline --cache_dir /path/to/asset-cache
python setup_vmaf.py --archive /path/to/asset-cache/vmaf-v3.2.1.tar.gz
python setup_fdim.py --archive /path/to/asset-cache/avs-cvqa-e01944a363089a636ed3ce092b1d20cf475d44f2.tar.gz
```

Then run the same `python eval.py` example. `--offline` disables network access in asset preparation and fails if a required file is missing or its checksum differs. It does not install Python packages or operating-system dependencies; those must already be available on an air-gapped target.

To prepare only selected resources, use `--group sources`, `--group weights` or `--group samples`.

## Calculation Conventions

- **PSNR:** compute each plane's PSNR for each frame using a peak of `2**bit_depth - 1`. The per-frame YUV value is `(6 * PSNR_Y + PSNR_U + PSNR_V) / 8`, in dB. Average each reported PSNR over frames. Identical planes produce infinity.
- **LPIPS/DISTS:** use DCVC-style full-range normalization, nearest-neighbor chroma upsampling and the BT.709 matrix, then clip RGB to `[0, 1]`. Run the PyIQA models at native resolution in float32 and average the frame scores. Do not manually normalize RGB to `[-1, 1]` before PyIQA.
- **VMAF:** pass raw YUV frames to the official libvmaf v3.2.1 C API with `vmaf_v0.6.1`, preserve temporal state and average all requested frame scores.
- **FDIM:** preserve the pinned upstream implementation's FFmpeg RGB24 conversion and per-frame deep model. Average raw deep scores before applying the nonlinear mapping. Its separate bundled VMAF is **3.0.0**, with upstream's automatic 4K model selection. Map the sequence-level VMAF mean and average the two mapped components. FDIM's internal VMAF must not be replaced with the separately reported VMAF backend.

## Code Organization

```text
calculation_tools/
  eval.py
  metrics/
    psnr.py
    lpips.py
    dists.py
    vmaf.py
    fdim.py
  requirements.txt
  prepare_assets.py
  assets.json
  setup_vmaf.py
  setup_fdim.py
  examples/expected_results.json
  validation/
  THIRD_PARTY.md
  README.md
```

`downloads/`, `.cache/`, `samples/` and `third_party/` are created during setup. The evaluator and metric functions retain the original calculation behavior. This distribution contains the YUV entry point present in the source directory; it does not provide a separate image CLI.

## References and Third-Party Materials

- [PyIQA](https://github.com/chaofengc/IQA-PyTorch/tree/v0.1.15)
- [Official VMAF v3.2.1](https://github.com/Netflix/vmaf/tree/v3.2.1)
- [Official FDIM source and usage](https://gitlab.com/jiaqi.zhangzju/avs-cvqa/-/tree/e01944a363089a636ed3ce092b1d20cf475d44f2)
- [Netflix example resources](https://github.com/Netflix/vmaf_resource/tree/c0ab6adbd7e41bb354f14686ed08500530622bc3/python/test/resource/yuv)
- [DCVC color conversion reference](https://github.com/microsoft/DCVC/blob/54e88645e1b4edac0694228ab2f44ed8818e345b/src/utils/transforms.py)
