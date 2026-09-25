# Release Validation

Validation date: **2026-09-25**.

## Environment and Scope

A new Python 3.10.9 virtual environment was created with `python -m venv .venv`. Its configuration explicitly sets `include-system-site-packages = false`. All Python dependencies were installed from `requirements.txt`; Torch and PyIQA imports were confirmed to resolve inside that environment.

The host already provided Linux x86_64, glibc 2.27, GCC 7.5.0, FFmpeg 4.3 and an NVIDIA GTX 1080 Ti with driver 535.146.02. The installed PyTorch runtime was 2.6.0+cu124. This check used a clean Python environment on the same server, not a newly installed operating system.

The evaluation entry point, all five metric modules and both backend installers were byte-for-byte identical to the source files supplied for packaging. `prepare_assets.py` is the new resource preparation helper.

## Executed Workflow

After extracting the candidate source ZIP and creating the environment, the following commands completed successfully:

```bash
python -m pip install -r requirements.txt
python -m pip check
export TORCH_HOME="$PWD/.cache/torch"
python prepare_assets.py --offline --cache_dir downloads
python setup_vmaf.py --archive downloads/vmaf-v3.2.1.tar.gz
python setup_fdim.py --archive downloads/avs-cvqa-e01944a363089a636ed3ce092b1d20cf475d44f2.tar.gz
python eval.py --help
python eval.py \
  --original_path samples/src01_hrc00_576x324.yuv \
  --recon_path samples/src01_hrc01_576x324.yuv \
  --HxW 324x576 \
  --frames 48 \
  --bit_depth 8
```

For the offline step, `downloads/` was populated with the eight official files listed in `assets.json`, obtained in earlier downloads. All sizes and SHA256 values were rechecked. The weights were installed into a new project-local `TORCH_HOME`, and both native backends were installed under the newly extracted package. The previous project's virtual environment, installed backends and model cache were not used for evaluation.

An online asset-preparation attempt was stopped after slow/unreliable GitHub access; Hugging Face was also inaccessible from this host. The complete validation therefore follows the README's **offline asset preparation** route. Uninterrupted online downloading of all eight assets was not validated in this run. Successful separate URL probes are not treated as a complete online installation test.

## Results

- `pip check`: no broken requirements.
- VMAF: **17 upstream tests passed**. The four tests requiring embedded models are excluded by the existing installer and listed in `run_report.json`.
- FDIM: pinned source/checkpoint/executable hashes verified; bundled VMAF version 3.0.0 confirmed.
- Full 48-frame sample: **all eight reported values matched the reference output to six decimal places** and passed the tolerances in `examples/expected_results.json`.

| Metric | Observed output |
| --- | ---: |
| PSNR_Y | 30.755064 |
| PSNR_U | 38.449441 |
| PSNR_V | 40.991910 |
| PSNR_YUV | 32.996467 |
| LPIPS | 0.143703 |
| DISTS | 0.130336 |
| VMAF | 76.667831 |
| FDIM | 4.004700 |

The output includes torchvision deprecation warnings from the pinned upstream model code. They did not prevent loading or evaluation.

## Included Records

- `run_report.json`: environment versions, command results, asset hashes, source hashes and numerical comparisons.
- `sample_output.txt`: captured sample output; the temporary package path is replaced with `<package>`.
- `asset_preparation.txt`: captured offline asset preparation output with the same path substitution.
- `pip-freeze.txt`: installed dependency versions for this validation environment.

The final ZIP adds documentation, validation records and a file manifest to the tested source files. These additions do not change the executable pipeline. Archive extraction, CRC integrity and SHA256 manifest checks are performed before delivery.
