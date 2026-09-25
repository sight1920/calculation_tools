import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request


RELEASE = "3.2.1"
SOURCE_URL = "https://api.github.com/repos/Netflix/vmaf/tarball/v3.2.1"
SOURCE_SHA256 = "1b4ded2ab0d336fbee99c7ecbd20ffaa1a51fd91b657ab1045db360714571294"
SOURCE_DIRECTORY = "Netflix-vmaf-f85a853"


def main() -> None:
    parser = argparse.ArgumentParser(description='Build the pinned official libvmaf release for this Linux evaluation tool.\n\nRequires a C/C++ compiler and `pip install meson==1.9.0 ninja==1.13.0`.\nRun once before eval.py. --archive accepts an already downloaded source tarball.\n')
    parser.add_argument("--archive", type=Path, help="Use a local official source tarball.")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("This build script targets Linux; use a Linux host or WSL.")

    code_dir = Path(__file__).resolve().parent
    prefix = code_dir / "third_party" / f"vmaf-{RELEASE}"
    project_id = hashlib.sha256(str(code_dir).encode()).hexdigest()[:12]
    work_dir = Path(tempfile.gettempdir()) / f"calculation-tools-vmaf-{RELEASE}-{project_id}"
    work_dir.mkdir(parents=True, exist_ok=True)
    archive = args.archive or work_dir / f"vmaf-{RELEASE}.tar.gz"
    if not archive.is_file():
        if args.archive:
            parser.error(f"Source archive does not exist: {archive}")
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "calculation-tools"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("Official source archive SHA256 mismatch; refusing to build.")

    with tarfile.open(archive, "r:gz") as source_archive:
        source_archive.extractall(work_dir)
    source = work_dir / SOURCE_DIRECTORY
    build = work_dir / "build"
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    meson = [sys.executable, "-m", "mesonbuild.mesonmain"]
    options = [
        "--buildtype=release", f"--prefix={prefix}", "--libdir=lib",
        "-Denable_asm=false", "-Denable_avx512=false", "-Dbuilt_in_models=false",
        "-Denable_docs=false", "-Denable_tests=true", "-Denable_float=false",
        "-Denable_cuda=false",
    ]
    setup = meson + ["setup"]
    if (build / "meson-private" / "coredata.dat").is_file():
        setup.append("--reconfigure")
    subprocess.run(setup + [str(build), str(source / "libvmaf")] + options, env=env, check=True)
    subprocess.run(meson + ["compile", "-C", str(build), "-j", "4"], env=env, check=True)
    embedded_model_tests = {"test_pic_preallocation", "test_model", "test_predict", "test_feature_collector"}
    test_info = json.loads(subprocess.check_output(meson + ["introspect", str(build), "--tests"], env=env))
    tests = [test["name"] for test in test_info if test["name"] not in embedded_model_tests]
    subprocess.run(meson + ["test", "-C", str(build), "--no-rebuild", "--print-errorlogs"] + tests,
                   env=env, check=True)
    subprocess.run(meson + ["install", "-C", str(build), "--no-rebuild"], env=env, check=True)

    model_dir = prefix / "model"
    model_dir.mkdir(exist_ok=True)
    model = model_dir / "vmaf_v0.6.1.json"
    shutil.copyfile(source / "model" / model.name, model)
    shutil.copyfile(source / "LICENSE", prefix / "LICENSE")
    lib = ctypes.CDLL(str(prefix / "lib" / "libvmaf.so"))
    lib.vmaf_version.argtypes = []
    lib.vmaf_version.restype = ctypes.c_char_p
    manifest = {
        "release": RELEASE, "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
        "source_directory": SOURCE_DIRECTORY, "runtime_version": lib.vmaf_version().decode(),
        "model": model.name, "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "meson_options": options,
        "upstream_tests_run": tests,
        "upstream_tests_requiring_embedded_models": sorted(embedded_model_tests),
    }
    (prefix / "build_info.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Installed official VMAF release {RELEASE} with {model.name} to {prefix}")
    print(f"Library reports {manifest['runtime_version']} (upstream release metadata).")


if __name__ == "__main__":
    main()
