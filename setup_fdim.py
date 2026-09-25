import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

from metrics.fdim import COMMIT, PINNED_FILES, SOURCE_ROOT, verify_installation


SOURCE_URL = (
    "https://gitlab.com/api/v4/projects/71521651/repository/archive.tar.gz"
    f"?sha={COMMIT}"
)
SOURCE_SHA256 = "7106fc87087a4f49658766dbe2142848334e88b03d488ceb20753a75135c3955"


def main() -> None:
    parser = argparse.ArgumentParser(description='Install the exact Annex C FDIM source and bundled weights (Linux x86_64).')
    parser.add_argument("--archive", type=Path, help="Use a downloaded official source tarball.")
    args = parser.parse_args()
    SOURCE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    if not SOURCE_ROOT.exists():
        with tempfile.TemporaryDirectory(prefix=".fdim-install-", dir=SOURCE_ROOT.parent) as temp:
            staging = Path(temp)
            archive = args.archive or staging / "source.tar.gz"
            if not args.archive:
                print(f"Downloading pinned FDIM source and checkpoint: {SOURCE_URL}")
                request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "calculation-tools"})
                with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as out:
                    shutil.copyfileobj(response, out)
            if hashlib.sha256(archive.read_bytes()).hexdigest() != SOURCE_SHA256:
                raise RuntimeError("FDIM archive SHA256 mismatch; refusing to install.")
            extracted = staging / "source"
            extracted.mkdir()
            with tarfile.open(archive, "r:gz") as source:
                for member in source.getmembers():
                    relative = Path(*Path(member.name).parts[1:])
                    if not relative.parts:
                        continue
                    if relative.is_absolute() or ".." in relative.parts or not (member.isfile() or member.isdir()):
                        raise RuntimeError(f"Unexpected FDIM archive entry: {member.name}")
                    destination = extracted / relative
                    if member.isdir():
                        destination.mkdir(parents=True, exist_ok=True)
                    else:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with source.extractfile(member) as src, destination.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                        destination.chmod(member.mode & 0o777)
            extracted.rename(SOURCE_ROOT)

    verify_installation()
    binary = SOURCE_ROOT / "fdim/vmaf/vmaf"
    binary.chmod(binary.stat().st_mode | 0o111)
    version = subprocess.check_output(
        [str(binary), "--version"], text=True, stderr=subprocess.STDOUT,
    ).strip()
    if version != "3.0.0":
        raise RuntimeError(f"Unexpected bundled FDIM VMAF version: {version}")
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("FDIM is installed; install FFmpeg and add it to PATH before evaluation.")
    ffmpeg_version = subprocess.check_output([ffmpeg, "-version"], text=True).splitlines()[0]
    manifest = {
        "repository": "https://gitlab.com/jiaqi.zhangzju/avs-cvqa.git",
        "commit": COMMIT, "archive_url": SOURCE_URL, "archive_sha256": SOURCE_SHA256,
        "pinned_files": PINNED_FILES, "vmaf_version": version,
        "ffmpeg_at_setup": ffmpeg_version,
    }
    (SOURCE_ROOT / "source_info.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"FDIM ready: {SOURCE_ROOT}")
    print(ffmpeg_version)


if __name__ == "__main__":
    main()
