import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request


ROOT = Path(__file__).resolve().parent


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def verify(path, asset):
    if path.stat().st_size != asset["bytes"] or digest(path) != asset["sha256"]:
        raise RuntimeError(f"Checksum or size mismatch: {path}. Move this file aside and retry.")


def fetch(asset, cache_dir, offline):
    destination = cache_dir / asset["name"]
    if destination.is_file():
        verify(destination, asset)
        print(f"Verified cache: {destination}", flush=True)
        return destination
    if offline:
        raise FileNotFoundError(f"Offline asset missing: {destination}")
    cache_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for source in asset["sources"]:
        temporary = None
        try:
            print(f"Downloading {asset['name']} from {source['url']}", flush=True)
            headers = {"User-Agent": "calculation-tools", **source.get("headers", {})}
            request = urllib.request.Request(source["url"], headers=headers)
            with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=".download-", delete=False) as output:
                temporary = Path(output.name)
                with urllib.request.urlopen(request, timeout=45) as response:
                    if source.get("encoding") == "github-blob":
                        payload = json.load(response)
                        if payload.get("encoding") != "base64" or payload.get("size") != asset["bytes"]:
                            raise RuntimeError("Unexpected GitHub blob metadata")
                        output.write(base64.b64decode(payload["content"]))
                    else:
                        shutil.copyfileobj(response, output, length=1024 * 1024)
            verify(temporary, asset)
            temporary.replace(destination)
            return destination
        except (OSError, ValueError, RuntimeError) as error:
            failures.append(f"{source['url']}: {error}")
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    detail = "\n".join(failures)
    raise RuntimeError(
        f"Could not download {asset['name']}.\n{detail}\n"
        f"Place the official file in {cache_dir} and rerun with --offline."
    )


def install(asset, cached, torch_home):
    if asset["group"] == "sources":
        return
    destination = (torch_home if asset["group"] == "weights" else ROOT) / asset["target"]
    if destination.exists():
        verify(destination, asset)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".install-", delete=False) as output:
            temporary = Path(output.name)
            try:
                with cached.open("rb") as stream:
                    shutil.copyfileobj(stream, output, length=1024 * 1024)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            verify(temporary, asset)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"Ready: {destination}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare checksum-verified sources, weights and example YUV files.")
    parser.add_argument("--cache_dir", type=Path, default=ROOT / "downloads")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--group", nargs="+", choices=("sources", "weights", "samples"), default=["sources", "weights", "samples"])
    args = parser.parse_args()
    default_cache = Path(os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache")))
    torch_home = Path(os.environ.get("TORCH_HOME", str(default_cache / "torch"))).expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve()
    manifest = json.loads((ROOT / "assets.json").read_text(encoding="utf-8"))
    try:
        for asset in manifest["assets"]:
            if asset["group"] in args.group:
                cached = fetch(asset, cache_dir, args.offline)
                install(asset, cached, torch_home)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    print("Requested assets are ready.")


if __name__ == "__main__":
    main()
