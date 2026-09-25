import ctypes as C
import hashlib
import json
import math
from pathlib import Path

import numpy as np


VMAF_RELEASE = "3.2.1"
VMAF_MODEL = "vmaf_v0.6.1"
MODEL_SHA256 = "5950d61fa1f861bd45d8149d80539ed9f3376cfc2495b8f0fa8e9f57cb131ee3"
SOURCE_SHA256 = "1b4ded2ab0d336fbee99c7ecbd20ffaa1a51fd91b657ab1045db360714571294"


class _Configuration(C.Structure):
    _fields_ = [
        ("log_level", C.c_int), ("n_threads", C.c_uint),
        ("n_subsample", C.c_uint), ("cpumask", C.c_uint64),
        ("gpumask", C.c_uint64),
    ]


class _ModelConfig(C.Structure):
    _fields_ = [("name", C.c_char_p), ("flags", C.c_uint64)]


class _Picture(C.Structure):
    _fields_ = [
        ("pix_fmt", C.c_int), ("bpc", C.c_uint),
        ("w", C.c_uint * 3), ("h", C.c_uint * 3),
        ("stride", C.c_ssize_t * 3), ("data", C.c_void_p * 3),
        ("ref", C.c_void_p), ("priv", C.c_void_p),
    ]


def _check(status: int, operation: str) -> None:
    if status:
        raise RuntimeError(f"libvmaf {operation} failed (error {status}).")


def _load_library() -> tuple[C.CDLL, Path]:
    prefix = Path(__file__).resolve().parents[1] / "third_party" / f"vmaf-{VMAF_RELEASE}"
    model_path = prefix / "model" / f"{VMAF_MODEL}.json"
    library_path = prefix / "lib" / "libvmaf.so"
    manifest_path = prefix / "build_info.json"
    if not all(path.is_file() for path in (model_path, library_path, manifest_path)):
        raise RuntimeError("VMAF is not installed. Run `python setup_vmaf.py` in the code directory.")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("release") != VMAF_RELEASE or manifest.get("source_sha256") != SOURCE_SHA256:
        raise RuntimeError("Unexpected libvmaf build; rebuild with setup_vmaf.py.")
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError("VMAF model checksum mismatch; rebuild with setup_vmaf.py.")

    lib = C.CDLL(str(library_path))
    signatures = {
        "vmaf_version": ([], C.c_char_p),
        "vmaf_init": ([C.POINTER(C.c_void_p), _Configuration], C.c_int),
        "vmaf_close": ([C.c_void_p], C.c_int),
        "vmaf_model_load_from_path": (
            [C.POINTER(C.c_void_p), C.POINTER(_ModelConfig), C.c_char_p], C.c_int),
        "vmaf_model_destroy": ([C.c_void_p], None),
        "vmaf_use_features_from_model": ([C.c_void_p, C.c_void_p], C.c_int),
        "vmaf_picture_alloc": (
            [C.POINTER(_Picture), C.c_int, C.c_uint, C.c_uint, C.c_uint], C.c_int),
        "vmaf_picture_unref": ([C.POINTER(_Picture)], C.c_int),
        "vmaf_read_pictures": (
            [C.c_void_p, C.POINTER(_Picture), C.POINTER(_Picture), C.c_uint], C.c_int),
        "vmaf_score_at_index": (
            [C.c_void_p, C.c_void_p, C.POINTER(C.c_double), C.c_uint], C.c_int),
    }
    for name, (argtypes, restype) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = argtypes, restype
    if lib.vmaf_version().decode() != "3.2.0":
        raise RuntimeError("Unexpected libvmaf ABI; rebuild the pinned release with setup_vmaf.py.")
    return lib, model_path


class VmafMetric:

    def __init__(self, height: int, width: int, bit_depth: int):
        if height < 32 or width < 32 or height % 2 or width % 2:
            raise ValueError("VMAF requires even YUV420 dimensions of at least 32.")
        if bit_depth not in range(8, 17):
            raise ValueError("VMAF bit depth must be between 8 and 16.")
        self.height, self.width, self.bit_depth = height, width, bit_depth
        self._lib, model_path = _load_library()
        self._context, self._model = C.c_void_p(), C.c_void_p()
        self._frame_count = 0
        self._flushed = False
        self._scores: list[float] | None = None

        context = C.c_void_p()
        config = _Configuration(log_level=1, n_threads=0, n_subsample=1, cpumask=0, gpumask=1)
        _check(self._lib.vmaf_init(C.byref(context), config), "init")
        self._context = context
        try:
            model = C.c_void_p()
            model_config = _ModelConfig(name=b"vmaf", flags=0)
            _check(self._lib.vmaf_model_load_from_path(
                C.byref(model), C.byref(model_config), str(model_path).encode()), "load model")
            self._model = model
            _check(self._lib.vmaf_use_features_from_model(self._context, self._model), "load features")
        except BaseException:
            self.close()
            raise

    def _copy_picture(self, frame: tuple[np.ndarray, np.ndarray, np.ndarray], picture: _Picture) -> None:
        shapes = ((self.height, self.width), (self.height // 2, self.width // 2))
        dtype = np.dtype("u1" if self.bit_depth == 8 else "u2")
        if len(frame) != 3:
            raise ValueError("Expected three Y, U, V planes.")
        for index, plane in enumerate(frame):
            if plane.shape != shapes[0 if index == 0 else 1]:
                raise ValueError("VMAF plane shape does not match the configured resolution.")
            if plane.dtype.kind != "u" or plane.dtype.itemsize != dtype.itemsize:
                raise ValueError("VMAF planes must contain raw uint8/uint16 samples.")
        _check(self._lib.vmaf_picture_alloc(
            C.byref(picture), 1, self.bit_depth, self.width, self.height), "allocate picture")
        for index, plane in enumerate(frame):
            buffer_size = picture.stride[index] * picture.h[index]
            buffer = (C.c_ubyte * buffer_size).from_address(picture.data[index])
            destination = np.ndarray(
                plane.shape, dtype=dtype, buffer=buffer,
                strides=(picture.stride[index], dtype.itemsize),
            )
            np.copyto(destination, plane, casting="equiv")

    def add_frame(
        self,
        original_yuv: tuple[np.ndarray, np.ndarray, np.ndarray],
        recon_yuv: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        if not self._context or self._flushed:
            raise RuntimeError("Cannot add frames to a closed or flushed VMAF context.")
        original, recon = _Picture(), _Picture()
        try:
            self._copy_picture(original_yuv, original)
            self._copy_picture(recon_yuv, recon)
            _check(self._lib.vmaf_read_pictures(
                self._context, C.byref(original), C.byref(recon), self._frame_count),
                f"read frame {self._frame_count}")
            self._frame_count += 1
        finally:
            for picture in (original, recon):
                if picture.ref:
                    self._lib.vmaf_picture_unref(C.byref(picture))

    def finish(self) -> list[float]:
        if self._scores is not None:
            return list(self._scores)
        if not self._context or not self._frame_count:
            raise RuntimeError("VMAF needs an open context with at least one frame.")
        if not self._flushed:
            _check(self._lib.vmaf_read_pictures(self._context, None, None, 0), "flush")
            self._flushed = True
        scores = []
        for index in range(self._frame_count):
            value = C.c_double()
            _check(self._lib.vmaf_score_at_index(
                self._context, self._model, C.byref(value), index), f"score frame {index}")
            if not math.isfinite(value.value):
                raise RuntimeError(f"Non-finite VMAF score at frame {index}.")
            scores.append(value.value)
        self._scores = scores
        return list(scores)

    def close(self) -> None:
        if self._context:
            self._lib.vmaf_close(self._context)
            self._context = C.c_void_p()
        if self._model:
            self._lib.vmaf_model_destroy(self._model)
            self._model = C.c_void_p()

    def __enter__(self) -> "VmafMetric":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
