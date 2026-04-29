# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NRT (Numba Runtime) LTOIR compilation and linking support.

This module compiles NRT CUDA sources to LTOIR for linking with
cusimt-compiled kernels that use NRT functions.
"""

import hashlib
import os
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from cusimt.numba_cuda.cudadrv.nvrtc import compile as nvrtc_compile

_NRT_DIR = Path(__file__).parent


def _get_cache_dir() -> Path:
    cache_root = os.environ.get("CUSIMT_CACHE_DIR")
    if cache_root:
        return Path(cache_root) / "nrt"
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return base / "cusimt" / "nrt"


def get_include():
    """Return the include path for the NRT headers."""
    return str(_NRT_DIR)


@lru_cache(maxsize=1)
def _get_nrt_source() -> bytes:
    """Get NRT source with all includes inlined."""
    memsys_cuh = (_NRT_DIR / "memsys.cuh").read_text()
    nrt_cuh = (_NRT_DIR / "nrt.cuh").read_text()
    nrt_cu = (_NRT_DIR / "nrt.cu").read_text()

    # Remove include directives and inline the content
    nrt_cu = nrt_cu.replace('#include "memsys.cuh"', memsys_cuh)
    nrt_cu = nrt_cu.replace('#include "nrt.cuh"', nrt_cuh)

    return nrt_cu.encode()


@lru_cache(maxsize=1)
def _get_source_hash() -> str:
    """Compute a hash of NRT source for cache invalidation."""
    return hashlib.sha256(_get_nrt_source()).hexdigest()[:16]


def _get_cache_path(cc: tuple[int, int], ltoir: bool) -> Path:
    """Get the cache file path for the given configuration."""
    from cusimt.tools import get_cuda_runtime_version

    rt_ver = get_cuda_runtime_version()
    suffix = "ltoir" if ltoir else "ptx"
    filename = f"nrt_cuda{rt_ver[0]}{rt_ver[1]}_sm{cc[0]}{cc[1]}_{_get_source_hash()}.{suffix}"
    return _get_cache_dir() / filename


def _load_from_cache(cc: tuple[int, int], ltoir: bool) -> bytes | None:
    """Load cached NRT object code if available."""
    cache_path = _get_cache_path(cc, ltoir)
    try:
        return cache_path.read_bytes()
    except OSError:
        return None


def _save_to_cache(cc: tuple[int, int], ltoir: bool, code: bytes) -> None:
    """Save NRT object code to disk cache."""
    cache_path = _get_cache_path(cc, ltoir)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=cache_path.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(code)
            os.replace(tmp, cache_path)
        except BaseException:
            os.unlink(tmp)
            raise
    except OSError:
        pass


def _cache_enabled() -> bool:
    if os.environ.get("CUSIMT_DISABLE_CACHE", ""):
        shutil.rmtree(_get_cache_dir(), ignore_errors=True)
        return False
    return True


def _compile_nrt(cc: tuple[int, int], ltoir: bool) -> bytes:
    """Compile NRT with disk caching."""
    if _cache_enabled():
        cached = _load_from_cache(cc, ltoir)
        if cached is not None:
            return cached

    nrt_src = _get_nrt_source()
    obj, log = nvrtc_compile(nrt_src, "nrt.cu", cc, ltoir=ltoir)
    code = obj.code

    if _cache_enabled():
        _save_to_cache(cc, ltoir, code)
    return code


@lru_cache(maxsize=8)
def compile_nrt_ltoir(cc: tuple[int, int]) -> bytes:
    """
    Compile NRT sources to LTOIR for the given compute capability.

    Results are cached both in-memory and on disk.
    """
    return _compile_nrt(cc, ltoir=True)


@lru_cache(maxsize=8)
def compile_nrt_object(cc: tuple[int, int]) -> bytes:
    """
    Compile NRT sources to PTX for the given compute capability.

    Used when linking with a non-LTO linker (which cannot mix PTX and LTOIR).
    Results are cached both in-memory and on disk.
    """
    return _compile_nrt(cc, ltoir=False)


# NRT function names that indicate NRT is being used
NRT_FUNCTIONS = frozenset(
    [
        "NRT_Allocate",
        "NRT_MemInfo_alloc",
        "NRT_MemInfo_init",
        "NRT_MemInfo_new",
        "NRT_Free",
        "NRT_dealloc",
        "NRT_MemInfo_destroy",
        "NRT_MemInfo_call_dtor",
        "NRT_MemInfo_data_fast",
        "NRT_MemInfo_alloc_aligned",
        "NRT_Allocate_External",
        "NRT_decref",
        "NRT_incref",
    ]
)


def needs_nrt_linking(asm: str) -> bool:
    """Check if the given assembly/PTX references NRT functions."""
    return any(fn in asm for fn in NRT_FUNCTIONS)
