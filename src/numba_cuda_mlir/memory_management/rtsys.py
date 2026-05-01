# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NRT Runtime System for numba_cuda_mlir.

This module provides the runtime system singleton (`rtsys`) that manages
device-side NRT memory allocator state. It compiles memsys.cu to initialize
the NRT_MemSys structure on the device before kernels that use NRT can run.
"""

import ctypes
import hashlib
import os
from collections import namedtuple
from functools import wraps
from pathlib import Path

import numpy as np

from numba_cuda_mlir import numba_cuda as cuda
from numba_cuda_mlir.numba_cuda.cudadrv.driver import _Linker, _have_nvjitlink
from numba_cuda_mlir.numba_cuda.api import get_current_device
from cuda.bindings import driver as drv

from numba_cuda_mlir.memory_management.config import is_nrt_stats_enabled

_nrt_mstats = namedtuple("nrt_mstats", ["alloc", "free", "mi_alloc", "mi_free"])


def _alloc_init_guard(method):
    """Ensure NRT memory allocation and initialization before running the method."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self.ensure_allocated()
        self.ensure_initialized()
        return method(self, *args, **kwargs)

    return wrapper


class _Runtime:
    """Singleton class for numba_cuda_mlir NRT runtime."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(_Runtime, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        """Initialize memsys module and variable."""
        if not hasattr(self, "_initialized_singleton"):
            self._reset()
            self._initialized_singleton = True

    def _reset(self):
        """Reset to the uninitialized state."""
        self._memsys_library = None
        self._memsys = None
        self._initialized = False

    def close(self):
        """Close and reset."""
        self._reset()

    @staticmethod
    def _memsys_cache_dir():
        return Path(
            os.environ.get(
                "NUMBA_CUDA_MLIR_CACHE_DIR",
                Path.home() / ".cache" / "numba_cuda_mlir",
            )
        )

    @staticmethod
    def _memsys_source_hash():
        """Hash memsys.cu + memsys.cuh to detect source changes."""
        src_dir = os.path.dirname(os.path.abspath(__file__))
        h = hashlib.sha256()
        for name in ("memsys.cu", "memsys.cuh"):
            with open(os.path.join(src_dir, name), "rb") as f:
                h.update(f.read())
        return h.hexdigest()[:16]

    def _compile_memsys_module(self):
        """Compile memsys.cu (with disk caching) and load as a CUlibrary."""
        cc = get_current_device().compute_capability
        src_hash = self._memsys_source_hash()
        cache_dir = self._memsys_cache_dir()
        cache_file = cache_dir / f"memsys_sm{cc[0]}{cc[1]}_{src_hash}.cubin"

        if cache_file.exists():
            cubin_bytes = cache_file.read_bytes()
        else:
            memsys_mod = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "memsys.cu"
            )
            linker = _Linker(max_registers=0, cc=cc, lto=_have_nvjitlink())
            linker.add_cu_file(memsys_mod)
            cubin = linker.complete()
            cubin_bytes = bytes(cubin.code)

            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file.write_bytes(cubin_bytes)

        err, library = drv.cuLibraryLoadData(cubin_bytes, [], [], 0, [], [], 0)
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLibraryLoadData failed: {err}")
        self._memsys_library = library

    def _launch_kernel(self, name, params_ctypes=()):
        """Launch a kernel from the memsys CUlibrary with grid=1, block=1.

        *params_ctypes* is a sequence of ctypes values to pass as kernel
        arguments.  Pass an empty tuple for kernels that take no args.
        """
        lib = self._memsys_library
        err, kernel = drv.cuLibraryGetKernel(lib, name.encode())
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLibraryGetKernel({name}) failed: {err}")

        err, func = drv.cuKernelGetFunction(kernel)
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuKernelGetFunction({name}) failed: {err}")

        n = len(params_ctypes)
        if n > 0:
            ptrs = [ctypes.pointer(p) for p in params_ctypes]
            params = (ctypes.c_void_p * n)(
                *(ctypes.cast(p, ctypes.c_void_p).value for p in ptrs)
            )
        else:
            params = None

        (err,) = drv.cuLaunchKernel(
            func, 1, 1, 1, 1, 1, 1, 0, drv.CUstream(0), params, 0
        )
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLaunchKernel({name}) failed: {err}")

        (err,) = drv.cuStreamSynchronize(drv.CUstream(0))
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuStreamSynchronize failed: {err}")

    def _get_library_global(self, name, nbytes):
        """Read *nbytes* from a device global in the memsys CUlibrary."""
        err, dptr, size = drv.cuLibraryGetGlobal(self._memsys_library, name.encode())
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLibraryGetGlobal({name}) failed: {err}")
        buf = (ctypes.c_char * nbytes)()
        (err,) = drv.cuMemcpyDtoH(buf, dptr, nbytes)
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuMemcpyDtoH for {name} failed: {err}")
        return buf

    def ensure_allocated(self):
        """If memsys is not allocated, allocate it; otherwise, perform a no-op."""
        if self._memsys is not None:
            return
        self.allocate()

    def allocate(self):
        """Allocate memsys on global memory."""
        if self._memsys_library is None:
            self._compile_memsys_module()

        buf = self._get_library_global("memsys_size", ctypes.sizeof(ctypes.c_uint64))
        memsys_size = ctypes.c_uint64.from_buffer_copy(buf).value

        self._memsys = cuda.device_array((memsys_size,), dtype="i1")
        self._set_memsys_ptr()

    def ensure_initialized(self):
        """If memsys is not initialized, initialize memsys."""
        if self._initialized:
            return
        self.initialize()

    def initialize(self):
        """Launch memsys initialization kernel."""
        self.ensure_allocated()

        self._launch_kernel("NRT_MemSys_init")
        self._initialized = True

        if is_nrt_stats_enabled():
            self.memsys_enable_stats()

    @_alloc_init_guard
    def memsys_enable_stats(self):
        """Enable memsys statistics."""
        self._launch_kernel("NRT_MemSys_enable_stats")

    @_alloc_init_guard
    def memsys_disable_stats(self):
        """Disable memsys statistics."""
        self._launch_kernel("NRT_MemSys_disable_stats")

    @_alloc_init_guard
    def memsys_stats_enabled(self):
        """Return a boolean indicating whether memsys stats are enabled."""
        enabled_ar = cuda.managed_array(1, np.uint8)
        enabled_ptr = enabled_ar.device_ctypes_pointer

        self._launch_kernel(
            "NRT_MemSys_stats_enabled",
            (enabled_ptr,),
        )

        return bool(enabled_ar[0])

    @_alloc_init_guard
    def _copy_memsys_to_host(self):
        """Copy all statistics of memsys to the host."""
        dt = np.dtype(
            [
                ("alloc", np.uint64),
                ("free", np.uint64),
                ("mi_alloc", np.uint64),
                ("mi_free", np.uint64),
            ]
        )

        stats_for_read = cuda.managed_array(1, dt)
        stats_ptr = stats_for_read.device_ctypes_pointer

        self._launch_kernel("NRT_MemSys_read", (stats_ptr,))

        return stats_for_read[0]

    @_alloc_init_guard
    def get_allocation_stats(self):
        """Get the allocation statistics."""
        enabled = self.memsys_stats_enabled()
        if not enabled:
            raise RuntimeError("NRT stats are disabled.")
        memsys = self._copy_memsys_to_host()
        return _nrt_mstats(
            alloc=memsys["alloc"],
            free=memsys["free"],
            mi_alloc=memsys["mi_alloc"],
            mi_free=memsys["mi_free"],
        )

    @_alloc_init_guard
    def _get_single_stat(self, stat):
        """Get a single stat from the memsys."""
        got = cuda.managed_array(1, np.uint64)
        got_ptr = got.device_ctypes_pointer

        self._launch_kernel(f"NRT_MemSys_read_{stat}", (got_ptr,))

        return got[0]

    @_alloc_init_guard
    def memsys_get_stats_alloc(self):
        """Get the allocation statistic."""
        if not self.memsys_stats_enabled():
            raise RuntimeError("NRT stats are disabled.")
        return self._get_single_stat("alloc")

    @_alloc_init_guard
    def memsys_get_stats_free(self):
        """Get the free statistic."""
        if not self.memsys_stats_enabled():
            raise RuntimeError("NRT stats are disabled.")
        return self._get_single_stat("free")

    @_alloc_init_guard
    def memsys_get_stats_mi_alloc(self):
        """Get the mi_alloc statistic."""
        if not self.memsys_stats_enabled():
            raise RuntimeError("NRT stats are disabled.")
        return self._get_single_stat("mi_alloc")

    @_alloc_init_guard
    def memsys_get_stats_mi_free(self):
        """Get the mi_free statistic."""
        if not self.memsys_stats_enabled():
            raise RuntimeError("NRT stats are disabled.")
        return self._get_single_stat("mi_free")

    def _set_memsys_ptr(self):
        """Set the memsys pointer on the memsys helper library itself."""
        memsys_ptr = ctypes.c_void_p(self._memsys.device_ctypes_pointer.value)
        self._launch_kernel("NRT_MemSys_set", (memsys_ptr,))

    def set_memsys_to_library(self, library):
        """Set the memsys pointer for an external CUlibrary handle.

        Looks up and launches ``NRT_MemSys_set`` from *library* so its
        ``TheMSys`` device global points to the shared NRT_MemSys
        allocation.  If the kernel is not found (i.e. the library was
        compiled without NRT), this is a no-op.
        """
        if self._memsys is None:
            raise RuntimeError(
                "Please allocate NRT Memsys first before setting to library."
            )

        err, kernel = drv.cuLibraryGetKernel(library, b"NRT_MemSys_set")
        if err != drv.CUresult.CUDA_SUCCESS:
            return

        err, func = drv.cuKernelGetFunction(kernel)
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuKernelGetFunction failed: {err}")

        memsys_ptr = self._memsys.device_ctypes_pointer
        param = ctypes.c_void_p(memsys_ptr.value)
        param_ptr = ctypes.pointer(param)
        params = (ctypes.c_void_p * 1)(ctypes.cast(param_ptr, ctypes.c_void_p).value)

        (err,) = drv.cuLaunchKernel(
            func, 1, 1, 1, 1, 1, 1, 0, drv.CUstream(0), params, 0
        )
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuLaunchKernel(NRT_MemSys_set) failed: {err}")

        (err,) = drv.cuStreamSynchronize(drv.CUstream(0))
        if err != drv.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuStreamSynchronize failed: {err}")

    @_alloc_init_guard
    def print_memsys(self):
        """Print the current statistics of memsys, for debugging purposes."""
        self._launch_kernel("NRT_MemSys_print")


# Create the singleton instance
rtsys = _Runtime()
