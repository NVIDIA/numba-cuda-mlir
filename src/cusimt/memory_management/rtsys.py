# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
NRT Runtime System for cusimt.

This module provides the runtime system singleton (`rtsys`) that manages
device-side NRT memory allocator state. It compiles memsys.cu to initialize
the NRT_MemSys structure on the device before kernels that use NRT can run.
"""

import ctypes
import os
from collections import namedtuple
from functools import wraps

import numpy as np

from cusimt import numba_cuda as cuda
from cusimt.numba_cuda.cudadrv.driver import (
    _Linker,
    driver,
    _to_core_stream,
    _have_nvjitlink,
)
from cuda.core import LaunchConfig, launch
from cusimt.numba_cuda.cudadrv import devices
from cusimt.numba_cuda.api import get_current_device

from cusimt.memory_management.config import is_nrt_stats_enabled

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
    """Singleton class for cusimt NRT runtime."""

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
        self._memsys_module = None
        self._memsys = None
        self._initialized = False

    def close(self):
        """Close and reset."""
        self._reset()

    def _compile_memsys_module(self):
        """Compile memsys.cu and create a module from it in the current context."""
        memsys_mod = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "memsys.cu"
        )
        cc = get_current_device().compute_capability

        linker = _Linker(max_registers=0, cc=cc, lto=_have_nvjitlink())
        linker.add_cu_file(memsys_mod)

        cubin = linker.complete()
        ctx = devices.get_context()
        module = ctx.create_module_image(cubin)

        self._memsys_module = module

    def ensure_allocated(self, stream=None):
        """If memsys is not allocated, allocate it; otherwise, perform a no-op."""
        if self._memsys is not None:
            return
        self.allocate(stream)

    def allocate(self, stream=None):
        """Allocate memsys on global memory."""
        if self._memsys_module is None:
            self._compile_memsys_module()

        # Get the size of NRT_MemSys from the device
        memsys_size = ctypes.c_uint64()
        ptr, nbytes = self._memsys_module.get_global_symbol("memsys_size")
        device_memsys_size = ptr.device_ctypes_pointer
        device_memsys_size = device_memsys_size.value
        driver.cuMemcpyDtoH(ctypes.addressof(memsys_size), device_memsys_size, nbytes)

        # Allocate device memory for NRT_MemSys
        self._memsys = cuda.device_array(
            (memsys_size.value,), dtype="i1", stream=stream
        )
        self.set_memsys_to_module(self._memsys_module, stream=stream)

    def _single_thread_launch(self, module, stream, name, params=()):
        """Launch the specified kernel with only 1 thread."""
        if stream is None:
            stream = cuda.default_stream()

        func = module.get_function(name)
        config = LaunchConfig(
            grid=(1, 1, 1),
            block=(1, 1, 1),
            shmem_size=0,
            cooperative_launch=False,
        )

        launch(_to_core_stream(stream), config, func.kernel, *params)

    def ensure_initialized(self, stream=None):
        """If memsys is not initialized, initialize memsys."""
        if self._initialized:
            return
        self.initialize(stream)

    def initialize(self, stream=None):
        """Launch memsys initialization kernel."""
        self.ensure_allocated()

        self._single_thread_launch(self._memsys_module, stream, "NRT_MemSys_init")
        self._initialized = True

        if is_nrt_stats_enabled():
            self.memsys_enable_stats()

    @_alloc_init_guard
    def memsys_enable_stats(self, stream=None):
        """Enable memsys statistics."""
        self._single_thread_launch(
            self._memsys_module, stream, "NRT_MemSys_enable_stats"
        )

    @_alloc_init_guard
    def memsys_disable_stats(self, stream=None):
        """Disable memsys statistics."""
        self._single_thread_launch(
            self._memsys_module, stream, "NRT_MemSys_disable_stats"
        )

    @_alloc_init_guard
    def memsys_stats_enabled(self, stream=None):
        """Return a boolean indicating whether memsys stats are enabled."""
        enabled_ar = cuda.managed_array(1, np.uint8)
        enabled_ptr = enabled_ar.device_ctypes_pointer

        self._single_thread_launch(
            self._memsys_module,
            stream,
            "NRT_MemSys_stats_enabled",
            (enabled_ptr,),
        )

        cuda.synchronize()
        return bool(enabled_ar[0])

    @_alloc_init_guard
    def _copy_memsys_to_host(self, stream):
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

        self._single_thread_launch(
            self._memsys_module, stream, "NRT_MemSys_read", [stats_ptr]
        )
        cuda.synchronize()

        return stats_for_read[0]

    @_alloc_init_guard
    def get_allocation_stats(self, stream=None):
        """Get the allocation statistics."""
        enabled = self.memsys_stats_enabled(stream)
        if not enabled:
            raise RuntimeError("NRT stats are disabled.")
        memsys = self._copy_memsys_to_host(stream)
        return _nrt_mstats(
            alloc=memsys["alloc"],
            free=memsys["free"],
            mi_alloc=memsys["mi_alloc"],
            mi_free=memsys["mi_free"],
        )

    @_alloc_init_guard
    def _get_single_stat(self, stat, stream=None):
        """Get a single stat from the memsys."""
        got = cuda.managed_array(1, np.uint64)
        got_ptr = got.device_ctypes_pointer

        self._single_thread_launch(
            self._memsys_module, stream, f"NRT_MemSys_read_{stat}", [got_ptr]
        )

        cuda.synchronize()
        return got[0]

    @_alloc_init_guard
    def memsys_get_stats_alloc(self, stream=None):
        """Get the allocation statistic."""
        enabled = self.memsys_stats_enabled(stream)
        if not enabled:
            raise RuntimeError("NRT stats are disabled.")
        return self._get_single_stat("alloc")

    @_alloc_init_guard
    def memsys_get_stats_free(self, stream=None):
        """Get the free statistic."""
        enabled = self.memsys_stats_enabled(stream)
        if not enabled:
            raise RuntimeError("NRT stats are disabled.")
        return self._get_single_stat("free")

    @_alloc_init_guard
    def memsys_get_stats_mi_alloc(self, stream=None):
        """Get the mi_alloc statistic."""
        enabled = self.memsys_stats_enabled(stream)
        if not enabled:
            raise RuntimeError("NRT stats are disabled.")
        return self._get_single_stat("mi_alloc")

    @_alloc_init_guard
    def memsys_get_stats_mi_free(self, stream=None):
        """Get the mi_free statistic."""
        enabled = self.memsys_stats_enabled(stream)
        if not enabled:
            raise RuntimeError("NRT stats are disabled.")
        return self._get_single_stat("mi_free")

    def set_memsys_to_module(self, module, stream=None):
        """
        Set the memsys pointer for a module.

        The module must contain `NRT_MemSys_set` kernel and declare a pointer
        to NRT_MemSys structure.
        """
        if self._memsys is None:
            raise RuntimeError(
                "Please allocate NRT Memsys first before setting to module."
            )

        memsys_ptr = self._memsys.device_ctypes_pointer

        self._single_thread_launch(module, stream, "NRT_MemSys_set", [memsys_ptr])

    @_alloc_init_guard
    def print_memsys(self, stream=None):
        """Print the current statistics of memsys, for debugging purposes."""
        cuda.synchronize()
        self._single_thread_launch(self._memsys_module, stream, "NRT_MemSys_print")


# Create the singleton instance
rtsys = _Runtime()
