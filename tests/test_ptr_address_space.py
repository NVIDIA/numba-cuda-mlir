# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pointers derived from arrays must keep their address space (issue #317).

``ffi.from_buffer`` and the atomics helper used to build the data pointer via
integer arithmetic (``ptrtoint`` + ``add`` + ``inttoptr``), which cuts LLVM's
address-space inference and leaves every access through the pointer generic
(``LD``/``ST`` instead of ``LDG``/``STG`` or ``LDS``/``STS``).
"""

import ctypes
import re
import shutil

import numpy as np
import pytest
from cffi import FFI

from numba_cuda_mlir import carray, cuda, types
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.cuda.experimental import intrin

requires_nvdisasm = pytest.mark.skipif(shutil.which("nvdisasm") is None, reason="nvdisasm needed")

ffi = FFI()
N = 256


def memory_opcodes(kernel):
    sass = "\n".join(str(v) for v in kernel.inspect_sass().values())
    return set(re.findall(r"\b((?:LD|ST|ATOM|RED)[GS]?)(?:\.E)?\b", sass))


@requires_nvdisasm
def test_from_buffer_device_array_is_global():
    @cuda.jit
    def kernel(a):
        v = carray(ffi.from_buffer(a), N)
        v[cuda.grid(1)] += 1.0

    a = cuda.to_device(np.zeros(N, np.float32))
    kernel[1, N](a)
    assert (a.copy_to_host() == 1).all()
    assert memory_opcodes(kernel) == {"LDG", "STG"}


@requires_nvdisasm
def test_from_buffer_sliced_array_is_global():
    @cuda.jit
    def kernel(a):
        v = carray(ffi.from_buffer(a[N // 2 :]), N // 2)
        v[cuda.grid(1)] += 1.0

    a = cuda.to_device(np.zeros(N, np.float32))
    kernel[1, N // 2](a)
    host = a.copy_to_host()
    assert (host[: N // 2] == 0).all() and (host[N // 2 :] == 1).all()
    assert memory_opcodes(kernel) == {"LDG", "STG"}


@requires_nvdisasm
def test_from_buffer_shared_array_with_offset_is_shared():
    @cuda.jit
    def kernel(a):
        s = cuda.shared.array(N, np.float32)
        v = carray(ffi.from_buffer(s[N // 2 :]), N // 2)
        i = cuda.grid(1)
        v[i] = 1.0
        cuda.syncthreads()
        a[i] = v[N // 2 - 1 - i]

    a = cuda.device_array(N // 2, np.float32)
    kernel[1, N // 2](a)
    assert (a.copy_to_host() == 1).all()
    assert memory_opcodes(kernel) == {"STS", "LDS", "STG"}


@requires_nvdisasm
def test_atomic_on_sliced_array_is_global():
    @cuda.jit
    def kernel(a):
        cuda.atomic.add(a, cuda.grid(1) % 4, 1)

    a = cuda.to_device(np.zeros(N, np.int32))
    kernel[1, N](a[N // 2 :])
    host = a.copy_to_host()
    assert (host[: N // 2] == 0).all()
    assert (host[N // 2 : N // 2 + 4] == N // 4).all()
    assert (host[N // 2 + 4 :] == 0).all()
    assert memory_opcodes(kernel) <= {"REDG", "ATOMG"}
    assert memory_opcodes(kernel)


@requires_nvdisasm
def test_ctypes_pointer_arithmetic_is_global():
    @cuda.jit
    def kernel(a):
        p = ctypes.cast(a, ctypes.POINTER(ctypes.c_float))
        p += cuda.grid(1)
        p[0] = 1.0

    a = cuda.device_array(N, np.float32)
    kernel[1, N](a)
    assert (a.copy_to_host() == 1).all()
    assert memory_opcodes(kernel) == {"STG"}


@requires_nvdisasm
def test_types_ptr_arithmetic_is_global():
    @intrin.define
    def store_f32(ptr: llvm.PointerType.get, value: types.float32) -> types.none:
        llvm.store(value, ptr)

    @cuda.jit
    def kernel(a):
        store_f32(types.ptr(a) + cuda.grid(1) * 4, np.float32(1.0))

    a = cuda.device_array(N, np.float32)
    kernel[1, N](a)
    assert (a.copy_to_host() == 1).all()
    assert memory_opcodes(kernel) == {"STG"}


def test_ctypes_data_includes_view_offset():
    @cuda.jit
    def kernel(a, out):
        out[0] = a[N // 2 :].ctypes.data

    a = cuda.device_array(N, np.float32)
    out = cuda.device_array(1, np.uint64)
    kernel[1, 1](a, out)
    assert out.copy_to_host()[0] == a.__cuda_array_interface__["data"][0] + (N // 2) * 4
