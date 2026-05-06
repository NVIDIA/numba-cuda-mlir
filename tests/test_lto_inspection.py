# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba_cuda_mlir import cuda
from numba_cuda_mlir.caching import MLIRCache


SIG = "void(float32[::1], float32[::1], float32[::1], int64)"


def _lto_add_kernel(a, b, c, n):
    idx = cuda.grid(1)
    if idx < n:
        c[idx] = a[idx] + b[idx]


def _cached_lto_add_kernel(a, b, c, n):
    idx = cuda.grid(1)
    if idx < n:
        c[idx] = a[idx] + b[idx]


def _cached_non_lto_add_kernel(a, b, c, n):
    idx = cuda.grid(1)
    if idx < n:
        c[idx] = a[idx] + b[idx]


class _FakeCodegen:
    def magic_tuple(self):
        return ("cuda-runtime", (8, 0))


def test_cache_key_separates_lto_and_non_lto_modes():
    lto_cache = MLIRCache(
        _cached_non_lto_add_kernel,
        {"lto": True, "output": "ltoir"},
    )
    non_lto_cache = MLIRCache(
        _cached_non_lto_add_kernel,
        {"lto": False, "output": "ptx"},
    )

    lto_key = lto_cache._index_key(SIG, _FakeCodegen())
    non_lto_key = non_lto_cache._index_key(SIG, _FakeCodegen())

    assert lto_key != non_lto_key


def test_lto_inspect_asm_and_lto_ptx_are_lazy_paths():
    kernel = cuda.jit(lto=True)(_lto_add_kernel)
    kernel.compile(SIG)
    sig = kernel.signatures[0]
    cres = kernel.overloads[sig]

    assert cres.metadata.get("ptx") == ""
    assert "lto_ptx" not in cres.metadata

    ptx = kernel.inspect_asm(sig)
    lto_ptx = kernel.inspect_lto_ptx(sig)

    assert ptx
    assert lto_ptx
    assert cres.metadata["ptx"] == ptx
    assert cres.metadata["lto_ptx"] == lto_ptx


def test_cached_lto_compile_preserves_lazy_inspection():
    first = cuda.jit(lto=True, cache=True)(_cached_lto_add_kernel)
    first.compile(SIG)

    second = cuda.jit(lto=True, cache=True)(_cached_lto_add_kernel)
    second.compile(SIG)
    sig = second.signatures[0]
    cres = second.overloads[sig]

    assert sum(second.stats.cache_hits.values()) > 0
    assert cres.metadata.get("ltoir")
    assert cres.metadata.get("ptx") == ""
    assert "lto_ptx" not in cres.metadata

    ptx = second.inspect_asm(sig)
    lto_ptx = second.inspect_lto_ptx(sig)

    assert ptx
    assert lto_ptx
    assert cres.metadata["ptx"] == ptx
    assert cres.metadata["lto_ptx"] == lto_ptx


def test_cached_non_lto_inspect_lto_ptx_falls_back_to_ptx():
    first = cuda.jit(lto=False, cache=True)(_cached_non_lto_add_kernel)
    first.compile(SIG)

    second = cuda.jit(lto=False, cache=True)(_cached_non_lto_add_kernel)
    second.compile(SIG)
    sig = second.signatures[0]
    cres = second.overloads[sig]

    assert sum(second.stats.cache_hits.values()) > 0
    assert cres.metadata.get("ltoir") is None
    assert second.inspect_lto_ptx(sig) == second.inspect_ptx(sig)
