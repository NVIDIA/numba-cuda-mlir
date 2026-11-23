# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from types import SimpleNamespace

import pytest

from numba_cuda_mlir import _context_cache
from numba_cuda_mlir.compiler import CodeLibrary
from numba_cuda_mlir.numba_cuda.cudadrv import devices


@pytest.mark.parametrize("load_function_first", [False, True])
def test_legacy_module_access_follows_context_and_reset(monkeypatch, load_function_first):
    from cuda.bindings import driver
    from numba_cuda_mlir.numba_cuda.cudadrv import driver as cudadrv

    def context(handle):
        ctx = object.__new__(cudadrv.Context)
        ctx.handle = handle
        ctx.device = SimpleNamespace(id=handle)
        ctx.modules = {}
        ctx.deallocations = cudadrv._PendingDeallocs()
        ctx.memory_manager = SimpleNamespace(reset=lambda: None)
        ctx.extras = {}
        return ctx

    a, b = context(100), context(200)
    current = [a]
    monkeypatch.setattr(devices, "get_context", lambda: current[0])
    monkeypatch.setattr(_context_cache, "_get_current_context", lambda: current[0].handle)
    handles = {}
    unloaded = []

    def load(data):
        assert data == b"cubin"
        handle = object()
        handles[handle] = object()  # The CUmodule for this CUlibrary.
        return SimpleNamespace(handle=handle)

    monkeypatch.setattr(cudadrv, "ObjectCode", SimpleNamespace(from_cubin=load))
    monkeypatch.setattr(cudadrv, "_logger", logging.getLogger(__name__), raising=False)
    monkeypatch.setitem(vars(cudadrv.driver), "cuLibraryGetModule", handles.__getitem__)
    monkeypatch.setitem(vars(cudadrv.driver), "cuLibraryUnload", unloaded.append)
    monkeypatch.setitem(vars(cudadrv.driver), "cuModuleGetFunction", lambda module, name: module)
    ok = driver.CUresult.CUDA_SUCCESS
    monkeypatch.setattr(driver, "cuModuleGetGlobal", lambda module, name: (ok, module, 4))
    library = CodeLibrary(b"cubin", "kernel")

    def query_global():
        if load_function_first:
            library.get_cufunc()
        # Numbast accesses this handle directly to look up linked globals.
        err, pointer, size = driver.cuModuleGetGlobal(library._module, b"linked_global")
        assert err == ok
        assert size == 4
        assert pointer is library.get_cufunc()._handle
        return pointer

    first = query_global()
    assert len(handles) == 1
    current[0] = b
    second = query_global()
    assert second is not first
    assert len(handles) == 2
    current[0] = a
    assert query_global() is first
    assert len(handles) == 2

    # Reset the same Context object: its previous module must not be returned.
    (first_library,) = a.modules
    a.reset()
    assert unloaded == [first_library]
    assert not a.modules
    assert len(b.modules) == 1
    replacement = query_global()
    assert replacement is not first
    assert replacement is not second
    assert len(handles) == 3
    a.reset()
    b.reset()
    assert len(unloaded) == len(set(unloaded)) == 3
