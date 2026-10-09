# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from numba_cuda_mlir.compiler import CodeLibrary
from numba_cuda_mlir.numba_cuda.cudadrv import devices


@pytest.mark.parametrize("load_function_first", [False, True])
def test_legacy_module_access_follows_context_and_reset(monkeypatch, load_function_first):
    from cuda.bindings import driver

    a = SimpleNamespace(extras={})
    b = SimpleNamespace(extras={})
    current = [a]
    monkeypatch.setattr(devices, "get_context", lambda: current[0])
    modules = []
    ok = driver.CUresult.CUDA_SUCCESS

    def load(data):
        assert data == b"cubin"
        module = object()
        modules.append(module)
        return ok, module

    monkeypatch.setattr(driver, "cuModuleLoadData", load)
    monkeypatch.setattr(driver, "cuModuleGetFunction", lambda module, name: (ok, module))
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
    assert len(modules) == 1
    current[0] = b
    second = query_global()
    assert second is not first
    assert len(modules) == 2
    current[0] = a
    assert query_global() is first
    assert len(modules) == 2

    # Reset the same Context object: its previous module must not be returned.
    a.extras.clear()
    replacement = query_global()
    assert replacement is not first
    assert replacement is not second
    assert len(modules) == 3
