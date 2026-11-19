# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import gc
import weakref

import numpy as np
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda.compiler import CUDACompiler
from numba_cuda_mlir.numba_cuda.core.inline_closurecall import InlineWorker
from numba_cuda_mlir.numba_cuda.descriptor import cuda_target
from numba_cuda_mlir.numba_cuda.flags import Flags

GLOBAL_SCALE = 2.0


def _make_worker():
    flags = Flags()
    tyctx = cuda_target.typing_context
    tgctx = cuda_target.target_context
    compiler_inst = CUDACompiler(tyctx, tgctx, None, None, None, flags, None)
    return InlineWorker(tyctx, tgctx, {}, compiler_inst, flags, None)


def test_global_change_recompiles_inlined_callee():
    global GLOBAL_SCALE
    GLOBAL_SCALE = 2.0
    try:

        @cuda.jit(device=True, inline="always")
        def callee(x):
            return x * GLOBAL_SCALE

        @cuda.jit
        def kernel(a):
            a[0] = callee(a[0])

        a = np.ones(1, dtype=np.float64)
        kernel[1, 1](a)
        assert a[0] == 2.0

        GLOBAL_SCALE = 10.0
        b = np.ones(1, dtype=np.float32)
        kernel[1, 1](b)
        assert b[0] == 10.0

        GLOBAL_SCALE = 20.0
        kernel.recompile()
        c = np.ones(1, dtype=np.float64)
        kernel[1, 1](c)
        assert c[0] == 20.0
    finally:
        GLOBAL_SCALE = 2.0


def test_closure_cell_change_recompiles_inlined_callee():
    def make_callee():
        c = 3.0

        @cuda.jit(device=True, inline="always")
        def callee(x):
            return x + c

        def set_c(value):
            nonlocal c
            c = value

        return callee, set_c

    callee, set_c = make_callee()

    @cuda.jit
    def kernel(a):
        a[0] = callee(a[0])

    a = np.zeros(1, dtype=np.float64)
    kernel[1, 1](a)
    assert a[0] == 3.0

    set_c(7.0)
    b = np.zeros(1, dtype=np.float32)
    kernel[1, 1](b)
    assert b[0] == 7.0

    set_c(11.0)
    kernel.recompile()
    c = np.zeros(1, dtype=np.float64)
    kernel[1, 1](c)
    assert c[0] == 11.0


def test_nested_inlines_share_pipeline_cache(monkeypatch):
    @cuda.jit(device=True, inline="always")
    def leaf(x):
        return x + 1.0

    @cuda.jit(device=True, inline="always")
    def branch(x):
        return leaf(x) + leaf(x)

    def root(x):
        return branch(x) + branch(x)

    runs = []
    run_untyped_passes = InlineWorker.run_untyped_passes

    def counting_run(self, func, enable_ssa=False):
        runs.append(func)
        return run_untyped_passes(self, func, enable_ssa)

    monkeypatch.setattr(InlineWorker, "run_untyped_passes", counting_run)

    worker = _make_worker()
    worker._fresh_callee_ir(root)
    worker._fresh_callee_ir(root)
    assert runs.count(root) == 1
    assert runs.count(branch.py_func) == 1
    assert runs.count(leaf.py_func) == 1

    _make_worker()._fresh_callee_ir(root)
    assert runs.count(root) == 2
    assert runs.count(branch.py_func) == 2
    assert runs.count(leaf.py_func) == 2


def test_cache_released_with_function():
    def populate_cache():
        def callee(x):
            return x + 1.0

        worker = _make_worker()
        worker._fresh_callee_ir(callee)
        return weakref.ref(callee)

    ref = populate_cache()
    gc.collect()
    assert ref() is None
