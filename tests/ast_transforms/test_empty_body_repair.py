# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import ast

import numba_cuda_mlir
from numba_cuda_mlir.ast_transforms.empty_body import repair_empty_bodies
from numba_cuda_mlir.cuda.experimental import consteval
from numba_cuda_mlir import cuda
import numpy as np


def test_repair_inserts_pass_into_empty_bodies():
    """Emptied if and for bodies each receive a single pass and compile."""
    tree = ast.parse("def f(x):\n    if x:\n        pass\n    for i in x:\n        pass\n")
    tree.body[0].body[0].body = []
    tree.body[0].body[1].body = []
    tree, modified = repair_empty_bodies(tree)
    assert modified
    compile(tree, "<test>", "exec")
    assert isinstance(tree.body[0].body[0].body[0], ast.Pass)
    assert isinstance(tree.body[0].body[1].body[0], ast.Pass)


def test_repair_leaves_populated_bodies_alone():
    """Bodies that still hold statements are not touched."""
    tree = ast.parse("def f(x):\n    if x:\n        return 1\n    return 2\n")
    tree, modified = repair_empty_bodies(tree)
    assert not modified
    assert ast.unparse(tree) == "def f(x):\n    if x:\n        return 1\n    return 2"


def test_zero_trip_loop_as_only_statement():
    """A zero-trip consteval loop that is the whole kernel body compiles to pass."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        for j in consteval(range(0)):
            arr[j] = 1.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert source.splitlines()[-2:] == ["def kernel(arr):", "    pass"]


def test_false_if_as_only_statement_in_loop():
    """A folded-away if that was the whole loop body leaves a pass loop body."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        for j in range(2):
            if consteval(False):
                arr[i] = 999.0
        arr[i] = 1.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert "for j in range(2):\n        pass" in source
    assert "999.0" not in source

    a = np.zeros(32, dtype=np.float32)
    d_a = cuda.to_device(a)
    kernel[1, 32](d_a)
    result = d_a.copy_to_host()

    assert all(result == 1.0)


def test_consteval_block_as_only_statement_in_if():
    """A removed consteval block that was the whole if body leaves a pass if body."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        if i > 0:
            with consteval():
                _unused = 1
        arr[i] = 1.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert "if i > 0:\n        pass" in source
    assert "with consteval" not in source

    a = np.zeros(32, dtype=np.float32)
    d_a = cuda.to_device(a)
    kernel[1, 32](d_a)
    result = d_a.copy_to_host()

    assert all(result == 1.0)


def test_none_statement_as_only_statement_in_if():
    """A removed None statement that was the whole if body leaves a pass if body."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        i = cuda.threadIdx.x
        if i > 0:
            consteval(None)
        arr[i] = 1.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert "if i > 0:\n        pass" in source
    assert "None" not in source


def test_repair_runs_after_all_passes():
    """Unrolling empties the if body, folding empties the kernel body, one pass remains."""

    @numba_cuda_mlir.cuda.jit
    def kernel(arr):
        if consteval(True):
            for j in consteval(range(0)):
                arr[j] = 1.0

    cres = kernel.compile("void(float32[:])")
    source = cres.metadata["transformed_source"]
    assert source is not None
    assert source.splitlines()[-2:] == ["def kernel(arr):", "    pass"]
