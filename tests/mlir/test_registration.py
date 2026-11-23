# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The bundled MLIR package registers only what numba-cuda-mlir uses
# (cext/mlir-python/Registration.cpp). These tests catch a dialect or pass
# that is used but no longer registered.
import pytest

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.passmanager import PassManager
from numba_cuda_mlir.mlir_optimization import get_base_pipeline

USED_DIALECTS = [
    "arith",
    "bufferization",
    "builtin",
    "cf",
    "complex",
    "func",
    "gpu",
    "linalg",
    "llvm",
    "math",
    "memref",
    "nvgpu",
    "nvvm",
    "scf",
    "shape",
    "tensor",
    "vector",
]


def test_base_pipeline_parses():
    with ir.Context():
        PassManager.parse(get_base_pipeline())


@pytest.mark.parametrize("name", USED_DIALECTS)
def test_used_dialect_is_registered(name):
    with ir.Context() as ctx:
        assert ctx.dialects[name].descriptor.namespace == name
