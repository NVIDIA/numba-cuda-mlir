# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import cuda.simt as cs
from cuda.simt import types, compiler
import numpy as np
import pytest


@pytest.mark.parametrize(
    "dtype",
    [
        types.f4E2M1FN,
        types.f6E2M3FN,
        types.f6E3M2FN,
        types.f8E3M4,
        types.f8E4M3B11FNUZ,
        types.f8E4M3FN,
        types.f8E4M3FNUZ,
        types.f8E4M3,
        types.f8E5M2FNUZ,
        types.f8E5M2,
        types.f8E8M0FNU,
        types.tf32,
        types.bf16,
        types.nvfp4,
        types._type_fp8_e5m2,
        types._type_fp8_e4m3,
        types._type_fp8_e8m0,
    ],
)
def test_print_exotic_float_types(dtype):
    if dtype in (types.nvfp4, types.tf32):
        return pytest.skip("NYI")

    @cs.jit(opt_level=3)
    def k(x: cs.DeviceNDArray):
        print(x)

    mlir = compiler.compile_mlir(k, types.void(dtype[:]))
    assert "test_print_exotic_float_types" in mlir

    mlir = compiler.compile_mlir(k, types.void(dtype[:]), optimized=True)
    assert "test_print_exotic_float_types" in mlir
