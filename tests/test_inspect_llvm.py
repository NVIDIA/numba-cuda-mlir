# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from numba_cuda_mlir import cuda, mlir_optimization, types
from numba_cuda_mlir.tools import generate_mangled_name


@pytest.mark.parametrize(
    "chip,source_filename,ptr_type",
    [
        pytest.param("sm_90", "llvm70_module", "i8*", id="typed-pointers"),
        pytest.param("sm_100", "LLVMDialectModule", "ptr %0", id="opaque-pointers"),
    ],
)
def test_inspect_llvm_uses_architecture_natural_ir(chip, source_filename, ptr_type):
    @cuda.jit(device=True, chip=chip)
    def foo(arr, value):
        arr[0] = value

    args = (types.int32[:], types.int32)
    cres = foo.compile_device(args)
    llvm_ir = foo.inspect_llvm(args)

    if os.name == "nt" and chip == "sm_100":
        source_filename = "numba-cuda-mlir-gpu-module"
    function_name = generate_mangled_name(cres.fndesc.qualname, cres.fndesc.argtypes)
    assert f'source_filename = "{source_filename}"' in llvm_ir
    assert f"{function_name}({ptr_type}" in llvm_ir
    assert foo.inspect_llvm()[args] == llvm_ir


@pytest.mark.parametrize("lto", [False, True])
def test_inspect_llvm_preserves_llvm70_lto_debug_mode(lto):
    @cuda.jit(device=True, chip="sm_90", debug=True, lto=lto, opt=False)
    def foo(value):
        return value + 1

    llvm_ir = foo.inspect_llvm((types.int32,))
    assert ("Debug Info Version" in llvm_ir) is not lto


def test_inspect_llvm_windows_preserves_debug_info(monkeypatch):
    @cuda.jit(device=True, chip="sm_100", debug=True, opt=False)
    def foo(value):
        return value + 1

    foo.compile_device((types.int32,))
    captured = {}

    def translate(mlir_text, *args, **kwargs):
        captured["mlir_text"] = mlir_text
        assert kwargs["inspect_llvmir"]
        return b"; LLVM IR"

    monkeypatch.setattr(mlir_optimization.os, "name", "nt")
    monkeypatch.setattr(mlir_optimization, "translate_gpu_module_to_libnvvm_ir", translate)

    assert foo.inspect_llvm((types.int32,)) == "; LLVM IR"
    assert "loc(" in captured["mlir_text"]
