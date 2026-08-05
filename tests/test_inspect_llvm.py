# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from numba_cuda_mlir import cuda, types
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

    function_name = generate_mangled_name(cres.fndesc.qualname, cres.fndesc.argtypes)
    assert f'source_filename = "{source_filename}"' in llvm_ir
    assert f"{function_name}({ptr_type}" in llvm_ir
    assert foo.inspect_llvm()[args] == llvm_ir
