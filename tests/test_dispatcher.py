# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.tools import generate_mangled_name

import pytest


class TestDispatcherMethods:
    @pytest.mark.parametrize(
        "legacy,source_filename,ptr_type",
        [
            pytest.param(False, "LLVMDialectModule", "ptr %0", id="modern-IR"),
            pytest.param(True, "llvm70_module", "i8*", id="legacy-IR"),
        ],
    )
    def test_inspect_llvm(self, legacy, source_filename, ptr_type):
        @cuda.jit(device=True)
        def foo(arr, y):
            arr[0] = y

        args = (types.int32[:], types.int32)
        cres = foo.compile_device(args)

        fname = generate_mangled_name(cres.fndesc.qualname, cres.fndesc.argtypes)
        # Verify that the function name has "foo" in it as in the python name
        assert "foo" in fname

        llvm = foo.inspect_llvm(args, legacy=legacy)

        # Check that the compiled function name and expected pointer type
        # made it to the resulting LLVM
        assert f'source_filename = "{source_filename}"' in llvm
        assert f"{fname}({ptr_type}" in llvm
