# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.tools import generate_mangled_name

import pytest


class TestDispatcherMethods:
    @pytest.mark.parametrize("legacy", [False, True])
    def test_inspect_llvm(self, legacy):
        @cuda.jit(device=True)
        def foo(x, y):
            return x + y

        args = (types.int32, types.int32)
        cres = foo.compile_device(args)

        fname = generate_mangled_name(cres.fndesc.qualname, cres.fndesc.argtypes)
        # Verify that the function name has "foo" in it as in the python name
        assert "foo" in fname

        # Check that the compiled function name is in the LLVM
        llvm = foo.inspect_llvm(args, legacy=legacy)
        assert fname in llvm
