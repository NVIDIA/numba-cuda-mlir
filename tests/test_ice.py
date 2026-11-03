# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import inspect
from pathlib import Path

import pytest

from numba_cuda_mlir import cuda, extending, types
from numba_cuda_mlir.errors import (
    InternalCompilerError,
    UserFacingInternalCompilerError,
    handle_lowering_error,
)
from numba_cuda_mlir.numba_cuda.core.errors import UnsupportedError
from numba_cuda_mlir.numba_cuda.core.ir import Loc


@pytest.mark.parametrize("full_tb", ["0", "1"])
@pytest.mark.parametrize("device", [False, True])
@pytest.mark.parametrize("debug", [False, True])
def test_missing_lowering_reports_user_instruction(monkeypatch, full_tb, device, debug):
    monkeypatch.setenv("NUMBA_CUDA_MLIR_ICE_FULL_TB", full_tb)

    @extending.intrinsic
    def unsupported(typingctx, value):
        def codegen(builder, target, args, kwargs):
            raise NotImplementedError("unsupported test operation")

        return value(value), codegen

    extending.refresh_registries()

    def callee(value):
        return unsupported(value)

    callee_line = callee.__code__.co_firstlineno + 1
    if device:
        device_callee = cuda.jit(device=True, debug=debug, opt=not debug)(callee)

        def kernel(out):
            out[0] = device_callee(out[0])
    else:

        def kernel(out):
            out[0] = unsupported(out[0])

    expected_line = callee_line if device else kernel.__code__.co_firstlineno + 1
    compiled = cuda.jit(debug=debug, opt=not debug)(kernel)
    with pytest.raises(UnsupportedError) as caught:
        compiled.compile((types.int64[::1],))

    error = caught.value
    assert error.loc.filename == __file__
    assert error.loc.line == expected_line
    assert "unsupported test operation" in error.msg
    if full_tb == "0":
        assert error.__suppress_context__
    else:
        assert isinstance(error.__cause__, NotImplementedError)


@pytest.mark.parametrize("full_tb", ["0", "1"])
def test_internal_compiler_failure_keeps_ice_policy(monkeypatch, full_tb):
    monkeypatch.setenv("NUMBA_CUDA_MLIR_ICE_FULL_TB", full_tb)

    @extending.intrinsic
    def broken(typingctx, value):
        def codegen(builder, target, args, kwargs):
            raise InternalCompilerError("invalid compiler invariant")

        return value(value), codegen

    extending.refresh_registries()

    @cuda.jit
    def kernel(out):
        out[0] = broken(out[0])

    expected = InternalCompilerError if full_tb == "1" else UserFacingInternalCompilerError
    with pytest.raises(expected) as caught:
        kernel.compile((types.int64[::1],))
    assert not isinstance(caught.value, UnsupportedError)


def test_package_lowering_location_replaced_by_user_call(monkeypatch):
    monkeypatch.setenv("NUMBA_CUDA_MLIR_ICE_FULL_TB", "0")
    package_file = inspect.getsourcefile(handle_lowering_error)
    user_loc = Loc(__file__, 1)
    assert package_file is not None
    implementation_loc = Loc(package_file, 1)

    class Lower:
        loc = user_loc

    class FuncIR:
        loc = Loc(__file__, 2)

    with pytest.raises(UnsupportedError) as caught:
        try:
            raise UnsupportedError("unsupported nested operation", loc=implementation_loc)
        except UnsupportedError:
            handle_lowering_error(Lower(), FuncIR())

    assert caught.value.loc is user_loc
    assert caught.value.msg == "unsupported nested operation"
    assert str(Path(package_file)) not in str(caught.value)
