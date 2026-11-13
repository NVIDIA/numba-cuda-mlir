# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import inspect
import os
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


@pytest.mark.parametrize("device", [False, True])
@pytest.mark.parametrize("debug", [False, True])
def test_missing_lowering_reports_user_instruction(monkeypatch, device, debug):
    monkeypatch.setenv("NUMBA_CUDA_MLIR_ICE_FULL_TB", "0")

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


@pytest.mark.parametrize("full_tb", ["0", "1"])
def test_package_lowering_failure_reports_user_call(monkeypatch, full_tb):
    monkeypatch.setenv("NUMBA_CUDA_MLIR_ICE_FULL_TB", full_tb)
    package_dir = Path(inspect.getsourcefile(handle_lowering_error)).parent

    def kernel(out):
        value = out[0]
        out[1] = out[2] in (value, 3)

    expected_line = kernel.__code__.co_firstlineno + 2
    with pytest.raises(UnsupportedError) as caught:
        cuda.jit(kernel).compile((types.int64[::1],))

    error = caught.value
    assert error.loc.filename == __file__
    assert error.loc.line == expected_line
    assert "Tuple contains is not implemented" in error.msg
    assert str(package_dir) not in str(error)
    assert os.path.relpath(package_dir) not in str(error)
    diagnostic = str(error)
    if full_tb == "0":
        assert error.__suppress_context__
        assert "Failed in " not in diagnostic
        assert "During: " not in diagnostic
    else:
        assert isinstance(error.__cause__, NotImplementedError)
        assert "Failed in " in diagnostic
        assert "During: " in diagnostic


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
