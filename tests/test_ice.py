# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from numba_cuda_mlir.errors import (
    InternalCompilerError,
    UserFacingInternalCompilerError,
    handle_lowering_error,
    ISSUES_URL,
)


class _FakeLoc:
    filename = __file__
    line = 10
    col = 4

    def strformat(self):
        return (
            f'File "{self.filename}", line {self.line}:\n    fake_source_line()\n    ^'
        )


class _FakeLower:
    loc = _FakeLoc()


class _FakeFuncIR:
    loc = _FakeLoc()


def _call_handler(lower, func_ir, exc=None):
    exc = exc or NotImplementedError("test ice")
    try:
        raise exc
    except (NotImplementedError, InternalCompilerError):
        handle_lowering_error(lower, func_ir)


def test_ice_from_internal_compiler_error():
    with pytest.raises(UserFacingInternalCompilerError):
        _call_handler(_FakeLower(), _FakeFuncIR(), InternalCompilerError("oops"))


def test_user_errors_not_caught():
    """ValueError and other user errors must propagate, not become ICEs."""
    with pytest.raises(ValueError):
        try:
            raise ValueError("bad alignment")
        except (NotImplementedError, InternalCompilerError):
            handle_lowering_error(_FakeLower(), _FakeFuncIR())


def test_ice_raises_user_facing_error():
    with pytest.raises(UserFacingInternalCompilerError):
        _call_handler(_FakeLower(), _FakeFuncIR())


def test_ice_contains_issues_url():
    with pytest.raises(UserFacingInternalCompilerError) as exc_info:
        _call_handler(_FakeLower(), _FakeFuncIR())
    assert ISSUES_URL in str(exc_info.value)


def test_ice_contains_log_path():
    with pytest.raises(UserFacingInternalCompilerError) as exc_info:
        _call_handler(_FakeLower(), _FakeFuncIR())
    assert "numba_cuda_mlir_error_" in str(exc_info.value)


def test_ice_contains_source_location():
    with pytest.raises(UserFacingInternalCompilerError) as exc_info:
        _call_handler(_FakeLower(), _FakeFuncIR())
    assert "fake_source_line" in str(exc_info.value)


def test_ice_fallback_to_func_ir_loc():
    class LowerWithNoLoc:
        loc = -1

    with pytest.raises(UserFacingInternalCompilerError) as exc_info:
        _call_handler(LowerWithNoLoc(), _FakeFuncIR())
    assert "fake_source_line" in str(exc_info.value)


def test_ice_message(capsys):
    """Show the full ICE message - run with pytest -s to read it."""
    with pytest.raises(UserFacingInternalCompilerError) as exc_info:
        _call_handler(_FakeLower(), _FakeFuncIR())
    print(f"\n{exc_info.value}")
