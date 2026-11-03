# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.numba_cuda.core.errors import ForceLiteralArg, TypingError, UnsupportedError
from numba_cuda_mlir._mlir.ir import Operation, Value, Block
from typing import TypeVar
import inspect
from pathlib import Path

T = TypeVar("T")


class MultipleIntrinsicFunctionsError(Exception):
    def __init__(self, func_names: list[str]):
        names_str = ", ".join(func_names)
        super().__init__(
            f"inline_intrinsic requires exactly one function in the MLIR source, "
            f"but found {len(func_names)}: {names_str}. "
            f"Use declare_mlir_library() for multiple functions."
        )


class UnsupportedIntrinsicTypeError(Exception):
    def __init__(self, mlir_type: str, context: str = ""):
        ctx_msg = f" in {context}" if context else ""
        super().__init__(f"Unsupported MLIR type '{mlir_type}'{ctx_msg}.")


class InternalCompilerError(RuntimeError):
    def __init__(self, message: str, *args, **kwargs):
        message += " This is always a bug in the compiler, please file an issue."
        super().__init__(message, *args, **kwargs)


class UserFacingInternalCompilerError(RuntimeError):
    """User-facing error for internal compiler failures. Full details are in a log file."""

    pass


ISSUES_URL = "https://github.com/NVIDIA/numba-cuda-mlir/issues"


def handle_lowering_error(lower, func_ir):
    """Report unsupported lowerings separately from internal compiler failures.

    Must be called from inside an except block so traceback.format_exc() works.
    Set NUMBA_CUDA_MLIR_ICE_FULL_TB=1 to see the full traceback instead of a summary.
    """
    import os
    import sys
    import tempfile
    import traceback

    exc = sys.exception()
    loc = getattr(lower, "loc", None)
    if loc is None or loc == -1 or not hasattr(loc, "filename"):
        loc = getattr(func_ir, "loc", None)

    if isinstance(exc, (NotImplementedError, UnsupportedError)):
        if isinstance(exc, UnsupportedError):
            # Keep a user callee's location; replace an overload implementation's
            # package location with the caller's instruction.
            if exc.loc is not None and not Path(exc.loc.filename).resolve().is_relative_to(
                Path(__file__).resolve().parent
            ):
                raise exc
            message = exc.msg
        else:
            message = f"Unsupported operation in MLIR lowering: {exc}"
        error = UnsupportedError(message, loc=loc)
        if os.environ.get("NUMBA_CUDA_MLIR_ICE_FULL_TB", "0").strip() == "1":
            raise error from exc
        raise error from None
    full_tb = traceback.format_exc()
    full_tb_env = os.environ.get("NUMBA_CUDA_MLIR_ICE_FULL_TB", "0").strip()

    log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix="numba_cuda_mlir_error_")
    try:
        with os.fdopen(log_fd, "w") as f:
            f.write(full_tb)
    except Exception:
        log_path = "<unavailable>"

    location_str = f"\n{loc.strformat()}" if loc is not None else ""

    msg = (
        f"This line in your program has triggered a compiler bug:{location_str}\n\n"
        f"Please open an issue at:\n"
        f"  {ISSUES_URL}\n"
        f"and attach the full traceback saved to:\n"
        f"  {log_path}\n\n"
        f"NOTE: To quell this message and see the full traceback, set the environment variable NUMBA_CUDA_MLIR_ICE_FULL_TB=1."
    )
    if full_tb_env == "1":
        raise UserFacingInternalCompilerError(msg)
    raise UserFacingInternalCompilerError(msg) from None


def _excepthook(exc_type, exc_value, exc_tb):
    import os
    import sys

    if issubclass(exc_type, UserFacingInternalCompilerError):
        full_tb_env = os.environ.get("NUMBA_CUDA_MLIR_ICE_FULL_TB", "0").strip()
        if full_tb_env == "1":
            _original_excepthook(exc_type, exc_value, exc_tb)
        else:
            sys.stderr.write(f"{exc_value}\n")
    else:
        _original_excepthook(exc_type, exc_value, exc_tb)


import sys as _sys

_original_excepthook = _sys.excepthook
_sys.excepthook = _excepthook


class ExtensionError(Exception):
    def __init__(self, message: str, *args, **kwargs):
        prefix = "Improper Configuration: "
        super().__init__(prefix + message, *args, **kwargs)


def ensure_verifies(op: T) -> T:
    if not isinstance(op, (Operation, Value, Block)):
        return op
    owner = op if isinstance(op, Operation) else op.owner
    if isinstance(owner, Block):
        return op

    try:
        owner.verify()
    except Exception as e:
        frame = inspect.stack()[1]
        file = Path(frame.filename).name
        # logging.debug(f'TRACE: {file}:{frame.lineno} {frame.function}(): {arg if arg is not None else ""}')
        asm = str(owner)
        raise InternalCompilerError(
            f"Operation {owner.name} does not verify:"
            + f"\n\n{e}"
            + f"\n\nDefined in {file}:{frame.lineno}::{frame.function}()\n\n"
            + asm
            + "\n\n"
        )
    return op
