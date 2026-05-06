# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.numba_cuda.core.errors import ForceLiteralArg, TypingError
from numba_cuda_mlir._mlir.ir import Operation, Value, Block
from typeguard import typechecked
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


ISSUES_URL = "https://gitlab-master.nvidia.com/cuda-python/numba-simt-mlir-compiler/-/issues"


def handle_lowering_error(lower, func_ir):
    """Catch a lowering exception and re-raise as a UserFacingInternalCompilerError.

    Must be called from inside an except block so traceback.format_exc() works.
    Set NUMBA_CUDA_MLIR_ICE_FULL_TB=1 to see the full traceback instead of a summary.
    """
    import os
    import tempfile
    import traceback

    full_tb = traceback.format_exc()
    full_tb_env = os.environ.get("NUMBA_CUDA_MLIR_ICE_FULL_TB", "0").strip()

    log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix="numba_cuda_mlir_error_")
    try:
        with os.fdopen(log_fd, "w") as f:
            f.write(full_tb)
    except Exception:
        log_path = "<unavailable>"

    loc = getattr(lower, "loc", None)
    if loc is None or loc == -1 or not hasattr(loc, "filename"):
        loc = getattr(func_ir, "loc", None)

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


@typechecked
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
