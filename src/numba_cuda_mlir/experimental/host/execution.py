# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import threading
from functools import lru_cache

from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.typing.ctypes_utils import to_ctypes

try:
    from numba_cuda_mlir._mlir import execution_engine
except ImportError as exc:
    execution_engine = None
    _EXECUTION_ENGINE_IMPORT_ERROR = exc
else:
    _EXECUTION_ENGINE_IMPORT_ERROR = None


@lru_cache(maxsize=None)
def scalar_ctype(typ):
    typ = types.unliteral(typ)
    if (
        isinstance(typ, types.Boolean)
        or isinstance(typ, types.Integer)
        and typ.bitwidth in (8, 16, 32, 64)
        or isinstance(typ, types.Float)
        and typ.bitwidth in (32, 64)
    ):
        return to_ctypes(typ)
    raise TypeError(
        f"host.jit supports only boolean, integer, and float32/float64 scalars; got {typ}"
    )


def _prepare_argument_slots(arg_ctypes, args):
    """Use ctypes coercion: integer narrowing wraps, integers may become floats,
    and floats cannot become integers. Specialization and validation belong to
    the dispatcher, not this Python-to-native slot preparation policy.
    """
    return [ctype(value) for ctype, value in zip(arg_ctypes, args, strict=True)]


class ExecutionEngine:
    def __init__(self, module, signature, symbol_name, targetoptions):
        if execution_engine is None:
            raise RuntimeError(
                "host.jit requires the MLIR ExecutionEngine Python binding "
                "(_mlirExecutionEngine), but it is not available in this build"
            ) from _EXECUTION_ENGINE_IMPORT_ERROR
        self.module = module
        self.signature = signature
        self.engine = execution_engine.ExecutionEngine(module, opt_level=targetoptions["opt_level"])
        self._errors = threading.local()
        # The native engine retains callbacks; do not capture its owner.
        errors = self._errors

        @ctypes.CFUNCTYPE(None, ctypes.c_int32)
        def report_error(code):
            errors.code = code

        self._report_error = report_error
        self.engine.register_runtime("host_scalar_error", report_error)
        self._packed_function = self.engine.lookup(symbol_name)

    def get_callable(self):
        arg_ctypes = tuple(scalar_ctype(typ) for typ in self.signature.args)
        return_type = self.signature.return_type
        return_ctype = (
            None if isinstance(return_type, types.NoneType) else scalar_ctype(return_type)
        )
        packed_type = ctypes.c_void_p * (len(arg_ctypes) + (return_ctype is not None))

        def call(*args):
            slots = _prepare_argument_slots(arg_ctypes, args)
            result = return_ctype() if return_ctype is not None else None
            if result is not None:
                slots.append(result)
            # ExecutionEngine's packed trampoline loads scalar values through
            # pointers; the scalar return buffer is the final packed slot.
            packed = packed_type(*(ctypes.addressof(slot) for slot in slots))
            self._errors.code = 0
            self._packed_function(packed)
            code = self._errors.code
            if code == 1:
                raise ZeroDivisionError("division by zero in host.jit")
            if code == 2:
                raise ValueError("range() arg 3 must not be zero")
            if code == 3:
                raise OverflowError("signed integer division overflow in host.jit")
            if code == 4:
                raise ValueError("cannot convert float NaN to integer")
            if code == 5:
                raise OverflowError("float-to-integer conversion overflow in host.jit")
            if code:
                raise RuntimeError(f"host.jit scalar execution failed with code {code}")
            return result.value if result is not None else None

        return call
