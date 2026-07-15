# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selective fastmath support.

``fastmath`` accepts a bool or a set/dict of LLVM fast-math flags, applied
per-operation as ``#arith.fastmath<...>`` attributes and carried through
the conversion passes to libnvvm. Three effects need separate handling:
the module-level libnvvm/ptxas knobs (:func:`nvvm_fastmath_options`), the
f32 division rewrite to ``__nv_fast_fdividef`` (libnvvm does not select
``div.approx`` from instruction flags), and the f32 tanh rewrite
(:func:`rewrite_approx_tanh`), which must run before convert-math-to-nvvm
drops the attribute.
"""

import inspect
from functools import cache

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.numba_cuda.core.options import FastMathOptions

# Canonical flag order used by the MLIR arith fastmath attribute printer.
_FLAG_ORDER = ("reassoc", "nnan", "ninf", "nsz", "arcp", "contract", "afn")


def parse_fastmath(value) -> FastMathOptions:
    """Normalize a user-facing fastmath value (bool | set | dict |
    FastMathOptions) into FastMathOptions, validating flag names."""
    return FastMathOptions(value)


def nvvm_fastmath_options(fastmath) -> dict:
    """Map a fastmath flag set to the libnvvm/ptxas knobs it implies.

    A key is absent when the flags do not speak to that knob (the caller
    keeps the toolchain default). ftz has no per-instruction flag, so
    only full ``fast`` enables it.
    """
    flags = parse_fastmath(fastmath).flags
    opts = {}
    if "fast" in flags:
        opts["ftz"] = True
    if flags & {"contract", "fast"}:
        opts["fma"] = True
    if flags & {"arcp", "fast"}:
        opts["prec_div"] = False
    if flags & {"afn", "fast"}:
        opts["prec_sqrt"] = False
    return opts


@cache
def _fastmath_capable_op_names() -> frozenset:
    """Names of arith/math dialect operations that carry a ``fastmath``
    attribute (i.e. implement ArithFastMathInterface), discovered from the
    generated Python bindings so the set tracks the bundled MLIR version."""
    from numba_cuda_mlir._mlir.dialects import _arith_ops_gen, _math_ops_gen

    names = set()
    for mod in (_arith_ops_gen, _math_ops_gen):
        for cls in vars(mod).values():
            if (
                inspect.isclass(cls)
                and hasattr(cls, "OPERATION_NAME")
                and isinstance(inspect.getattr_static(cls, "fastmath", None), property)
            ):
                names.add(cls.OPERATION_NAME)
    return frozenset(names)


def fastmath_attr(flags: set) -> ir.Attribute:
    """Build an ``#arith.fastmath<...>`` attribute for the given flag set.
    Must be called with an active MLIR context."""
    if "fast" in flags:
        mnemonic = "fast"
    else:
        mnemonic = ",".join(f for f in _FLAG_ORDER if f in flags)
    assert mnemonic, f"no valid fastmath flags in {flags}"
    return ir.Attribute.parse(f"#arith.fastmath<{mnemonic}>")


def _chip_number(chip) -> int:
    """sm_89 -> 89; 0 when unknown."""
    if not chip:
        return 0
    digits = "".join(c for c in str(chip) if c.isdigit())
    return int(digits) if digits else 0


def apply_fastmath_to_function(func_op, fastmath) -> None:
    """Stamp the fastmath attribute onto every fastmath-capable op nested
    in ``func_op``. Device callees are cloned in already stamped under
    their own options, so flags scope per function.
    """
    flags = parse_fastmath(fastmath).flags
    if not flags:
        return

    attr = fastmath_attr(flags)
    capable = _fastmath_capable_op_names()

    def _stamp(op):
        if op.name in capable:
            op.attributes["fastmath"] = attr
        return ir.WalkResult.ADVANCE

    func_op.operation.walk(_stamp)


def rewrite_approx_tanh(func_op, fastmath, chip=None) -> None:
    """Replace f32 ``math.tanh`` with ``tanh.approx.f32`` under ``afn`` or
    ``fast`` on sm_75+, as numba-cuda does. Must run before
    convert-math-to-nvvm, which drops the fastmath attribute.
    """
    from numba_cuda_mlir._mlir.dialects import llvm

    flags = parse_fastmath(fastmath).flags
    if not (flags & {"afn", "fast"}) or _chip_number(chip) < 75:
        return

    tanh_ops = []

    def _collect(op):
        if op.name == "math.tanh" and isinstance(op.results[0].type, ir.F32Type):
            tanh_ops.append(op)
        return ir.WalkResult.ADVANCE

    func_op.operation.walk(_collect)

    for op in tanh_ops:
        with ir.InsertionPoint(op), op.location:
            result = llvm.inline_asm(
                op.results[0].type,
                [op.operands[0]],
                "tanh.approx.f32 $0, $1;",
                "=f,f",
            )
        op.results[0].replace_all_uses_with(result)
        op.erase()
