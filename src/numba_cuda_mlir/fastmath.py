# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selective fastmath support.

The ``fastmath`` target option accepts, like numba-cuda, either a bool or a
set/dict of individual LLVM fast-math flags ({'fast', 'nnan', 'ninf', 'nsz',
'arcp', 'contract', 'afn', 'reassoc'}).  The flags are applied per-operation
as ``#arith.fastmath<...>`` attributes on every arith/math dialect op that
implements the ArithFastMathInterface.  The conversion passes in the standard
pipeline (convert-arith-to-llvm, convert-gpu-to-nvvm, convert-math-to-nvvm)
translate these into LLVM fast-math flags, which libnvvm honours
per-instruction (e.g. ``arcp`` selects ``div.approx`` for f32 division).

Two effects have no per-instruction encoding and are handled separately:

- The module-level libnvvm/ptxas knobs (``-ftz``, ``-fma``, ``-prec-div``,
  ``-prec-sqrt``) are derived from the flag set by
  :func:`nvvm_fastmath_options`.
- f32 ``math.tanh`` is rewritten to ``tanh.approx.f32`` by
  :func:`rewrite_approx_tanh`; this must happen while the op is still a
  ``math.tanh``, because ``convert-math-to-nvvm`` lowers it to a plain
  libdevice call and drops the fastmath attribute in the process.
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
    """Map a fastmath flag set to the module-level libnvvm/ptxas knobs it
    implies.

    Returns a dict with keys ``ftz``, ``fma``, ``prec_div`` and
    ``prec_sqrt``; a key is absent when the flag set does not speak to that
    knob (callers leave the toolchain default in place).

    Each knob is enabled only by the flag that requests the corresponding
    relaxation: ``arcp`` (approximate reciprocal/division) implies
    ``prec_div=False``; ``afn`` (approximate functions) implies
    ``prec_sqrt=False``; ``contract`` implies ``fma=True``; denormal
    flushing has no per-instruction fast-math flag, so ``ftz=True`` is
    implied only by full ``fast``.  ``fastmath=True`` (i.e. ``{'fast'}``)
    enables all four, matching numba-cuda's ``nvvm.compile_ir`` gating for
    the bool form.
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
    """Stamp the fastmath attribute onto every fastmath-capable op nested in
    ``func_op``.

    Called once per lowered function, before device callees are considered:
    callee functions are compiled separately with their own target options and
    cloned into the caller's module already stamped, so walking only the
    caller's func op scopes the flags exactly like numba-cuda's
    per-instruction fast-math flags.
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
    """Replace f32 ``math.tanh`` with the hardware ``tanh.approx.f32``
    instruction when the flags permit approximate functions (``afn`` or
    ``fast``) and the target is sm_75+, as numba-cuda does.

    This runs at stamping time rather than with the pre-codegen rewrites
    (see optimization/__init__.py) because ``convert-math-to-nvvm`` lowers
    ``math.tanh`` to a plain ``__nv_tanhf`` libdevice call and discards the
    fastmath attribute, so after conversion there is nothing left to key
    the rewrite on.
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
