# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Per-op ``#arith.fastmath`` stamping, module-level libnvvm/ptxas flags, and f32 div/tanh rewrites."""

import inspect
from functools import cache

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.numba_cuda.core.options import FastMathOptions

# Flag order used by the attribute printer.
_FLAG_ORDER = ("reassoc", "nnan", "ninf", "nsz", "arcp", "contract", "afn")

_MODULE_ONLY_FLAGS = frozenset({"ftz"})


def parse_fastmath(value) -> FastMathOptions:
    """Normalize a user-facing fastmath value (bool | set | dict |
    FastMathOptions) into FastMathOptions, validating flag names."""
    return FastMathOptions(value)


def nvvm_fastmath_options(fastmath) -> dict:
    """Module-level libnvvm/ptxas flags; absent keys keep toolchain defaults."""
    flags = parse_fastmath(fastmath).flags
    opts = {}
    if flags & {"ftz", "fast"}:
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
    """arith/math op names carrying a ``fastmath`` attribute, discovered from the generated bindings."""
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
    """Build an ``#arith.fastmath<...>`` attribute; needs an active MLIR context."""
    if "fast" in flags:
        # fast keeps NaN/Inf checks intact, as CUDA fast math does; nnan/ninf apply only when named.
        flags = (flags - {"fast"}) | (set(_FLAG_ORDER) - {"nnan", "ninf"})
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
    """Stamp the fastmath attribute onto every fastmath-capable op nested in ``func_op``."""
    flags = parse_fastmath(fastmath).flags - _MODULE_ONLY_FLAGS
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
    """Replace f32 ``math.tanh`` with ``tanh.approx.f32`` under ``afn``/``fast`` on sm_75+; runs before convert-math-to-nvvm drops the attribute."""
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
