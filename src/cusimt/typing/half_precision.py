# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing for half-precision (fp16/bf16) intrinsic functions.

Registers bf16 intrinsics dynamically at typing context initialization time
to handle module identity issues with the cusimt redirector.
"""

from numba.core.typing.templates import (
    ConcreteTemplate,
    Registry,
    signature,
)
from numba import types
from cusimt.numba_cuda.types.ext_types import bfloat16 as bf16

registry = Registry()

# List of bf16 intrinsic names - only unary ops exist in numba-cuda's cuda_bf16
_BF16_UNARY_INTRINSICS = [
    "htrunc",
    "hceil",
    "hfloor",
    "hrint",
    "hsqrt",
    "hrsqrt",
    "hrcp",
    "hlog",
    "hlog2",
    "hlog10",
    "hcos",
    "hsin",
    "hexp",
    "hexp2",
    "hexp10",
    "htanh",
    "htanh_approx",
]


def register_bf16_globals():
    """Register bf16 intrinsics as globals. Must be called after bf16 module is imported."""
    import sys

    mod = sys.modules.get("numba.cuda._internal.cuda_bf16")
    if not mod:
        return

    # Create and register templates for unary intrinsics
    for name in _BF16_UNARY_INTRINSICS:
        func = getattr(mod, name, None)
        if func:

            class UnaryTemplate(ConcreteTemplate):
                key = func
                cases = [signature(bf16, bf16)]

            UnaryTemplate.__name__ = f"BF16_{name}_Template"
            registry.register(UnaryTemplate)
            registry.register_global(func, types.Function(UnaryTemplate))

    # Register typing for bitcast functions that also accept integer types
    # This allows passing integer bit patterns to be treated as bf16
    _bfloat16_as_short = getattr(mod, "__bfloat16_as_short", None)
    if _bfloat16_as_short:
        _key = _bfloat16_as_short

        class BF16AsShortTemplate(ConcreteTemplate):
            key = _key
            cases = [
                signature(types.int16, bf16),
                signature(types.int16, types.int64),
                signature(types.int16, types.int32),
                signature(types.int16, types.int16),
            ]

        registry.register(BF16AsShortTemplate)
        registry.register_global(
            _bfloat16_as_short, types.Function(BF16AsShortTemplate)
        )

    _bfloat16_as_ushort = getattr(mod, "__bfloat16_as_ushort", None)
    if _bfloat16_as_ushort:
        _key = _bfloat16_as_ushort

        class BF16AsUShortTemplate(ConcreteTemplate):
            key = _key
            cases = [
                signature(types.uint16, bf16),
                signature(types.uint16, types.int64),
                signature(types.uint16, types.int32),
                signature(types.uint16, types.int16),
                signature(types.uint16, types.uint64),
                signature(types.uint16, types.uint32),
                signature(types.uint16, types.uint16),
            ]

        registry.register(BF16AsUShortTemplate)
        registry.register_global(
            _bfloat16_as_ushort, types.Function(BF16AsUShortTemplate)
        )

    _short_as_bfloat16 = getattr(mod, "__short_as_bfloat16", None)
    if _short_as_bfloat16:
        _key = _short_as_bfloat16

        class ShortAsBF16Template(ConcreteTemplate):
            key = _key
            cases = [
                signature(bf16, types.int16),
                signature(bf16, types.int64),
                signature(bf16, types.int32),
            ]

        registry.register(ShortAsBF16Template)
        registry.register_global(
            _short_as_bfloat16, types.Function(ShortAsBF16Template)
        )

    _ushort_as_bfloat16 = getattr(mod, "__ushort_as_bfloat16", None)
    if _ushort_as_bfloat16:
        _key = _ushort_as_bfloat16

        class UShortAsBF16Template(ConcreteTemplate):
            key = _key
            cases = [
                signature(bf16, types.uint16),
                signature(bf16, types.int64),
                signature(bf16, types.int32),
                signature(bf16, types.uint64),
                signature(bf16, types.uint32),
            ]

        registry.register(UShortAsBF16Template)
        registry.register_global(
            _ushort_as_bfloat16, types.Function(UShortAsBF16Template)
        )
