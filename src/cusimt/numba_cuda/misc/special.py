# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause


class prange:
    """Provides a 1D parallel iterator that generates a sequence of integers.
    In non-parallel contexts, prange is identical to range.
    """

    def __new__(cls, *args):
        return range(*args)


def literally(obj):
    """Forces Numba to interpret *obj* as an Literal value.

    *obj* must be either a literal or an argument of the caller function, where
    the argument must be bound to a literal. The literal requirement
    propagates up the call stack.

    This function is intercepted by the compiler to alter the compilation
    behavior to wrap the corresponding function parameters as ``Literal``.
    It has **no effect** outside of nopython-mode (interpreter, and objectmode).

    The current implementation detects literal arguments in two ways:

    1. Scans for uses of ``literally`` via a compiler pass.
    2. ``literally`` is overloaded to raise ``numba.errors.ForceLiteralArg``
       to signal the dispatcher to treat the corresponding parameter
       differently. This mode is to support indirect use (via a function call).

    The execution semantic of this function is equivalent to an identity
    function.

    See :ghfile:`numba/tests/test_literal_dispatch.py` for examples.
    """
    return obj


def literal_unroll(container):
    return container


__all__ = [
    "prange",
    "literally",
    "literal_unroll",
]
