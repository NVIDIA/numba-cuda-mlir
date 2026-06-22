# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tombstone replacement for the removed ``llvmlite`` dependency.

numba-cuda-mlir compiles via MLIR, not llvmlite. The vendored numba-cuda code
still contains llvmlite-based code paths (the old LLVM codegen, ``@lower``
implementations, and llvmlite-typed data-model construction), but on the MLIR
path these are never executed - their results are discarded. Rather than keep
llvmlite as a dependency for code that never runs, the ``from llvmlite import
ir`` / ``import llvmlite.binding`` imports in those modules are redirected here.

Any attempt to actually *use* llvmlite (build a type, a constant, an IR module,
run a pass) raises ``NotImplementedError`` with a clear message, so that if a
genuinely-live path is ever discovered it fails loudly rather than silently
importing llvmlite. Code that merely needs the module to import (e.g. a
discarded type alias built at import time) is fixed at its definition site to
not call into here.
"""


class _Tombstone:
    """An object that raises on any attribute access or call.

    Stands in for an llvmlite module (``ir``, ``binding``) or symbol
    (``Constant``, ``IRBuilder``, ...). It deliberately provides no working
    behaviour: reaching it means dead llvmlite code was executed on the MLIR
    path.
    """

    __slots__ = ("_qualname",)

    def __init__(self, qualname):
        object.__setattr__(self, "_qualname", qualname)

    def _raise(self, extra=""):
        raise NotImplementedError(
            f"llvmlite has been removed from numba-cuda-mlir; {self._qualname}{extra} "
            "is not available on the MLIR compilation path"
        )

    def __getattr__(self, name):
        self._raise(f".{name}")

    def __call__(self, *args, **kwargs):
        self._raise("()")


ir = _Tombstone("llvmlite.ir")
binding = _Tombstone("llvmlite.binding")


def __getattr__(name):
    # PEP 562: satisfy `from numba_cuda_mlir.numba_cuda._llvmlite_removed import
    # Constant` (and any other previously-llvmlite symbol) with a tombstone.
    return _Tombstone(f"llvmlite.{name}")
