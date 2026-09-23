# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from typing import Any, Callable, ParamSpec, TypeVar

if sys.version_info >= (3, 11):
    from typing import TypeVarTuple, Unpack
else:
    from typing_extensions import TypeVarTuple, Unpack
from numba_cuda_mlir.numba_cuda.core import ir

# Prototype for intrinsic code generation functions
Builder = Callable[["MLIRLower", ir.Var, list[ir.Var], list[tuple[str, ir.Var]]], None]

PS = ParamSpec("PS")
AnyCallable = Callable[PS, Any]
