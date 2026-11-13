# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, ParamSpec, Sequence, TypeVar, TypeVarTuple, Unpack
from numba_cuda_mlir.numba_cuda.typing import Signature
from numba_cuda_mlir.numba_cuda.core import ir

# Prototype for intrinsic code generation functions
Builder = Callable[["MLIRLower", ir.Var, list[ir.Var], list[tuple[str, ir.Var]]], None]
ValuesBuilder = Callable[["MLIRLower", Signature, Sequence[Any]], Any]

PS = ParamSpec("PS")
AnyCallable = Callable[PS, Any]
