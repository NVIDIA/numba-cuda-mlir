# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Callable, TypeVar, TypeVarTuple, Unpack, ParamSpec
from cusimt.numba_cuda.core import ir

# Prototype for intrinsic code generation functions
Builder = Callable[["MLIRLower", ir.Var, list[ir.Var], list[tuple[str, ir.Var]]], None]

PS = ParamSpec("PS")
AnyCallable = Callable[PS, Any]
