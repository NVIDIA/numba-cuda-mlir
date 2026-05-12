# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from typing import TypeVar
from numba_cuda_mlir._mlir.dialects import math
from numba_cuda_mlir.mlir.dialect_exts import func, scf, arith
from numba_cuda_mlir.mlir.context import mlir_mod_ctx
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.mlir.dialect_exts.scf import (
    if_ctx_manager as if_,
    else_ctx_manager as else_,
)
from numba_cuda_mlir._mlir.ir import UnitAttr
from numba_cuda_mlir.lowering_utilities import convert

FTy = TypeVar("FTy")
ITy = TypeVar("ITy")


def fpowi32(base, exponent):
    exponent = convert(exponent, T.i32())
    return math.fpowi(base, exponent)


def round_ties_to_even(x):
    # Match Numba-CUDA's use of llrint/round-to-even semantics for halfway cases.
    # MLIR math.round is ties-away-from-zero; math.roundeven is banker’s rounding.
    return math.roundeven(x)


@func.func(sym_visibility="private", generics=[FTy, ITy])
def round_ndigits(x: FTy, ndigits: ITy):
    c10 = arith.constant(10.0, x.type)
    c1 = arith.constant(1.0, x.type)
    c1e22 = arith.constant(1e22, x.type)
    c0_i = arith.constant(0, ITy)
    with if_(math.isinf(x) or math.isnan(x), results=[FTy]) as result_if_op:
        scf.yield_(x)
    with else_(result_if_op):
        with if_(ndigits >= 0, results=[FTy]) as if_digits_ge_0:
            ndigits_gt_22 = ndigits > 22
            pow1 = arith.select(ndigits_gt_22, fpowi32(c10, ndigits - 22), fpowi32(c10, ndigits))
            pow2 = arith.select(ndigits_gt_22, c1e22, c1)
            y = (x * pow1) * pow2
            with if_(math.isinf(y), results=[FTy]) as if_isinf_y:
                scf.yield_(x)
            with else_(if_isinf_y):
                scf.yield_((round_ties_to_even(y) / pow2) / pow1)
            scf.yield_(if_isinf_y.result)
        with else_(if_digits_ge_0):
            pow1 = fpowi32(c10, arith.subi(c0_i, ndigits))
            y = x / pow1
            scf.yield_(round_ties_to_even(y) * pow1)

        scf.yield_(if_digits_ge_0.result)
    return result_if_op.result


def get_round_intrinsics_module():
    with mlir_mod_ctx() as ctx:
        for fty in [T.f32(), T.f64()]:
            for ity in [T.i32(), T.i64()]:
                f = round_ndigits[fty, ity]
                f.func_attrs["alwaysinline"] = UnitAttr.get()
                f.emit()
    return str(ctx.module)
