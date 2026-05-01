# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir._mlir.dialects import cf, scf
from numba_cuda_mlir._mlir.dialects.arith import (
    AndIOp,
    CmpIOp,
    CmpIPredicate,
    ConstantOp,
    MulIOp,
    SelectOp,
    ShRUIOp,
)
from numba_cuda_mlir._mlir.ir import InsertionPoint, Location, StringAttr


def ipowi(base, exp, *, loc: Location = None):
    """Integer power via binary exponentiation (exponentiation by squaring)."""
    int_type = base.type
    one = ConstantOp(int_type, 1, loc=loc).result
    zero = ConstantOp(int_type, 0, loc=loc).result

    is_non_negative = CmpIOp(CmpIPredicate.sge, exp, zero, loc=loc).result
    cf.assert_(
        is_non_negative,
        StringAttr.get("negative exponent not supported for integer exponentiation"),
        loc=loc,
    )

    while_op = scf.WhileOp(
        [int_type, int_type, int_type],
        [one, base, exp],
        loc=loc,
    )

    before_block = while_op.before.blocks.append(int_type, int_type, int_type)
    with InsertionPoint(before_block):
        result_arg, base_arg, exp_arg = before_block.arguments
        cond = CmpIOp(CmpIPredicate.sgt, exp_arg, zero, loc=loc).result
        scf.condition(cond, [result_arg, base_arg, exp_arg], loc=loc)

    after_block = while_op.after.blocks.append(int_type, int_type, int_type)
    with InsertionPoint(after_block):
        result_arg, base_arg, exp_arg = after_block.arguments
        exp_and_one = AndIOp(exp_arg, one, loc=loc).result
        is_odd = CmpIOp(CmpIPredicate.ne, exp_and_one, zero, loc=loc).result
        new_result = SelectOp(
            is_odd, MulIOp(result_arg, base_arg, loc=loc).result, result_arg, loc=loc
        ).result
        new_base = MulIOp(base_arg, base_arg, loc=loc).result
        new_exp = ShRUIOp(exp_arg, one, loc=loc).result
        scf.YieldOp([new_result, new_base, new_exp], loc=loc)

    return while_op.results[0]
