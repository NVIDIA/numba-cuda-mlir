# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import operator

from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir._mlir.dialects import arith, cf, func, math
from numba_cuda_mlir.lowering_utilities import constant, get_or_insert_function
from numba_cuda_mlir import mlir_lowering
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core import ir as numba_ir
from numba_cuda_mlir.numba_cuda.core.errors import TypingError
from numba_cuda_mlir.tools import generate_mangled_name


class Lower(mlir_lowering.MLIRLower):
    def _initialize_target_and_linker(self):
        """Skip GPU initialization and return no link inputs."""
        self.linker = None
        return ()

    def setup_func_op(self):
        args = [self.get_mlir_type(arg) for arg in self.fndesc.argtypes]
        results = (
            []
            if isinstance(self.fndesc.restype, types.NoneType)
            else [self.get_return_type(self.fndesc.restype)]
        )
        name = generate_mangled_name(self.fndesc.qualname, self.fndesc.argtypes)
        self.metadata["host_symbol_name"] = name
        with ir.InsertionPoint(self.mlir_module.body):
            self.mlir_funcOp = func.FuncOp(
                name, ir.FunctionType.get(args, results), loc=self.mlir_loc
            )
            self.mlir_funcOp.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            self.mlir_funcOp.add_entry_block()

    def lower_block(self, offset, block):
        self._host_active_block = self.blkmap[offset]
        for inst in block.body:
            with ir.InsertionPoint(self._host_active_block):
                self.lower_inst(inst)

    def _fail_if(self, failed, code):
        error_block = self.mlir_funcOp.body.blocks.append()
        continuation = self.mlir_funcOp.body.blocks.append()
        cf.cond_br(failed, [], [], error_block, continuation)
        with ir.InsertionPoint(error_block):
            callback = getattr(self, "_scalar_error_callback", None)
            if callback is None:
                callback = get_or_insert_function(
                    "host_scalar_error",
                    ir.FunctionType.get([ir.IntegerType.get_signless(32)], []),
                    self.mlir_module.operation,
                )
                callback.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
                self._scalar_error_callback = callback
            func.call(
                [],
                callback.name.value,
                [arith.constant(ir.IntegerType.get_signless(32), code)],
            )
            results = self.mlir_funcOp.function_type.value.results
            func.ReturnOp(
                [arith.constant(ty, 0.0 if isinstance(ty, ir.FloatType) else 0) for ty in results]
            )
        self._host_active_block = continuation

    def checked_range_step_context(self, step):
        """Validate the range step and return the continuation insertion context."""
        self._fail_if(arith.cmpi(arith.CmpIPredicate.eq, step, arith.constant(step.type, 0)), 2)
        return ir.InsertionPoint(self._host_active_block)

    def lower_inst(self, inst):
        if not isinstance(
            inst,
            (
                numba_ir.Assign,
                numba_ir.Branch,
                numba_ir.Jump,
                numba_ir.Del,
                numba_ir.Return,
            ),
        ):
            raise TypingError(f"host.jit does not support {type(inst).__name__}: {inst}")
        super().lower_inst(inst)

    def lower_expr_assign(self, target, expr):
        if expr.op not in (
            "binop",
            "inplace_binop",
            "unary",
            "cast",
            "call",
            "getattr",
            "getiter",
            "iternext",
            "pair_first",
            "pair_second",
            "null",
        ):
            raise TypingError(f"host.jit does not support expression {expr.op}")
        super().lower_expr_assign(target, expr)

    def lower_assign(self, inst):
        typ = types.unliteral(self.get_numba_type(inst.target.name))
        if isinstance(typ, (types.Array, types.BaseTuple, types.Complex, types.Optional)):
            raise TypingError(f"host.jit supports scalar values only; got {typ}")
        super().lower_assign(inst)

    def _check_float_to_integer(self, source_type, target_type, value):
        source_type = types.unliteral(source_type)
        target_type = types.unliteral(target_type)
        if not isinstance(source_type, types.Float) or not isinstance(target_type, types.Integer):
            return
        if not isinstance(value, ir.Value):
            value = arith.constant(self.get_mlir_type(source_type), float(value))
        self._fail_if(arith.cmpf(arith.CmpFPredicate.UNO, value, value), 4)
        with ir.InsertionPoint(self._host_active_block):
            # Check the truncated value, so e.g. uint8(-0.5) remains valid.
            truncated = math.trunc(value)
            lower = -(1 << (target_type.bitwidth - 1)) if target_type.signed else 0
            upper = 1 << (target_type.bitwidth - int(target_type.signed))
            overflow = arith.ori(
                arith.cmpf(
                    arith.CmpFPredicate.OLT, truncated, arith.constant(value.type, float(lower))
                ),
                arith.cmpf(
                    arith.CmpFPredicate.OGE, truncated, arith.constant(value.type, float(upper))
                ),
            )
            self._fail_if(overflow, 5)

    def scalar_conversion_context(self):
        return ir.InsertionPoint(self._host_active_block)

    def lower_cast(self, source_type, target_type, value):
        if not isinstance(value, ir.Value) and isinstance(
            types.unliteral(source_type), (types.Number, types.Boolean)
        ):
            value = constant(value, self.get_mlir_type(source_type))
        self._check_float_to_integer(source_type, target_type, value)
        with self.scalar_conversion_context():
            return super().lower_cast(source_type, target_type, value)

    def lower_getattr_assign(self, target, value, attr):
        typ = self.get_numba_type(target.name)
        # Module attributes identifying supported scalar constructors may be
        # resolved by shared lowering; scalar/custom attribute access is not.
        if not isinstance(typ, types.NumberClass):
            raise TypingError(f"host.jit does not support attribute access {attr!r}")
        super().lower_getattr_assign(target, value, attr)

    def lower_call_expr_assign(self, target, fn, args, kws, expr=None):
        value = self.load_var(fn)
        typ = self.get_numba_type(fn.name)
        if value not in (range, bool, int, float, abs) and not isinstance(typ, types.NumberClass):
            raise TypingError(
                f"host.jit does not support calls to {getattr(value, '__qualname__', value)!r}; "
                "only range and scalar constructors/abs are supported"
            )
        if value not in (range, abs) and (kws or len(args) != 1):
            raise TypingError("host.jit scalar constructors require one positional argument")
        super().lower_call_expr_assign(target, fn, args, kws, expr)

    def lower_binop_expr_assign(self, target, fn, lhs, rhs):
        if fn in (
            operator.pow,
            operator.ipow,
            operator.lshift,
            operator.rshift,
            operator.ilshift,
            operator.irshift,
        ):
            raise TypingError(f"host.jit does not support operator {fn.__name__}")
        return super().lower_binop_expr_assign(target, fn, lhs, rhs)

    def checked_division_context(self, operation, typ, left, right):
        """Check the converted operands immediately before division or remainder."""
        zero = arith.constant(right.type, 0.0 if isinstance(right.type, ir.FloatType) else 0)
        failed = (
            arith.cmpf(arith.CmpFPredicate.OEQ, right, zero)
            if isinstance(right.type, ir.FloatType)
            else arith.cmpi(arith.CmpIPredicate.eq, right, zero)
        )
        self._fail_if(failed, 1)
        typ = types.unliteral(typ)
        with ir.InsertionPoint(self._host_active_block):
            if operation is operator.floordiv and isinstance(typ, types.Integer) and typ.signed:
                minimum = arith.constant(left.type, -(1 << (typ.bitwidth - 1)))
                minus_one = arith.constant(right.type, -1)
                overflow = arith.andi(
                    arith.cmpi(arith.CmpIPredicate.eq, left, minimum),
                    arith.cmpi(arith.CmpIPredicate.eq, right, minus_one),
                )
                self._fail_if(overflow, 3)
        return ir.InsertionPoint(self._host_active_block)
