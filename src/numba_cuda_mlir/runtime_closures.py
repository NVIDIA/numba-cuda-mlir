# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only escaping device closures with runtime aggregate environments."""

import dis
import inspect
import types as pytypes
from dataclasses import dataclass
from functools import partial

from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import builtin, llvm
from numba_cuda_mlir.models import StructModel, register_model
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.core import errors, ir, postproc
from numba_cuda_mlir.numba_cuda.core.compiler_machinery import FunctionPass, register_pass
from numba_cuda_mlir.numba_cuda.core.ir_utils import build_definitions


@dataclass(frozen=True)
class CallableSpecialization:
    signature: object
    body_signature: object
    body_function: object
    capture_names: tuple
    capture_types: tuple
    code_identity: object


def _make_closure_body(code, globals_dict):
    """Expose a capture-first signature without changing the closure's code.

    The Python callable remains executable for host-side inspection. Compilation
    uses ``_closure_body_ir`` instead of translating this adapter's implementation.
    """
    if code.co_cellvars:
        raise errors.UnsupportedError(
            "Escaping closure bodies containing nested shared cells are unsupported"
        )
    if any(i.opname in ("STORE_DEREF", "DELETE_DEREF") for i in dis.get_instructions(code)):
        raise errors.UnsupportedError("Escaping closures do not support nonlocal mutation")

    function_type = pytypes.FunctionType
    cell_type = pytypes.CellType
    original = function_type(
        code, globals_dict, closure=tuple(cell_type(None) for _ in code.co_freevars)
    )
    original_signature = inspect.signature(original)
    capture_parameters = []
    used_names = set(code.co_varnames)
    for name in code.co_freevars:
        parameter_name = "__capture_" + name
        while parameter_name in used_names:
            parameter_name += "_"
        used_names.add(parameter_name)
        capture_parameters.append(
            inspect.Parameter(parameter_name, inspect.Parameter.POSITIONAL_ONLY)
        )
    signature = original_signature.replace(
        parameters=capture_parameters + list(original_signature.parameters.values())
    )
    capture_count = len(capture_parameters)

    def body(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        function = function_type(
            code,
            globals_dict,
            closure=tuple(cell_type(value) for value in bound.args[:capture_count]),
        )
        return function(*bound.args[capture_count:], **bound.kwargs)

    # Retain the original globals identity used by callable specialization keys.
    body = function_type(body.__code__, globals_dict, closure=body.__closure__)
    body.__name__ = "__runtime_closure_%x" % id(code)
    body.__qualname__ = "%s.%s" % (code.co_qualname, body.__name__)
    body.__signature__ = signature
    body._numba_cuda_mlir_closure = original
    body._numba_cuda_mlir_ir = partial(_closure_body_ir, body)
    return body


def _closure_body_ir(body):
    """Convert free-variable reads to leading arguments in fresh untyped IR."""
    from numba_cuda_mlir.numba_cuda.compiler import run_frontend
    from numba_cuda_mlir.numba_cuda.core.bytecode import FunctionIdentity

    original = body._numba_cuda_mlir_closure
    func_ir = run_frontend(original)
    capture_names = original.__code__.co_freevars
    count = len(capture_names)
    parameter_names = tuple(inspect.signature(body).parameters)[:count]
    entry = func_ir.blocks[min(func_ir.blocks)]
    captures = {}
    arguments = []
    for index, (name, parameter) in enumerate(zip(capture_names, parameter_names)):
        variable = entry.scope.define(parameter, entry.loc)
        captures[name] = variable
        arguments.append(ir.Assign(ir.Arg(parameter, index, entry.loc), variable, entry.loc))
    for block in func_ir.blocks.values():
        for stmt in block.body:
            if isinstance(stmt, ir.Assign):
                if isinstance(stmt.value, ir.Arg):
                    stmt.value = ir.Arg(stmt.value.name, stmt.value.index + count, stmt.value.loc)
                elif isinstance(stmt.value, ir.FreeVar):
                    stmt.value = captures[stmt.value.name]
    entry.body[:0] = arguments
    func_ir.arg_count += count
    func_ir.arg_names = parameter_names + tuple(func_ir.arg_names)
    func_ir.func_id = FunctionIdentity.from_function(body)
    func_ir._definitions = build_definitions(func_ir.blocks)
    postproc.PostProcessor(func_ir).run()
    return func_ir


class RuntimeClosure(types.Callable):
    """Static code identity and typed runtime fields; never a function pointer."""

    def __init__(self, body_function, original_code, capture_types):
        self.body_function = body_function
        self.original_code = original_code
        self.capture_names = original_code.co_freevars
        self.capture_types = tuple(types.unliteral(t) for t in capture_types)
        self.code_identity = (original_code, id(body_function.__globals__))
        from numba_cuda_mlir import cuda

        self.body_dispatcher = cuda.jit(device=True)(body_function)
        super().__init__("RuntimeClosure(%s, %s)" % (original_code.co_qualname, self.capture_types))

    @property
    def key(self):
        return self.code_identity, self.capture_types

    @property
    def mangling_args(self):
        return "RuntimeClosure", (
            id(self.original_code),
            id(self.body_function.__globals__),
            *self.capture_types,
        )

    def specialize(self, context, argument_types, kws=None):
        body_sig = types.Dispatcher(self.body_dispatcher).get_call_type(
            context, self.capture_types + tuple(argument_types), kws or {}
        )
        if body_sig is None:
            return None
        from numba_cuda_mlir.numba_cuda.typing.templates import signature

        user_sig = signature(body_sig.return_type, *body_sig.args[len(self.capture_types) :])
        return CallableSpecialization(
            user_sig,
            body_sig,
            self.body_function,
            self.capture_names,
            self.capture_types,
            self.code_identity,
        )

    def get_call_type(self, context, args, kws):
        specialization = self.specialize(context, args, kws)
        return specialization.signature if specialization is not None else None

    def get_call_signatures(self):
        return [], True

    def get_impl_key(self, sig):
        return self


def get_callable_specialization(callable_type, argument_types, typing_context=None):
    """Resolve code and ABI field metadata without capturing runtime values.

    ``body_signature`` has capture arguments first, followed by user arguments.
    Its body can be compiled with ``compile(..., abi='c', output='ltoir')``.
    Ordinary device dispatchers have an empty capture list.
    Return ``None`` when the callable cannot specialize for these arguments.
    """
    if typing_context is None:
        from numba_cuda_mlir.descriptor import mlir_target

        typing_context = mlir_target.typing_context
    if isinstance(callable_type, RuntimeClosure):
        return callable_type.specialize(typing_context, argument_types)
    if isinstance(callable_type, types.Dispatcher):
        sig = callable_type.get_call_type(typing_context, tuple(argument_types), {})
        if sig is None:
            return None
        function = callable_type.dispatcher.py_func
        return CallableSpecialization(
            sig, sig, function, (), (), (function.__code__, id(function.__globals__))
        )
    raise errors.TypingError(
        "Expected a runtime closure or device function, got %s" % callable_type
    )


class _CaptureCodec:
    """Keep memrefs out of LLVM structs, preserving extension model storage."""

    def __init__(self, dmm, typ):
        self.model = dmm.lookup(typ)
        self.value_type = self.model.get_value_type()
        self.children = None
        self.is_static = isinstance(self.value_type, mlir_ir.NoneType)
        if self.is_static:
            self.data_type = mlir_ir.IntegerType.get_signless(8)
        elif isinstance(self.value_type, mlir_ir.MemRefType):
            from numba_cuda_mlir.lowering_utilities import memref_descriptor_type

            self.data_type = memref_descriptor_type(self.value_type.rank)
        elif isinstance(self.model, StructModel):
            self.children = tuple(_CaptureCodec(dmm, t) for t in self.model._members)
            self.data_type = llvm.StructType.get_literal([c.data_type for c in self.children])
        else:
            self.data_type = self.model.get_data_type()

    def pack(self, builder, value):
        if self.is_static:
            return llvm.mlir_undef(self.data_type)
        if isinstance(self.value_type, mlir_ir.MemRefType):
            return builtin.unrealized_conversion_cast([self.data_type], [value])
        if self.children is not None:
            result = llvm.UndefOp(self.data_type).result
            for i, child in enumerate(self.children):
                field = llvm.extractvalue(child.value_type, value, [i])
                result = llvm.insertvalue(result, child.pack(builder, field), [i])
            return result
        return self.model.as_data(builder, value)

    def unpack(self, builder, value):
        if self.is_static:
            return self.value_type
        if isinstance(self.value_type, mlir_ir.MemRefType):
            return builtin.unrealized_conversion_cast([self.value_type], [value])
        if self.children is not None:
            result = llvm.UndefOp(self.value_type).result
            for i, child in enumerate(self.children):
                field = llvm.extractvalue(child.data_type, value, [i])
                result = llvm.insertvalue(result, child.unpack(builder, field), [i])
            return result
        return self.model.from_data(builder, value)


@register_model(RuntimeClosure)
class RuntimeClosureModel(StructModel):
    def __init__(self, dmm, fe_type):
        self.codecs = tuple(_CaptureCodec(dmm, t) for t in fe_type.capture_types)
        super().__init__(dmm, fe_type, list(zip(fe_type.capture_names, fe_type.capture_types)))

    def get_value_type(self):
        return self.get_data_type()

    def get_data_type(self):
        if self._data_type is None:
            self._data_type = llvm.StructType.get_literal([c.data_type for c in self.codecs])
        return self._data_type

    def as_data(self, builder, value):
        return value

    def from_data(self, builder, value):
        return value

    def pack_captures(self, builder, values):
        result = llvm.UndefOp(self.get_data_type()).result
        for i, (codec, value) in enumerate(zip(self.codecs, values)):
            result = llvm.insertvalue(result, codec.pack(builder, value), [i])
        return result

    def unpack_captures(self, builder, value):
        return tuple(
            codec.unpack(builder, llvm.extractvalue(codec.data_type, value, [i]))
            for i, codec in enumerate(self.codecs)
        )


def _closure_constructor(body_function, original_code):
    from numba_cuda_mlir.extending import lower_builtin, refresh_registries, type_callable

    def construct(*captures):
        raise RuntimeError("Closure construction is only supported in device code")

    @type_callable(construct)
    def type_construct(context):
        def typer(*capture_types):
            return RuntimeClosure(body_function, original_code, capture_types)

        return typer

    @lower_builtin(construct, types.VarArg(types.Any))
    def lower_construct(builder, target, args, kwargs):
        typ = builder.get_numba_type(target.name)
        values = [builder.load_var(arg) for arg in args]
        model = builder.context.data_model_manager.lookup(typ)
        builder.store_var(target, model.pack_captures(builder, values))

    refresh_registries(typing=False, target=True)
    return construct


@register_pass(mutates_CFG=True, analysis_only=False)
class ConvertRuntimeClosures(FunctionPass):
    _name = "convert_runtime_closures"

    def __init__(self):
        super().__init__()

    def run_pass(self, state):
        func_ir = state.func_ir
        changed = False
        for block in func_ir.blocks.values():
            body = []
            for stmt in block.body:
                if not (
                    isinstance(stmt, ir.Assign)
                    and isinstance(stmt.value, ir.Expr)
                    and stmt.value.op == "make_function"
                ):
                    body.append(stmt)
                    continue
                node = stmt.value
                if not node.code.co_freevars:
                    body.append(stmt)
                    continue
                if node.defaults is not None:
                    defaults = func_ir.get_definition(node.defaults)
                    if not (isinstance(defaults, ir.Const) and defaults.value is None):
                        raise errors.UnsupportedError(
                            "Escaping closures with default arguments are unsupported", loc=stmt.loc
                        )
                captures = []
                if node.closure is not None:
                    closure = func_ir.get_definition(node.closure)
                    if not (isinstance(closure, ir.Expr) and closure.op == "build_tuple"):
                        raise errors.UnsupportedError(
                            "Cannot resolve escaping closure environment", loc=stmt.loc
                        )
                    captures = list(closure.items)
                # Reject reassignment of this closure's cells, not other locals.
                outer_code = func_ir.func_id.func.__code__
                instructions = list(dis.get_instructions(outer_code))
                for instruction in instructions:
                    if (
                        instruction.opname == "DELETE_DEREF"
                        and instruction.argval in node.code.co_freevars
                    ):
                        raise errors.UnsupportedError(
                            "Escaping closure captures a deleted shared cell", loc=stmt.loc
                        )
                    if (
                        instruction.opname == "LOAD_CLOSURE"
                        and instruction.argval in node.code.co_freevars
                    ):
                        # A sibling closure can mutate the same cell even when
                        # the enclosing function itself never reassigns it.
                        for sibling in outer_code.co_consts:
                            if (
                                isinstance(sibling, pytypes.CodeType)
                                and instruction.argval in sibling.co_freevars
                            ):
                                if any(
                                    i.opname in ("STORE_DEREF", "DELETE_DEREF")
                                    and i.argval == instruction.argval
                                    for i in dis.get_instructions(sibling)
                                ):
                                    raise errors.UnsupportedError(
                                        "Escaping closure shares a mutable nonlocal cell '%s'"
                                        % instruction.argval,
                                        loc=stmt.loc,
                                    )
                for name in node.code.co_freevars:
                    writes = [
                        i for i in instructions if i.opname == "STORE_DEREF" and i.argval == name
                    ]
                    initial_writes = (
                        0
                        if name
                        in outer_code.co_varnames[
                            : outer_code.co_argcount + outer_code.co_kwonlyargcount
                        ]
                        else 1
                    )
                    if len(writes) > initial_writes:
                        raise errors.UnsupportedError(
                            "Escaping closure captures reassigned shared cell '%s'" % name,
                            loc=stmt.loc,
                        )
                function = _make_closure_body(node.code, func_ir.func_id.func.__globals__)
                constructor = _closure_constructor(function, node.code)
                from numba_cuda_mlir.numba_cuda.core.ir_utils import mk_unique_var

                var = ir.Var(block.scope, mk_unique_var("$runtime_closure_ctor"), stmt.loc)
                body.append(ir.Assign(ir.Global(var.name, constructor, stmt.loc), var, stmt.loc))
                stmt.value = ir.Expr.call(var, captures, (), stmt.loc)
                body.append(stmt)
                changed = True
            block.body = body
        if changed:
            func_ir._definitions = build_definitions(func_ir.blocks)
            from numba_cuda_mlir.numba_cuda.core.ir_utils import dead_code_elimination

            dead_code_elimination(func_ir)
            postproc.PostProcessor(func_ir).run()
        return changed
