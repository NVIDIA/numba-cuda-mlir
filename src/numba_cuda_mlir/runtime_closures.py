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


class _CaptureLayout:
    """Encode model data representations without changing their conversions.

    Python tuples become LLVM aggregates, memrefs become descriptors, and static
    placeholders occupy no field. LLVM model storage is already an aggregate
    representation and remains opaque, including previously packed closures.
    """

    def __init__(self, storage_type):
        self.storage_type = storage_type
        self.children = None
        if storage_type is None or isinstance(storage_type, mlir_ir.NoneType):
            self.data_type = None
        elif isinstance(storage_type, tuple):
            children = []
            data_types = []
            for typ in storage_type:
                child = _CaptureLayout(typ)
                field = None
                if child.data_type is not None:
                    field = len(data_types)
                    data_types.append(child.data_type)
                children.append((child, field))
            self.children = tuple(children)
            self.data_type = llvm.StructType.get_literal(data_types) if data_types else None
        elif isinstance(storage_type, mlir_ir.MemRefType):
            from numba_cuda_mlir.lowering_utilities import memref_descriptor_type

            self.data_type = memref_descriptor_type(storage_type.rank)
        else:
            self.data_type = storage_type

    def pack(self, value):
        if self.children is not None:
            if len(value) != len(self.children):
                raise ValueError("Closure environment does not match its data layout")
            if self.data_type is None:
                return None
            result = llvm.UndefOp(self.data_type).result
            for (child, field), member in zip(self.children, value):
                if field is not None:
                    result = llvm.insertvalue(result, child.pack(member), [field])
            return result
        if self.data_type is None:
            return None
        if isinstance(self.storage_type, mlir_ir.MemRefType):
            return builtin.unrealized_conversion_cast([self.data_type], [value])
        return value

    def unpack(self, value):
        if self.children is not None:
            return tuple(
                child.unpack(
                    llvm.extractvalue(child.data_type, value, [field])
                    if field is not None
                    else None
                )
                for child, field in self.children
            )
        if self.data_type is None:
            return self.storage_type
        if isinstance(self.storage_type, mlir_ir.MemRefType):
            return builtin.unrealized_conversion_cast([self.storage_type], [value])
        return value


class _CaptureCodec:
    """Preserve each model's value/data boundary around environment encoding."""

    def __init__(self, dmm, typ):
        self.model = dmm.lookup(typ)
        self.value_type = self.model.get_value_type()
        self.is_static = isinstance(self.value_type, mlir_ir.NoneType)
        self.layout = _CaptureLayout(
            self.value_type if self.is_static else self.model.get_data_type()
        )
        self.data_type = self.layout.data_type

    def pack(self, builder, value):
        if self.is_static:
            return None
        return self.layout.pack(self.model.as_data(builder, value))

    def unpack(self, builder, value):
        if self.is_static:
            return self.value_type
        return self.model.from_data(builder, self.layout.unpack(value))


@register_model(RuntimeClosure)
class RuntimeClosureModel(StructModel):
    def __init__(self, dmm, fe_type):
        self.codecs = tuple(_CaptureCodec(dmm, t) for t in fe_type.capture_types)
        self.layout = _CaptureLayout(tuple(codec.data_type for codec in self.codecs))
        super().__init__(dmm, fe_type, list(zip(fe_type.capture_names, fe_type.capture_types)))

    def get_value_type(self):
        return self.get_data_type()

    def get_data_type(self):
        if self._data_type is None:
            self._data_type = self.layout.data_type
            if self._data_type is None:
                self._data_type = llvm.StructType.get_literal([])
        return self._data_type

    def as_data(self, builder, value):
        return value

    def from_data(self, builder, value):
        return value

    def pack_captures(self, builder, values):
        if len(values) != len(self.codecs):
            raise ValueError("Closure captures do not match their environment layout")
        packed = self.layout.pack(
            tuple(codec.pack(builder, value) for codec, value in zip(self.codecs, values))
        )
        return llvm.UndefOp(self.get_data_type()).result if packed is None else packed

    def unpack_captures(self, builder, value):
        return tuple(
            codec.unpack(builder, member)
            for codec, member in zip(self.codecs, self.layout.unpack(value))
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


def annotate_closure_captures(func_ir):
    """Prove snapshot safety before any inlining erases shared-cell identity.

    Annotate, rather than reject, so directly inlined closures retain their
    existing shared-cell semantics. Only surviving escaping closures require
    an immutable environment. A single IR definition must dominate construction
    and must not execute in a loop; counting bytecode writes cannot prove that.
    """
    closures = [
        (label, index, stmt.value)
        for label, block in func_ir.blocks.items()
        for index, stmt in enumerate(block.body)
        if isinstance(stmt, ir.Assign)
        and isinstance(stmt.value, ir.Expr)
        and stmt.value.op == "make_function"
        and stmt.value.code.co_freevars
    ]
    if not closures:
        return

    from numba_cuda_mlir.numba_cuda.core.analysis import compute_cfg_from_blocks

    code = func_ir.func_id.func.__code__
    cfg = compute_cfg_from_blocks(func_ir.blocks)
    dominators = cfg.dominators()
    names = {name for _, _, node in closures for name in node.code.co_freevars}
    definitions = {name: [] for name in names}
    for label in cfg.nodes():
        for index, stmt in enumerate(func_ir.blocks[label].body):
            if isinstance(stmt, ir.Assign) and stmt.target.name in definitions:
                definitions[stmt.target.name].append((label, index))

    # Detect writes through sibling/nested closures before local inlining can
    # hide them. These semantic cell operations are inspected, never rewritten.
    def nonlocal_writes(parent, shared):
        written = set()
        for child in parent.co_consts:
            if isinstance(child, pytypes.CodeType):
                child_shared = shared.intersection(child.co_freevars)
                if child_shared:
                    written.update(
                        instruction.argval
                        for instruction in dis.get_instructions(child)
                        if instruction.opname in ("STORE_DEREF", "DELETE_DEREF")
                        and instruction.argval in child_shared
                    )
                    written.update(nonlocal_writes(child, child_shared))
        return written

    mutated = nonlocal_writes(code, names)
    deleted = {
        instruction.argval
        for instruction in dis.get_instructions(code)
        if instruction.opname == "DELETE_DEREF"
    }
    for label, index, node in closures:
        error = None
        for name in node.code.co_freevars:
            if name in deleted:
                error = "Escaping closure captures a deleted shared cell '%s'" % name
            elif name in mutated:
                error = "Escaping closure shares a mutable nonlocal cell '%s'" % name
            elif name in code.co_freevars and not definitions[name]:
                # Host free variables are supplied as compile-time values.
                continue
            elif len(definitions[name]) != 1:
                error = "Escaping closure captures reassigned shared cell '%s'" % name
            else:
                definition_label, definition_index = definitions[name][0]
                if cfg.in_loops(definition_label):
                    error = "Escaping closure captures loop-carried shared cell '%s'" % name
                elif definition_label not in dominators[label] or (
                    definition_label == label and definition_index >= index
                ):
                    error = "Escaping closure captures an uninitialized shared cell '%s'" % name
            if error is not None:
                break
        node._kws["_runtime_capture_error"] = error


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
                if "_runtime_capture_error" not in node._kws:
                    raise errors.UnsupportedError(
                        "Cannot prove escaping closure capture safety", loc=stmt.loc
                    )
                capture_error = node._kws["_runtime_capture_error"]
                if capture_error is not None:
                    raise errors.UnsupportedError(capture_error, loc=stmt.loc)
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
