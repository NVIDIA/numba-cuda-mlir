# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for extending type inference (``extending.TypeInferExtension``).

A toy payload type stands in for an extension: a per-thread array whose dtype
comes from indexed writes (weak evidence) or from a producer method (strong
evidence, through ``Signature.recvr``).
"""

import contextlib
import operator

import numpy as np
import pytest

from numba_cuda_mlir import cuda, extending, types
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import arith, gpu, memref
from numba_cuda_mlir.extending import lowering_registry, typing_registry
from numba_cuda_mlir.lowering_utilities import index_of
from numba_cuda_mlir.models import ArrayModel, register_model
from numba_cuda_mlir.numba_cuda.core import errors
from numba_cuda_mlir.numba_cuda.types import undefined, unliteral
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AttributeTemplate,
    bound_function,
    signature,
)


class ToyPayload(types.Array):
    def __init__(self, dtype, strong, items):
        self.strong = bool(strong)
        self.items = int(items)
        kind = "strong" if self.strong else "weak"
        super().__init__(dtype, 1, "C", name=f"ToyPayload({dtype}, {kind}, {self.items})")

    @property
    def key(self):
        return self.dtype, self.strong, self.items

    def copy(self, dtype=None, ndim=None, layout=None, readonly=None):
        if ndim in (None, 1) and layout in (None, "C") and not readonly:
            return ToyPayload(self.dtype if dtype is None else dtype, self.strong, self.items)
        return super().copy(dtype=dtype, ndim=ndim, layout=layout, readonly=readonly)

    def unify(self, typingctx, other):
        if not isinstance(other, ToyPayload) or other.items != self.items:
            return None
        if not other.dtype.is_precise():
            return self
        if not self.dtype.is_precise():
            return other
        if self.dtype == other.dtype or self.strong != other.strong:
            return self if self.strong else other
        return None


register_model(ToyPayload)(ArrayModel)


def toy_payload(items):
    raise NotImplementedError("kernel only")


@extending.type_callable(toy_payload)
def _type_toy_payload(context):
    def typer(items):
        if isinstance(items, types.IntegerLiteral):
            return ToyPayload(undefined, False, items.literal_value)

    return typer


@typing_registry.register_attr
class _ToyPayloadAttributes(AttributeTemplate):
    key = ToyPayload

    @bound_function("toy_payload.load")
    def resolve_load(self, p, args, kws):
        if kws or len(args) != 1 or not isinstance(args[0], types.Array):
            return None
        if not args[0].dtype.is_precise():
            return None
        return signature(types.none, args[0], recvr=ToyPayload(args[0].dtype, True, p.items))

    @bound_function("toy_payload.fill_from")
    def resolve_fill_from(self, p, args, kws):
        if kws or len(args) != 1 or not isinstance(args[0], types.Number):
            return None
        return signature(types.none, args[0], recvr=ToyPayload(args[0], True, p.items))


@lowering_registry.lower(toy_payload, types.IntegerLiteral)
def _lower_toy_payload(builder, target, args, kwargs):
    p = builder.get_numba_type(target.name)
    with builder.alloca_insertion_point():
        storage = builder.get_storage_type(p.dtype)
        mr = memref.alloca(mlir_ir.MemRefType.get([p.items], storage), [], [])
    builder.store_var(target, mr)


@lowering_registry.lower("toy_payload.load", ToyPayload, types.Array)
def _lower_load(builder, target, args, kwargs):
    p = builder.get_numba_type(args[0].name)
    p_mr, src_mr = builder.load_vars(args)
    base = arith.muli(gpu.thread_id(gpu.Dimension.x), index_of(p.items))
    for i in range(p.items):
        memref.store(memref.load(src_mr, [arith.addi(base, index_of(i))]), p_mr, [index_of(i)])


@lowering_registry.lower("toy_payload.fill_from", ToyPayload, types.Number)
def _lower_fill_from(builder, target, args, kwargs):
    p = builder.get_numba_type(args[0].name)
    p_mr, value = builder.load_vars(args)
    for i in range(p.items):
        memref.store(value, p_mr, [index_of(i)])


extending.refresh_registries()


def _is_weak_read(constraint, args):
    return (
        constraint.func is operator.getitem
        and args
        and isinstance(args[0], ToyPayload)
        and not args[0].strong
    )


class WaitForStrong(extending.TypeInferExtension):
    """A read of a payload with only weak evidence is deferred."""

    def __init__(self):
        self.deferred = []

    def defer_call(self, typeinfer, constraint, fnty, args, kws, released):
        if not _is_weak_read(constraint, args):
            return False
        self.deferred.append((constraint.loc.line, released))
        return not released


class PayloadRules(WaitForStrong):
    """WaitForStrong, plus release in order and two end-of-inference checks."""

    def __init__(self, release_in_order=True):
        super().__init__()
        self.release_in_order = release_in_order
        self.released_with = {}

    def defer_call(self, typeinfer, constraint, fnty, args, kws, released):
        if released and _is_weak_read(constraint, args):
            self.released_with.setdefault(constraint, args[0])
        return super().defer_call(typeinfer, constraint, fnty, args, kws, released)

    def at_fixpoint(self, typeinfer, final):
        if not final:
            if self.release_in_order and typeinfer.deferred_calls:
                pending = _pending_producers(typeinfer)
                deferred = sorted(typeinfer.deferred_calls, key=lambda c: c.loc.line)
                ready = [c for c in deferred if c.args[0].name not in pending]
                assert typeinfer.release((ready or deferred)[0])
            return None
        found = []
        from numba_cuda_mlir.numba_cuda.core import typeinfer as ti

        written = {}
        for c in typeinfer.constraints.constraints:
            if isinstance(c, (ti.SetItemConstraint, ti.StaticSetItemConstraint)):
                target = typeinfer.typevars[c.target.name].type
                value = typeinfer.typevars[c.value.name].type
                if isinstance(target, ToyPayload) and not target.strong and value is not None:
                    written.setdefault(c.target.name, set()).add(unliteral(value))
        for name, tys in written.items():
            if len(tys) > 1:
                found.append(errors.TypingError(f"writes to {name} disagree"))
        for constraint, seen in self.released_with.items():
            if typeinfer.typevars[constraint.args[0].name].type != seen:
                found.append(errors.TypingError(f"stale read at line {constraint.loc.line}"))
        return found


def _pending_producers(typeinfer):
    from numba_cuda_mlir.numba_cuda.core import ir

    methods = {}
    for block in typeinfer.func_ir.blocks.values():
        for inst in block.find_insts(ir.Assign):
            expr = inst.value
            if (
                isinstance(expr, ir.Expr)
                and expr.op == "getattr"
                and expr.attr in ("load", "fill_from")
            ):
                methods[inst.target.name] = expr.value.name
    return {
        methods[c.func]
        for c in typeinfer.constraints.constraints
        if getattr(c, "func", None) in methods and c.signature is None
    }


@contextlib.contextmanager
def registered(extension):
    extending.register_typeinfer_extension(extension)
    try:
        yield extension
    finally:
        extending.unregister_typeinfer_extension(extension)


def registered_extensions():
    from numba_cuda_mlir.numba_cuda.core import typeinfer

    return typeinfer._registered_extensions


def _typemap(kernel):
    return next(iter(kernel.overloads.values())).type_annotation.typemap


def _mlir(kernel):
    return next(iter(kernel.inspect_mlir().values()))


def stale_read(src, out):
    p = toy_payload(1)
    p[0] = 0.1
    observed = p[0]
    p.load(src)
    out[0] = observed + p[0]


def writes_disagree(out):
    p = toy_payload(2)
    p[0] = 1
    p[1] = 2.5
    out[0] = p[0] + p[1]


def weak_only(out):
    p = toy_payload(1)
    p[0] = 1.5
    out[0] = p[0]


def two_payloads(out):
    a = toy_payload(1)
    a[0] = 1.5
    b = toy_payload(1)
    b[0] = 1
    y = b[0]
    x = a[0]
    b.fill_from(x)
    out[0] = y + x


def _run_stale_read():
    k = cuda.jit(stale_read)
    src = np.ones(1, np.float32)
    out = np.zeros(1, np.float64)
    k[1, 1](src, out)
    return k, out[0]


F32_SUM = float(np.float32(0.1) + np.float32(1.0))
F64_SUM = float(np.float64(np.float32(0.1)) + 1.0)


def test_register_and_unregister():
    extension = WaitForStrong()
    with pytest.raises(TypeError):
        extending.register_typeinfer_extension(object())
    with pytest.raises(KeyError):
        extending.unregister_typeinfer_extension(extension)
    extending.register_typeinfer_extension(extension)
    extending.register_typeinfer_extension(extension)
    extending.unregister_typeinfer_extension(extension)
    with pytest.raises(KeyError):
        extending.unregister_typeinfer_extension(extension)


def test_without_extensions_a_read_keeps_weak_evidence():
    k, value = _run_stale_read()
    assert str(_typemap(k)["observed"]) == "float64"
    assert value == F64_SUM


def test_defer_call_makes_a_read_wait_for_strong_evidence():
    with registered(WaitForStrong()) as extension:
        k, value = _run_stale_read()
    assert str(_typemap(k)["observed"]) == "float32"
    assert value == F32_SUM
    assert extension.deferred and not any(released for _, released in extension.deferred)


def test_deferred_calls_are_released_when_nothing_else_moves():
    with registered(WaitForStrong()) as extension:
        k = cuda.jit(weak_only)
        out = np.zeros(1)
        k[1, 1](out)
    assert out[0] == 1.5
    lines = [line for line, released in extension.deferred if released]
    assert lines, "the deferred read was never released"


def test_final_at_fixpoint_reports_errors():
    k = cuda.jit(writes_disagree)
    out = np.zeros(1)
    k[1, 1](out)
    assert out[0] == 3.0  # the second write is truncated without the extension

    with registered(PayloadRules()):
        k = cuda.jit(writes_disagree)
        with pytest.raises(errors.TypingError, match="writes to p disagree"):
            k[1, 1](out)


def test_the_extension_chooses_the_release_order():
    with registered(PayloadRules()):
        k = cuda.jit(two_payloads)
        out = np.zeros(1)
        k[1, 1](out)
    assert out[0] == 2.5
    assert str(_typemap(k)["y"]) == "float64"

    # Releasing every deferred call at once types the read of b before b's
    # producer, which the extension's own check then refuses.
    with registered(PayloadRules(release_in_order=False)):
        k = cuda.jit(two_payloads)
        with pytest.raises(errors.TypingError, match="stale read"):
            k[1, 1](np.zeros(1))


def test_unregistering_restores_typing_and_mlir():
    before, _ = _run_stale_read()
    with registered(WaitForStrong()):
        extended, _ = _run_stale_read()
    after, value = _run_stale_read()
    assert _mlir(extended) != _mlir(before)
    assert _mlir(after) == _mlir(before)
    assert str(_typemap(after)["observed"]) == "float64"
    assert value == F64_SUM


def test_a_noop_extension_changes_nothing():
    before, _ = _run_stale_read()

    class NoOp(extending.TypeInferExtension):
        calls = 0

        def defer_call(self, typeinfer, constraint, fnty, args, kws, released):
            NoOp.calls += 1
            return False

        def at_fixpoint(self, typeinfer, final):
            NoOp.calls += 1

    with registered(NoOp()):
        extended, _ = _run_stale_read()
    assert NoOp.calls
    assert _mlir(extended) == _mlir(before)


def test_extensions_are_read_when_an_inference_starts():
    late = WaitForStrong()

    class RegistersLate(extending.TypeInferExtension):
        def defer_call(self, typeinfer, constraint, fnty, args, kws, released):
            if not any(e is late for e in registered_extensions()):
                extending.register_typeinfer_extension(late)
            return False

    with registered(RegistersLate()):
        try:
            k, _ = _run_stale_read()
            assert not late.deferred  # registered mid-inference: not consulted
            assert str(_typemap(k)["observed"]) == "float64"
            k, _ = _run_stale_read()
            assert late.deferred  # the next inference sees it
            assert str(_typemap(k)["observed"]) == "float32"
        finally:
            extending.unregister_typeinfer_extension(late)


def test_final_at_fixpoint_must_not_change_types():
    class Changes(extending.TypeInferExtension):
        def at_fixpoint(self, typeinfer, final):
            if final:
                typeinfer.add_type("observed", types.complex128, loc=None)

    with registered(Changes()):
        with pytest.raises(RuntimeError, match="final at_fixpoint"):
            _run_stale_read()


def test_extensions_run_in_device_function_bodies():
    @cuda.jit(device=True)
    def helper(src):
        p = toy_payload(1)
        p[0] = 0.1
        observed = p[0]
        p.load(src)
        return observed + p[0]

    def kernel(src, out):
        out[0] = helper(src)

    with registered(WaitForStrong()) as extension:
        k = cuda.jit(kernel)
        src = np.ones(1, np.float32)
        out = np.zeros(1)
        k[1, 1](src, out)
    assert extension.deferred
    assert out[0] == F32_SUM
