# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir import compiler, cuda, extending, types, testing
from numba_cuda_mlir.models import PrimitiveModel, register_model
from numba_cuda_mlir.numba_cuda.extending import overload as numba_cuda_overload
from numba_cuda_mlir.numba_cuda.extending import typeof_impl
from numba_cuda_mlir.numba_cuda import serialize
from numba_cuda_mlir.numba_cuda.typing.templates import AbstractTemplate, ConcreteTemplate
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof
from numba_cuda_mlir.numba_cuda.core.errors import TypingError
from numba_cuda_mlir.numba_cuda.types import StarArgTuple
import inspect
import numpy as np
import pytest


def test_extending_intrinsic():
    def codegen(builder, signature, args):
        from numba_cuda_mlir._mlir.dialects import cf, arith
        from numba_cuda_mlir._mlir.extras import types as T

        true = arith.constant(result=T.bool(), value=1)
        cf.assert_(true, "This should not be executed")
        return args[0]

    @extending.intrinsic(api="values")
    def do_nothing(typingctx, x):
        return x(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(x):
        x[0] = do_nothing(x[0])

    x = np.array([1])

    k[1, 1](x)
    assert x[0] == 1

    mlir = next(iter(k.inspect_mlir().values()))
    # CHECK: %true = arith.constant true
    # CHECK-NEXT: cf.assert %true, "This should not be executed"
    testing.filecheck_with_comments(mlir)


def test_extending_intrinsic_with_array_argument():
    @extending.intrinsic(api="values")
    def identity(typingctx, arr):
        def codegen(builder, signature, args):
            return args[0]

        return arr(arr), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(x, out):
        out[0] = identity(x)[0]

    x = np.array([7])
    out = np.zeros(1, dtype=x.dtype)
    k[1, 1](x, out)
    assert out[0] == 7


def test_intrinsic_serialization_preserves_values_api_and_literal_dispatch():
    @extending.intrinsic(api="values")
    def identity(typingctx, value):
        def codegen(builder, signature, args):
            return args[0]

        return value(value), codegen

    @extending.intrinsic(api="values", prefer_literal=True)
    def lit_identity(typingctx, value):
        if not isinstance(value, types.IntegerLiteral):
            return None

        def codegen(builder, signature, args):
            from numba_cuda_mlir._mlir.dialects import arith
            from numba_cuda_mlir._mlir.extras import types as T

            return arith.constant(T.i64(), value.literal_value + 1)

        return types.int64(value), codegen

    from numba_cuda_mlir.extending import _Intrinsic

    payload = serialize.dumps(identity)
    payload_lit = serialize.dumps(lit_identity)
    del _Intrinsic._memo[identity._uuid]
    del _Intrinsic._memo[lit_identity._uuid]
    rebuilt = serialize.loads(payload)
    rebuilt_lit = serialize.loads(payload_lit)

    # The round-tripped instance is registered and usable in a kernel.
    extending.refresh_registries()

    @cuda.jit
    def k(x, out):
        out[0] = rebuilt(x)[0]
        out[1] = rebuilt_lit(41)

    x = np.array([7])
    out = np.zeros(2, dtype=x.dtype)
    k[1, 1](x, out)
    np.testing.assert_array_equal(out, [7, 42])


@pytest.mark.parametrize("lowered_for", [None, types.Float], ids=["unlowered", "other_signature"])
def test_overload_on_vendored_intrinsic_gives_way_to_native_typing(lowered_for):
    """An intrinsic whose codegen is vendored (numba-cuda, llvmlite convention)
    and that has no MLIR lowering for the call's signature fails typing, so an
    overload built on it gives way to other templates instead of reaching
    lowering (issue #327)."""
    from numba_cuda_mlir._mlir.dialects import arith
    from numba_cuda_mlir.extending import lowering_registry

    def vendored_codegen(context, builder, sig, args):
        raise AssertionError("vendored codegen must never run")

    vendored_codegen.__module__ = "numba_cuda_mlir.numba_cuda.fake"

    @extending.intrinsic(api="values")
    def vendored_intrinsic(typingctx, x):
        return x(x), vendored_codegen

    if lowered_for is not None:
        # An MLIR lowering that does not cover the integer call below.
        @lowering_registry.lower(vendored_intrinsic, lowered_for, api="values")
        def lower_vendored_intrinsic(builder, signature, args):
            raise AssertionError("lowering for another signature must not be selected")

    def double(x):
        pass

    @numba_cuda_overload(double)
    def ol_double(x):
        return lambda x: vendored_intrinsic(x)

    @extending.type_callable(double)
    def type_double(context):
        return lambda x: x

    @lowering_registry.lower(double, types.Integer, api="values")
    def lower_double(builder, signature, args):
        value = args[0]
        return arith.addi(value, value)

    # Makes vendored_intrinsic visible to typing, as in issue #327.
    extending.refresh_registries()

    @cuda.jit
    def k(x):
        x[0] = double(x[0])

    x = np.array([3])
    k[1, 1](x)
    assert x[0] == 6


def test_extending_overload_with_lowering():
    from numba_cuda_mlir.extending import lowering_registry

    @lowering_registry.lower(np.sum, types.Any, types.Any, types.Any, api="values")
    def codegen(builder, signature, args):
        """
        TODO: we shouldn't need this...
        Numba should be able to type _and lower_ the overload as it is
        without defining a new lowering. Likely due to a misconfiguration in our
        lowering registry.
        """
        return args[0]

    @extending.overload(np.sum, typing_registry=extending.typing_registry)
    def sum_overload(a, b, c):
        """
        Silly overload of np.sum that takes three arguments and returns the first.
        """

        def ol(a, b, c):
            return a

        return ol

    extending.refresh_registries()

    @cuda.jit
    def k(x):
        x[0] = np.sum(x, x, x)[2]

    input = x = np.array([1, 2, 3])
    k[1, 1](x)
    assert x[0] == 3


def test_extending_overload_without_lowering():
    import logging

    logging.basicConfig(level=logging.DEBUG)

    @extending.overload(np.sum, typing_registry=extending.typing_registry)
    def sum_overload(a, c):
        def ol(a, c):
            return a

        return ol

    extending.refresh_registries()

    @cuda.jit
    def k(x):
        np.sum(x, x)

    x = np.array([1, 2, 3])
    k[1, 1](x)


def test_extending_overload_with_literal_argument():
    def double(x):
        raise NotImplementedError

    @extending.overload(double, typing_registry=extending.typing_registry)
    def overload_double(x):
        def impl(x):
            return x + x

        return impl

    extending.refresh_registries()

    @cuda.jit
    def k(out):
        out[0] = double(21)

    out = np.zeros(1, dtype=np.int64)
    k[1, 1](out)
    assert out[0] == 42


def test_extending_overload_with_prefer_literal():
    """A `prefer_literal=True` overload is typed with — and lowered from — the literal."""

    def tile_size(x):
        raise NotImplementedError

    @extending.overload(tile_size, typing_registry=extending.typing_registry, prefer_literal=True)
    def overload_tile_size(x):
        value = x.literal_value

        def impl(x):
            return value

        return impl

    extending.refresh_registries()

    @cuda.jit
    def k(out):
        out[0] = tile_size(32)

    out = np.zeros(1, dtype=np.int64)
    k[1, 1](out)
    assert out[0] == 32


def test_extending_overload_with_literally():
    """`literally()` retypes the argument, and the literal signature still lowers."""
    from numba_cuda_mlir.numba_cuda.misc.special import literally

    def unroll_factor(x):
        raise NotImplementedError

    @extending.overload(unroll_factor, typing_registry=extending.typing_registry)
    def overload_unroll_factor(x):
        if not isinstance(x, types.IntegerLiteral):
            return lambda x: literally(x)

        value = x.literal_value

        def impl(x):
            return value

        return impl

    extending.refresh_registries()

    @cuda.jit
    def k(out):
        out[0] = unroll_factor(8)

    out = np.zeros(1, dtype=np.int64)
    k[1, 1](out)
    assert out[0] == 8


def test_extending_overload_method_prefer_literal_omitted_default():
    """A literal-aware method overload still lowers when a trailing default is omitted.

    The `_impl_cache` key holds only the supplied arguments (receiver plus the
    literal), while the lowered signature also carries the omitted default, so
    the lookup has to compare the literal form with the omitted arguments
    stripped as well.
    """

    @extending.overload_method(
        types.Array,
        "tile_size",
        typing_registry=extending.typing_registry,
        prefer_literal=True,
    )
    def array_tile_size(arr, n, extra=None):
        value = n.literal_value

        def impl(arr, n, extra=None):
            return value

        return impl

    extending.refresh_registries()

    @cuda.jit
    def kernel(out):
        out[0] = out.tile_size(32)

    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](out)
    assert out[0] == 32


def test_extending_overload_method():
    """User-defined @overload_method dispatches through BoundFunction."""

    @extending.overload_method(
        types.Array,
        "doubled_first",
        typing_registry=extending.typing_registry,
    )
    def array_doubled_first(arr):
        def impl(arr):
            return arr[0] * 2

        return impl

    extending.refresh_registries()

    @cuda.jit
    def kernel(arr, out):
        out[0] = arr.doubled_first()

    arr = np.array([21.0], dtype=np.float64)
    out = np.zeros(1, dtype=np.float64)
    kernel[1, 1](arr, out)
    assert out[0] == 42.0


def test_extern_function_typeof():
    from numba_cuda_mlir.descriptor import mlir_target

    sig = types.void(types.int64)
    device_func = cuda.declare_device("my_device_func", sig)

    function_type = typeof(device_func)

    assert isinstance(function_type, types.Function)
    assert function_type.get_call_type(mlir_target.typing_context, sig.args, {}) == sig


def test_numba_cuda_overload_captures_extern_function():
    sig = types.void(types.int64)
    device_func = cuda.declare_device("my_device_func", sig)

    def my_func(val):
        pass

    @numba_cuda_overload(my_func)
    def ol_my_func(val):
        def impl(val):
            device_func(val)

        return impl

    extending.refresh_registries()

    @cuda.jit
    def kernel(val):
        my_func(val)

    with pytest.raises(
        Exception, match=r"(Undefined reference to|Unresolved extern function) 'my_device_func'"
    ):
        kernel[1, 1](1)


def test_overload_method_custom_type_uses_mlir_model_only():
    class MyObj:
        pass

    class MyObjType(types.Type):
        def __init__(self, obj):
            self.obj = obj
            super().__init__(name="MyObjType")

    @typeof_impl.register(MyObj)
    def typeof_myobj(val, c):
        return MyObjType(val)

    from numba_cuda_mlir.lowering_utilities import constant, unverified_convert

    @unverified_convert.register(MyObj)
    def convert_myobj(_val, target_type, **_kwargs):
        return constant(0, target_type)

    @register_model(MyObjType)
    class MyObjMlirModel(PrimitiveModel):
        def __init__(self, dmm, fe_type):
            from numba_cuda_mlir._mlir.extras import types as T

            super().__init__(dmm, fe_type, T.i8())

    @extending.overload_method(
        MyObjType,
        "execute",
        typing_registry=extending.typing_registry,
    )
    def ol_execute(obj, val):
        def impl(obj, val):
            pass

        return impl

    extending.refresh_registries()

    obj = MyObj()

    @cuda.jit
    def kernel(out, val):
        obj.execute(val)
        out[0] = val

    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](out, 7)
    assert out[0] == 7


def test_register_jitable():
    """register_jitable makes a plain Python function callable from device code."""

    @extending.register_jitable(typing_registry=extending.typing_registry)
    def triple(x):
        return x * 3

    extending.refresh_registries()

    @cuda.jit
    def kernel(arr):
        arr[0] = triple(arr[0])

    arr = np.array([7], dtype=np.int64)
    kernel[1, 1](arr)
    assert arr[0] == 21


def test_register_jitable_calls_register_jitable():
    """Chained register_jitable: one jitable function calls another."""

    @extending.register_jitable(typing_registry=extending.typing_registry)
    def add_one(x):
        return x + 1

    @extending.register_jitable(typing_registry=extending.typing_registry)
    def add_two(x):
        return add_one(add_one(x))

    extending.refresh_registries()

    @cuda.jit
    def kernel(arr):
        arr[0] = add_two(arr[0])

    arr = np.array([10], dtype=np.int64)
    kernel[1, 1](arr)
    assert arr[0] == 12


def test_overload_attribute():
    """overload_attribute exposes a read-only property on a Numba type."""

    @extending.overload_attribute(
        types.Array,
        "doubled_size",
        typing_registry=extending.typing_registry,
        lowering_registry=extending.lowering_registry,
    )
    def array_doubled_size(arr):
        def get(arr):
            return arr.size * 2

        return get

    extending.refresh_registries()

    @cuda.jit
    def kernel(arr, out):
        out[0] = arr.doubled_size

    arr = np.zeros(10, dtype=np.float64)
    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](arr, out)
    assert out[0] == 20


def test_overload_attribute_returns_string_literal():
    """String-literal attribute overloads can be returned from device functions."""

    from numba_cuda_mlir._mlir.extras import types as T

    class Dummy:
        pass

    class DummyType(types.Type):
        def __init__(self):
            super().__init__(name="Dummy")

    dummy_type = DummyType()

    @typeof_impl.register(Dummy)
    def typeof_dummy(val, c):
        return dummy_type

    class DummyModel(PrimitiveModel):
        def __init__(self, dmm, fe_type):
            super().__init__(dmm, fe_type, T.i8())

    register_model(DummyType)(DummyModel)

    @extending.overload_attribute(
        DummyType,
        "arrangement",
        inline="always",
        typing_registry=extending.typing_registry,
        lowering_registry=extending.lowering_registry,
    )
    def ol_arrangement(dummy):
        return lambda dummy: "col_major"

    dummy = Dummy()

    @cuda.jit
    def kernel(out):
        out[0] = 1 if dummy.arrangement == "col_major" else 0

    out = np.zeros(1, dtype=np.int32)
    kernel[1, 1](out)
    assert out[0] == 1


def test_overload_method_with_args():
    """overload_method with arguments beyond self."""

    @extending.overload_method(
        types.Array,
        "elem_plus",
        typing_registry=extending.typing_registry,
    )
    def array_elem_plus(arr, idx, val):
        def impl(arr, idx, val):
            return arr[idx] + val

        return impl

    extending.refresh_registries()

    @cuda.jit
    def kernel(arr, out):
        out[0] = arr.elem_plus(1, 100)

    arr = np.array([10, 20, 30], dtype=np.int64)
    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](arr, out)
    assert out[0] == 120


def test_overload_dispatches_on_type():
    """overload can return different implementations based on argument types."""

    def my_func(x):
        raise NotImplementedError

    @extending.overload(my_func, typing_registry=extending.typing_registry)
    def my_func_overload(x):
        if isinstance(x, types.Integer):

            def impl(x):
                return x + 1

            return impl
        elif isinstance(x, types.Float):

            def impl(x):
                return x * 2.0

            return impl

    extending.refresh_registries()

    @cuda.jit
    def kernel(int_out, float_out):
        int_out[0] = my_func(int_out[0])
        float_out[0] = my_func(float_out[0])

    int_out = np.array([10], dtype=np.int64)
    float_out = np.array([3.0], dtype=np.float64)
    kernel[1, 1](int_out, float_out)
    assert int_out[0] == 11
    assert float_out[0] == 6.0


def test_struct_field_model():
    """UnicodeTypeModel exposes field names and positions for struct access."""
    from numba_cuda_mlir._mlir import ir
    from numba_cuda_mlir.models import UnicodeTypeModel

    with ir.Context(), ir.Location.unknown():
        from numba_cuda_mlir.descriptor import MLIRTargetContext, MLIRTypingContext

        tc = MLIRTypingContext()
        tc.refresh()
        ctx = MLIRTargetContext(tc, target="numba_cuda_mlir")
        ctx.refresh()

        model = ctx.data_model_manager.lookup(types.UnicodeType("unicode_type"))
        assert isinstance(model, UnicodeTypeModel)
        assert model._fields == (
            "data",
            "length",
            "kind",
            "is_ascii",
            "hash",
            "meminfo",
            "parent",
        )
        assert model.get_field_position("data") == 0
        assert model.get_field_position("length") == 1
        assert model.get_field_position("kind") == 2
        assert model.get_field_position("is_ascii") == 3
        assert model.get_field_position("hash") == 4
        assert model.get_field_position("meminfo") == 5
        assert model.get_field_position("parent") == 6


def test_aggregate_type_field_model():
    """AggregateTypeModel exposes field names and positions for struct access."""
    from numba_cuda_mlir._mlir import ir
    from numba_cuda_mlir.models import AggregateTypeModel
    from numba_cuda_mlir.type_defs.aggregate_types import AggregateType

    with ir.Context(), ir.Location.unknown():
        from numba_cuda_mlir.descriptor import MLIRTargetContext, MLIRTypingContext

        tc = MLIRTypingContext()
        tc.refresh()
        ctx = MLIRTargetContext(tc, target="numba_cuda_mlir")
        ctx.refresh()

        my_type = AggregateType(
            "TestPoint",
            [("x", types.float32), ("y", types.float64), ("z", types.int32)],
        )
        model = ctx.data_model_manager.lookup(my_type)
        assert isinstance(model, AggregateTypeModel)
        assert model._fields == ("x", "y", "z")
        assert model.get_field_position("x") == 0
        assert model.get_field_position("y") == 1
        assert model.get_field_position("z") == 2


def test_register_jitable_tuple_return_cast_from_branches():
    @extending.register_jitable(typing_registry=extending.typing_registry)
    def select(flag, a, b):
        if flag:
            return (a, b)
        return (b, a)

    @cuda.jit
    def kernel(out, flag):
        x, y = select(flag, 1, 2)
        out[0] = x
        out[1] = y

    out = np.zeros(2, dtype=np.int64)
    kernel[1, 1](out, True)
    assert out.tolist() == [1, 2]

    kernel[1, 1](out, False)
    assert out.tolist() == [2, 1]


def test_overload_method_omitted_trailing_default_arg():
    from numba_cuda_mlir._mlir.dialects import llvm
    from numba_cuda_mlir._mlir.extras import types as T
    from numba_cuda_mlir.lowering_utilities import unverified_convert

    class Widget:
        pass

    class WidgetType(types.Type):
        def __init__(self):
            super().__init__(name="Widget")

    widget_type = WidgetType()

    @typeof_impl.register(Widget)
    def typeof_widget(val, c):
        return widget_type

    class WidgetModel(PrimitiveModel):
        def __init__(self, dmm, fe):
            super().__init__(dmm, fe, T.i8())

    register_model(WidgetType)(WidgetModel)

    @unverified_convert.register(Widget)
    def convert_widget(val, tt, signed=False):
        return llvm.mlir_undef(T.i8())

    @extending.overload_method(
        WidgetType,
        "poke",
        strict=False,
        typing_registry=extending.typing_registry,
    )
    def ol_poke(w, buf, extra=None):
        if not isinstance(buf, types.Array):
            return

        def impl(w, buf, extra=None):
            buf[0] = buf[0] + 1

        return impl

    widget = Widget()

    @cuda.jit
    def kernel(buf):
        widget.poke(buf)

    buf = np.zeros(1, dtype=np.int32)
    kernel[1, 1](buf)
    assert buf[0] == 1


def test_register_jitable_omitted_default_arg():
    @extending.register_jitable(typing_registry=extending.typing_registry)
    def fill(out, lda=None):
        out[0] = 42

    @cuda.jit
    def kernel(out):
        fill(out)

    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](out)
    assert out[0] == 42


def test_inline_overload_returning_dtype_token_for_local_array():
    def get_value_type():
        return np.float32

    @extending.overload(
        get_value_type,
        inline="always",
        typing_registry=extending.typing_registry,
        lowering_registry=extending.lowering_registry,
    )
    def ol_get_value_type():
        return lambda: types.float32

    @cuda.jit
    def kernel(buf):
        arr = cuda.local.array(shape=(1,), dtype=get_value_type())
        arr[0] = buf[0]
        buf[0] = arr[0] + 1

    buf = np.zeros(1, dtype=np.float32)
    kernel[1, 1](buf)
    assert buf[0] == 1


def _register_selected_identity(selected_type, api="values"):
    def selected_identity(value):
        raise NotImplementedError

    class SelectedIdentityTemplate(ConcreteTemplate):
        key = selected_identity
        cases = [selected_type(selected_type)]

    extending.typing_registry.register_global(
        selected_identity, types.Function(SelectedIdentityTemplate)
    )

    if api == "values":

        @extending.lowering_registry.lower(selected_identity, selected_type, api="values")
        def lower_selected_identity(builder, signature, args):
            return args[0]
    else:

        @extending.lowering_registry.lower(selected_identity, selected_type)
        def lower_selected_identity(builder, target, args, kws):
            builder.store_var(target, builder.load_var(args[0]))

    extending.refresh_registries()
    return selected_identity


@pytest.mark.parametrize("api", ["vars", "values"])
@pytest.mark.parametrize(
    "source_dtype,selected_type,value",
    [
        (np.uint32, types.uint64, 7),
        (np.uint32, types.uint64, np.iinfo(np.uint32).max),
        (np.int32, types.int64, -7),
        (np.int32, types.int64, np.iinfo(np.int32).min),
        (np.int32, types.float64, -7),
        (np.float32, types.float64, 1.75),
    ],
    ids=["unsigned", "unsigned-boundary", "signed", "signed-boundary", "int-float", "float"],
)
def test_registered_call_selected_argument_conversion(source_dtype, selected_type, value, api):
    """Selected operand conversions must not change subsequent uses of the input."""
    selected_identity = _register_selected_identity(selected_type, api)

    @cuda.jit
    def kernel(source, converted, original):
        value = source[0]
        converted[0] = selected_identity(value)
        original[0] = value
        original[1] = value - source[0]

    source = np.array([value], dtype=source_dtype)
    converted = np.zeros(1, dtype=str(selected_type))
    original = np.zeros(2, dtype=source_dtype)
    kernel[1, 1](source, converted, original)
    assert converted[0] == value
    assert original[0] == value
    assert original[1] == 0


def test_registered_call_keyword_binding_uses_parameter_names():
    from numba_cuda_mlir._mlir.dialects import arith
    from numba_cuda_mlir._mlir.extras import types as T

    def subtract(b, c):
        raise NotImplementedError

    class SubtractTemplate(AbstractTemplate):
        key = subtract

        def generic(self, args, kws):
            return types.float64(types.int64, types.float64).replace(
                pysig=inspect.signature(subtract)
            )

    extending.typing_registry.register_global(subtract, types.Function(SubtractTemplate))

    @extending.lowering_registry.lower(subtract, types.int64, types.float64, api="values")
    def lower_subtract(builder, signature, args):
        b, c = args
        return arith.subf(arith.sitofp(T.f64(), b), c)

    extending.refresh_registries()

    @cuda.jit
    def kernel(b, c, out):
        out[0] = subtract(b, c)
        out[1] = subtract(b, c=c)
        out[2] = subtract(b=b, c=c)
        out[3] = subtract(c=c, b=b)

    out = np.zeros(4, dtype=np.float64)
    kernel[1, 1](np.int32(3), np.float32(1.75), out)
    np.testing.assert_array_equal(out, [1.25] * 4)


@pytest.mark.parametrize("keyword_only", [False, True])
def test_registered_call_materializes_omitted_defaults(keyword_only):
    """Both defaults are consumed as operands, including reordered keyword-only calls."""
    from numba_cuda_mlir._mlir.dialects import arith
    from numba_cuda_mlir._mlir.extras import types as T

    if keyword_only:

        def weighted(a, *, b=7, c=1.75):
            raise NotImplementedError
    else:

        def weighted(a, b=7, c=1.75):
            raise NotImplementedError

    pysig = inspect.signature(weighted)

    class WeightedTemplate(AbstractTemplate):
        key = weighted

        def generic(self, args, kws):
            bound = pysig.bind(*args, **kws)
            selected_args = (
                types.int64,
                types.int64 if "b" in bound.arguments else types.Omitted(7),
                types.float64 if "c" in bound.arguments else types.Omitted(1.75),
            )
            return types.float64(*selected_args).replace(pysig=pysig)

    extending.typing_registry.register_global(weighted, types.Function(WeightedTemplate))

    @extending.lowering_registry.lower(weighted, types.int64, types.Any, types.Any, api="values")
    def lower_weighted(builder, signature, args):
        a, b, c = args
        a = arith.sitofp(T.f64(), a)
        b = arith.sitofp(T.f64(), b)
        return arith.addf(arith.mulf(a, b), c)

    extending.refresh_registries()

    @cuda.jit
    def kernel(a, b, c, out):
        out[0] = weighted(a)
        out[1] = weighted(a, c=c)
        out[2] = weighted(c=c, a=a)
        out[3] = weighted(a, b=b)
        out[4] = weighted(b=b, a=a)
        out[5] = weighted(c=c, b=b, a=a)

    out = np.zeros(6, dtype=np.float64)
    kernel[1, 1](np.int32(3), np.int32(2), np.float32(4.5), out)
    np.testing.assert_array_equal(out, [22.75, 25.5, 25.5, 7.75, 7.75, 10.5])


def test_registered_call_keywords_require_python_signature():
    def missing_binding(b, c):
        raise NotImplementedError

    class MissingBindingTemplate(AbstractTemplate):
        key = missing_binding

        def generic(self, args, kws):
            return types.float64(types.int64, types.float64)

    extending.typing_registry.register_global(
        missing_binding, types.Function(MissingBindingTemplate)
    )

    @extending.lowering_registry.lower(missing_binding, types.int64, types.float64, api="values")
    def lower_missing_binding(builder, signature, args):
        return args[1]

    extending.refresh_registries()

    @cuda.jit
    def kernel(out):
        out[0] = missing_binding(c=1.75, b=3)

    with pytest.raises(TypingError, match="(?i)(binding|pysig|python signature)"):
        kernel[1, 1](np.zeros(1, dtype=np.float64))


@pytest.mark.parametrize("api", ["vars", "values"])
def test_registered_call_selected_conversions_with_ssa_disabled(monkeypatch, api):
    from numba_cuda_mlir import mlir_compiler

    get_compiler_class = mlir_compiler.get_compiler_class

    def get_non_ssa_compiler(*args, **kwargs):
        compiler_class = get_compiler_class(*args, **kwargs)

        class NonSSACompiler(compiler_class):
            def define_pipelines(self):
                self.state.flags.enable_ssa = False
                return super().define_pipelines()

        return NonSSACompiler

    monkeypatch.setattr(mlir_compiler, "get_compiler_class", get_non_ssa_compiler)
    selected_identity = _register_selected_identity(types.uint64, api)

    @cuda.jit
    def kernel(flag, a, b, out):
        if flag:
            result = selected_identity(a)
        else:
            result = selected_identity(b)
        out[0] = result
        out[1] = selected_identity(a)
        out[2] = selected_identity(b)

    out = np.zeros(3, dtype=np.uint64)
    for flag, expected in [(True, 7), (False, 19)]:
        kernel[1, 1](flag, np.uint32(7), np.uint32(19), out)
        np.testing.assert_array_equal(out, [expected, 7, 19])


@pytest.mark.parametrize("api", ["vars", "values"])
@pytest.mark.parametrize("case", ["empty", "homogeneous", "heterogeneous", "tuple"])
def test_registered_call_folded_varargs_and_keyword_scale(case, api):
    from numba_cuda_mlir._mlir.dialects import arith
    from numba_cuda_mlir._mlir.extras import types as T

    def scaled_sum(*values, scale):
        raise NotImplementedError

    element_types = {
        "empty": (),
        "homogeneous": (types.float64, types.float64),
        "heterogeneous": (types.int64, types.float64),
        "tuple": (types.Tuple((types.int32, types.float32)),),
    }[case]
    selected_args = (StarArgTuple(element_types), types.float64)

    class ScaledSumTemplate(AbstractTemplate):
        key = scaled_sum

        def generic(self, args, kws):
            return types.float64(*selected_args).replace(pysig=inspect.signature(scaled_sum))

    extending.typing_registry.register_global(scaled_sum, types.Function(ScaledSumTemplate))

    def lower_scaled_sum(builder, signature, args):
        values, scale = args
        total = arith.constant(T.f64(), 0.0)
        for value, arg_type in zip(values, signature.args[0].types):
            if isinstance(arg_type, types.BaseTuple):
                values_and_types = zip(value, arg_type.types)
            else:
                values_and_types = [(value, arg_type)]
            for value, value_type in values_and_types:
                if isinstance(value_type, types.Integer):
                    value = arith.sitofp(T.f64(), value)
                elif value_type == types.float32:
                    value = arith.extf(T.f64(), value)
                total = arith.addf(total, value)
        # Keep the bundle's arity numerically observable: a tuple passed as one
        # argument must remain distinct from the two expanded scalar arguments.
        total = arith.addf(total, arith.constant(T.f64(), float(len(values))))
        return arith.mulf(total, scale)

    if api == "values":
        extending.lowering_registry.lower(scaled_sum, *selected_args, api="values")(
            lower_scaled_sum
        )
    else:

        @extending.lowering_registry.lower(scaled_sum, *selected_args)
        def lower_legacy_scaled_sum(builder, target, args, kws):
            values = tuple(builder.load_var(arg) for arg in args)
            scale = builder.load_var(dict(kws)["scale"])
            result = lower_scaled_sum(builder, types.float64(*selected_args), (values, scale))
            builder.store_var(target, result)

    extending.refresh_registries()

    if case == "empty":

        @cuda.jit
        def kernel(a, b, scale, out):
            out[0] = scaled_sum(*(), scale=scale)
    elif case == "tuple":

        @cuda.jit
        def kernel(a, b, scale, out):
            out[0] = scaled_sum((a, b), scale=scale)
    else:

        @cuda.jit
        def kernel(a, b, scale, out):
            out[0] = scaled_sum(*(a, b), scale=scale)

    b = np.int32(2) if case == "homogeneous" else np.float32(1.75)
    out = np.zeros(1, dtype=np.float64)
    kernel[1, 1](np.int32(3), b, np.int32(2), out)
    arity = 1 if case == "tuple" else 2
    expected = 0 if case == "empty" else (3 + b + arity) * 2
    assert out[0] == expected


def test_legacy_registered_call_preserves_keywords_defaults_and_literals():
    from numba_cuda_mlir._mlir.dialects import arith
    from numba_cuda_mlir._mlir.extras import types as T
    from numba_cuda_mlir.numba_cuda.core.ir import Var

    def weighted(a, b=7, *, c=1.75):
        raise NotImplementedError

    pysig = inspect.signature(weighted)

    class WeightedTemplate(AbstractTemplate):
        key = weighted

        def generic(self, args, kws):
            bound = pysig.bind(*args, **kws)
            return types.float64(
                types.int64,
                types.int64 if "b" in bound.arguments else types.Omitted(7),
                types.float64 if "c" in bound.arguments else types.Omitted(1.75),
            ).replace(pysig=pysig)

    extending.typing_registry.register_global(weighted, types.Function(WeightedTemplate))

    @extending.lowering_registry.lower(weighted, types.int64, types.Any, types.Any)
    def lower_weighted(builder, target, args, kws):
        bound = pysig.bind(*args, **dict(kws))
        bound.apply_defaults()

        def load(value):
            if not isinstance(value, Var):
                return arith.constant(T.f64(), float(value))
            value_type = builder.get_numba_type(value)
            if isinstance(value_type, types.IntegerLiteral):
                return arith.constant(T.f64(), float(value_type.literal_value))
            lowered = builder.load_var(value)
            if isinstance(value_type, types.Integer):
                return arith.sitofp(T.f64(), lowered)
            return lowered

        a, b, c = (load(bound.arguments[name]) for name in ("a", "b", "c"))
        builder.store_var(target, arith.addf(arith.mulf(a, b), c))

    extending.refresh_registries()

    @cuda.jit
    def kernel(value, b, c, out, original):
        out[0] = weighted(value)
        out[1] = weighted(value, c=c)
        out[2] = weighted(c=c, a=value)
        out[3] = weighted(value, b=b)
        out[4] = weighted(b=b, a=value)
        out[5] = weighted(c=c, b=b, a=value)
        out[6] = weighted(value, c=value)
        out[7] = weighted(3, b=2, c=4.5)
        original[0] = value

    out = np.zeros(8, dtype=np.float64)
    original = np.zeros(1, dtype=np.uint32)
    kernel[1, 1](np.uint32(3), np.int32(2), np.float32(4.5), out, original)
    np.testing.assert_array_equal(out, [22.75, 25.5, 25.5, 7.75, 7.75, 10.5, 24, 10.5])
    assert original[0] == 3


def test_legacy_intrinsic_preserves_literal_metadata():
    @extending.intrinsic
    def add_literal_offset(typingctx, value, offset):
        def codegen(builder, target, args, kws):
            from numba_cuda_mlir._mlir.dialects import arith
            from numba_cuda_mlir._mlir.extras import types as T

            offset = builder.get_numba_type(args[1]).literal_value
            result = arith.addi(builder.load_var(args[0]), arith.constant(T.i64(), offset))
            builder.store_var(target, result)

        return types.int64(types.int64, types.int64), codegen

    extending.refresh_registries()

    @cuda.jit
    def kernel(value, out):
        out[0] = add_literal_offset(value, 11)

    out = np.zeros(1, dtype=np.int64)
    kernel[1, 1](np.int32(-7), out)
    assert out[0] == 4


def test_legacy_setitem_statement_and_expression_targets():
    import operator
    from enum import IntEnum
    from numba_cuda_mlir._mlir.dialects import memref
    from numba_cuda_mlir.lowering_utilities import index_of

    class StoredValue(IntEnum):
        seven = 7

    # A test-local enum confines the operator registration to this consumer.
    selected_args = (types.Array(types.int64, 1, "C"), types.int64, typeof(StoredValue.seven))

    def set_value(array, index, value):
        raise NotImplementedError

    class SetValueTemplate(ConcreteTemplate):
        key = set_value
        cases = [types.none(*selected_args)]

    extending.typing_registry.register_global(set_value, types.Function(SetValueTemplate))

    @extending.lowering_registry.lower(operator.setitem, *selected_args)
    @extending.lowering_registry.lower(set_value, *selected_args)
    def legacy_setitem(builder, target, args, kws):
        array, index, value = builder.load_vars(args)
        memref.store(value, array, [index_of(index)])
        if target is not None:
            builder.store_var(target, None)

    extending.refresh_registries()

    @cuda.jit
    def kernel(array, value):
        array[0] = value
        result = set_value(array, 1, value)
        if result is None:
            array[2] = array[0] + array[1]

    array = np.zeros(3, dtype=np.int64)
    kernel[1, 1](array, StoredValue.seven)
    np.testing.assert_array_equal(array, [7, 7, 14])
