# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from numba_cuda_mlir import extending
from numba_cuda_mlir.testing import NumbaCUDATestCase

import numpy as np
import numba_cuda_mlir
from numba_cuda_mlir import cuda
from numba_cuda_mlir import extending

from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import func
from numba_cuda_mlir.lowering_utilities import convert, get_or_insert_function
from numba_cuda_mlir.numba_cuda import types

import functools
import inspect
import math
import pickle
import pytest
import unittest


from numba_cuda_mlir.numba_cuda import cgutils
from numba_cuda_mlir.numba_cuda.typing.templates import AttributeTemplate
from numba_cuda_mlir.numba_cuda.cudadecl import registry as cuda_registry
from numba_cuda_mlir.numba_cuda.cudaimpl import lower_attr as cuda_lower_attr

from numba_cuda_mlir.numba_cuda.core import errors
from numba_cuda_mlir.numba_cuda.errors import LoweringError

from numba_cuda_mlir.extending import (
    type_callable,
    lower_builtin,
    overload,
    overload_method,
    intrinsic,
    _Intrinsic,
    register_jitable,
    register_model,
    typeof_impl,
    make_attribute_wrapper,
    typing_registry as extending_typing_registry,
)
from numba_cuda_mlir.models import StructModel

overload = functools.partial(overload, typing_registry=extending_typing_registry)
overload_method = functools.partial(overload_method, typing_registry=extending_typing_registry)
register_jitable = functools.partial(register_jitable, typing_registry=extending_typing_registry)


class Interval:
    """
    A half-open interval on the real number line.
    """

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    def __repr__(self):
        return "Interval(%f, %f)" % (self.lo, self.hi)

    @property
    def width(self):
        return self.hi - self.lo


@cuda.jit
def interval_width(interval):
    return interval.width


@cuda.jit
def sum_intervals(i, j):
    return Interval(i.lo + j.lo, i.hi + j.hi)


class IntervalType(types.Type):
    def __init__(self):
        super().__init__(name="Interval")


interval_type = IntervalType()


@typeof_impl.register(Interval)
def typeof_interval(val, c):
    return interval_type


@type_callable(Interval)
def type_interval(context):
    def typer(lo, hi):
        if isinstance(lo, types.Float) and isinstance(hi, types.Float):
            return interval_type

    return typer


@register_model(IntervalType)
class IntervalModel(StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ("lo", types.float64),
            ("hi", types.float64),
        ]
        super().__init__(dmm, fe_type, members)


make_attribute_wrapper(IntervalType, "lo", "lo")
make_attribute_wrapper(IntervalType, "hi", "hi")


@lower_builtin(Interval, types.Float, types.Float)
def impl_interval(context, builder, sig, args):
    typ = sig.return_type
    lo, hi = args
    interval = cgutils.create_struct_proxy(typ)(context, builder)
    interval.lo = lo
    interval.hi = hi
    return interval._getvalue()


@cuda_registry.register_attr
class Interval_attrs(AttributeTemplate):
    key = IntervalType

    def resolve_width(self, mod):
        return types.float64


@cuda_lower_attr(IntervalType, "width")
def cuda_Interval_width(context, builder, sig, arg):
    lo = builder.extract_value(arg, 0)
    hi = builder.extract_value(arg, 1)
    return builder.fsub(hi, lo)


# -----------------------------------------------------------------------
# Define a function's typing and implementation using the classical
# two-step API


def func1(x=None):
    raise NotImplementedError


def type_func1_(context):
    def typer(x=None):
        if x in (None, types.none):
            # 0-arg or 1-arg with None
            return types.int32
        elif isinstance(x, types.Float):
            # 1-arg with float
            return x

    return typer


type_func1 = type_callable(func1)(type_func1_)


@lower_builtin(func1)
@lower_builtin(func1, types.none)
def func1_nullary(context, builder, sig, args):
    return context.get_constant(sig.return_type, 42)


@lower_builtin(func1, types.Float)
def func1_unary(context, builder, sig, args):
    def func1_impl(x):
        return math.sqrt(2 * x)

    return context.compile_internal(builder, func1_impl, sig, args)


# -----------------------------------------------------------------------
# Overload an already defined built-in function, extending it for new types.


def call_func1_nullary(res):
    res[0] = func1()


def call_func1_unary(x, res):
    res[0] = func1(x)


@pytest.mark.xfail(True, reason="Extension API not supported")
class TestExtending(NumbaCUDATestCase):
    def test_attributes(self):
        @numba_cuda_mlir.cuda.jit
        def f(r, x):
            iv = Interval(x[0], x[1])
            r[0] = iv.lo
            r[1] = iv.hi

        x = np.asarray((1.5, 2.5))
        r = np.zeros_like(x)

        f[1, 1](r, x)

        np.testing.assert_equal(r, x)

    def test_property(self):
        @numba_cuda_mlir.cuda.jit
        def f(r, x):
            iv = Interval(x[0], x[1])
            r[0] = iv.width

        x = np.asarray((1.5, 2.5))
        r = np.zeros(1)

        f[1, 1](r, x)

        np.testing.assert_allclose(r[0], x[1] - x[0])

    def test_extension_type_as_arg(self):
        @numba_cuda_mlir.cuda.jit
        def f(r, x):
            iv = Interval(x[0], x[1])
            r[0] = interval_width(iv)

        x = np.asarray((1.5, 2.5))
        r = np.zeros(1)

        f[1, 1](r, x)

        np.testing.assert_allclose(r[0], x[1] - x[0])

    def test_extension_type_as_retvalue(self):
        @numba_cuda_mlir.cuda.jit
        def f(r, x):
            iv1 = Interval(x[0], x[1])
            iv2 = Interval(x[2], x[3])
            iv_sum = sum_intervals(iv1, iv2)
            r[0] = iv_sum.lo
            r[1] = iv_sum.hi

        x = np.asarray((1.5, 2.5, 3.0, 4.0))
        r = np.zeros(2)

        f[1, 1](r, x)

        expected = np.asarray((x[0] + x[2], x[1] + x[3]))
        np.testing.assert_allclose(r, expected)


class TestExtendingLinkage(NumbaCUDATestCase):
    @pytest.mark.numba_cuda_test_binaries("a", "cubin", "cu", "fatbin", "o", "ptx", "ltoir")
    def test_extension_adds_linkable_code(self):
        binaries = self.numba_cuda_test_binaries
        files = (
            (binaries.test_device_functions_a, cuda.Archive),
            (binaries.test_device_functions_cubin, cuda.Cubin),
            (binaries.test_device_functions_cu, cuda.CUSource),
            (binaries.test_device_functions_fatbin, cuda.Fatbin),
            (binaries.test_device_functions_o, cuda.Object),
            (binaries.test_device_functions_ptx, cuda.PTXSource),
            (binaries.test_device_functions_ltoir, cuda.LTOIR),
        )

        lto = True

        for path, ctor in files:
            if ctor == cuda.LTOIR and not lto:
                # Don't try to test with LTOIR if LTO is not enabled
                continue

            with open(path, "rb") as f:
                code_object = ctor(f.read())

            def external_add(x, y):
                return x + y

            @type_callable(external_add)
            def type_external_add(context):
                def typer(x, y):
                    if x == types.uint32 and y == types.uint32:
                        return types.uint32

                return typer

            @lower_builtin(external_add, types.uint32, types.uint32)
            def lower_external_add(builder, target, args, kwargs):
                builder.link_external_item(code_object)
                i32 = builder.get_mlir_type(types.uint32)
                fnty = mlir_ir.FunctionType.get([i32, i32], [i32])
                fn = get_or_insert_function("add_cabi", fnty, builder.mlir_gpu_module)
                operands = [convert(arg, i32) for arg in builder.load_vars(args)]
                result = func.call(
                    result=[i32],
                    callee=fn.name.value,
                    operands_=operands,
                )
                builder.store_var(target, result)

            @numba_cuda_mlir.cuda.jit(lto=lto)
            def use_external_add(r, x, y):
                r[0] = external_add(x[0], y[0])

            r = np.zeros(1, dtype=np.uint32)
            x = np.ones(1, dtype=np.uint32)
            y = np.ones(1, dtype=np.uint32) * 2

            use_external_add[1, 1](r, x, y)

            np.testing.assert_equal(r[0], 3)

            @numba_cuda_mlir.cuda.jit(lto=lto)
            def use_external_add_device(x, y):
                return external_add(x, y)

            @numba_cuda_mlir.cuda.jit(lto=lto)
            def use_external_add_kernel(r, x, y):
                r[0] = use_external_add_device(x[0], y[0])

            r = np.zeros(1, dtype=np.uint32)
            x = np.ones(1, dtype=np.uint32)
            y = np.ones(1, dtype=np.uint32) * 2

            use_external_add_kernel[1, 1](r, x, y)

            np.testing.assert_equal(r[0], 3)

    @pytest.mark.xfail(True, reason="Typing error")
    def test_linked_called_through_overload(self):
        cu_code = cuda.CUSource(
            """
            extern "C" __device__
            int bar(int *out, int a)
            {
              *out = a * 2;
              return 0;
            }
        """
        )

        bar = cuda.declare_device("bar", "int32(int32)", link=cu_code)

        def bar_call(val):
            pass

        @overload(bar_call, target="cuda")
        def ol_bar_call(a):
            return lambda a: bar(a)

        @numba_cuda_mlir.cuda.jit("void(int32[::1], int32[::1])")
        def foo(r, x):
            i = cuda.grid(1)
            if i < len(r):
                r[i] = bar_call(x[i])

        x = np.arange(10, dtype=np.int32)
        r = np.empty_like(x)

        foo[1, 32](r, x)

        np.testing.assert_equal(r, x * 2)


class TestLowLevelExtending(NumbaCUDATestCase):
    """
    Test the low-level two-tier extension API.
    """

    # Check with `@cuda.jit` from within the test process and also in a new test
    # process so as to check the registration mechanism.

    @pytest.mark.xfail(True, reason="ICE")
    def test_func1(self):
        pyfunc = call_func1_nullary
        cfunc = cuda.jit(pyfunc)
        res = np.zeros(1)
        cfunc[1, 1](res)
        self.assertPreciseEqual(res[0], 42.0)
        pyfunc = call_func1_unary
        cfunc = cuda.jit(pyfunc)
        self.assertPreciseEqual(res[0], 42.0)
        cfunc[1, 1](18.0, res)
        self.assertPreciseEqual(res[0], 6.0)

    def test_type_callable_keeps_function(self):
        self.assertIs(type_func1, type_func1_)
        self.assertIsNotNone(type_func1)


class TestHighLevelExtending(NumbaCUDATestCase):
    """
    Test the high-level combined API.
    """

    def test_typing_vs_impl_signature_mismatch_handling(self):
        """
        Tests that an overload which has a differing typing and implementing
        signature raises an exception.
        """

        def gen_ol(impl=None):
            def myoverload(a, b, c, kw=None):
                pass

            @overload(myoverload)
            def _myoverload_impl(a, b, c, kw=None):
                return impl

            extending.refresh_registries()

            @cuda.jit
            def foo(a, b, c, d):
                myoverload(a, b, c, kw=d)

            return foo

        sentinel = "Typing and implementation arguments differ in"

        # kwarg value is different
        def impl1(a, b, c, kw=12):
            if a > 10:
                return 1
            else:
                return -1

        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl1)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("keyword argument default values", msg)
        self.assertIn('<Parameter "kw=12">', msg)
        self.assertIn('<Parameter "kw=None">', msg)

        # kwarg name is different
        def impl2(a, b, c, kwarg=None):
            if a > 10:
                return 1
            else:
                return -1

        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl2)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("keyword argument names", msg)
        self.assertIn('<Parameter "kwarg=None">', msg)
        self.assertIn('<Parameter "kw=None">', msg)

        # arg name is different
        def impl3(z, b, c, kw=None):
            if a > 10:  # noqa: F821
                return 1
            else:
                return -1

        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl3)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("argument names", msg)
        self.assertFalse("keyword" in msg)
        self.assertIn('<Parameter "a">', msg)
        self.assertIn('<Parameter "z">', msg)

        from .overload_usecases import impl4, impl5

        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl4)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("argument names", msg)
        self.assertFalse("keyword" in msg)
        self.assertIn("First difference: 'z'", msg)

        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl5)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("argument names", msg)
        self.assertFalse("keyword" in msg)
        self.assertIn('<Parameter "a">', msg)
        self.assertIn('<Parameter "z">', msg)

        # too many args
        def impl6(a, b, c, d, e, kw=None):
            if a > 10:
                return 1
            else:
                return -1

        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl6)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("argument names", msg)
        self.assertFalse("keyword" in msg)
        self.assertIn('<Parameter "d">', msg)
        self.assertIn('<Parameter "e">', msg)

        # too few args
        def impl7(a, b, kw=None):
            if a > 10:
                return 1
            else:
                return -1

        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl7)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("argument names", msg)
        self.assertFalse("keyword" in msg)
        self.assertIn('<Parameter "c">', msg)

        # too many kwargs
        def impl8(a, b, c, kw=None, extra_kwarg=None):
            if a > 10:
                return 1
            else:
                return -1

        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl8)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("keyword argument names", msg)
        self.assertIn('<Parameter "extra_kwarg=None">', msg)

        # too few kwargs
        def impl9(a, b, c):
            if a > 10:
                return 1
            else:
                return -1

        with self.assertRaises(errors.TypingError) as e:
            gen_ol(impl9)[1, 1](1, 2, 3, 4)
        msg = str(e.exception)
        self.assertIn(sentinel, msg)
        self.assertIn("keyword argument names", msg)
        self.assertIn('<Parameter "kw=None">', msg)

    def test_typing_vs_impl_signature_mismatch_handling_var_positional(self):
        """
        Tests that an overload which has a differing typing and implementing
        signature raises an exception and uses VAR_POSITIONAL (*args) in typing
        """

        def myoverload(a, kw=None):
            pass

        from .overload_usecases import var_positional_impl

        overload(myoverload)(var_positional_impl)
        extending.refresh_registries()

        @cuda.jit
        def foo(a, b):
            myoverload(a, b, 9, kw=11)

        with self.assertRaises(errors.TypingError) as e:
            foo[1, 1](1, 5)
        msg = str(e.exception)
        self.assertIn("VAR_POSITIONAL (e.g. *args) argument kind", msg)
        self.assertIn("offending argument name is '*star_args_token'", msg)

    @pytest.mark.xfail(True, reason="Typing error")
    def test_typing_vs_impl_signature_mismatch_handling_var_keyword(self):
        """
        Tests that an overload which uses **kwargs (VAR_KEYWORD)
        """

        def gen_ol(impl, strict=True):
            def myoverload(a, kw=None):
                pass

            overload(myoverload, strict=strict)(impl)

            @cuda.jit
            def foo(a, b):
                myoverload(a, kw=11)

            return foo

        # **kwargs in typing
        def ol1(a, **kws):
            def impl(a, kw=10):
                return a

            return impl

        gen_ol(ol1, False)[1, 1](1, 2)  # no error if strictness not enforced
        with self.assertRaises(errors.TypingError) as e:
            gen_ol(ol1)[1, 1](1, 2)
        msg = str(e.exception)
        self.assertIn("use of VAR_KEYWORD (e.g. **kwargs) is unsupported", msg)
        self.assertIn("offending argument name is '**kws'", msg)

        # **kwargs in implementation
        def ol2(a, kw=0):
            def impl(a, **kws):
                return a

            return impl

        with self.assertRaises(errors.TypingError) as e:
            gen_ol(ol2)[1, 1](1, 2)
        msg = str(e.exception)
        self.assertIn("use of VAR_KEYWORD (e.g. **kwargs) is unsupported", msg)
        self.assertIn("offending argument name is '**kws'", msg)

    def test_overload_method_kwargs(self):
        # Issue #3489
        @overload_method(types.Array, "foo")
        def fooimpl(arr, a_kwarg=10):
            def impl(arr, a_kwarg=10):
                return a_kwarg

            return impl

        @cuda.jit
        def bar(A, res):
            res[0] = A.foo()
            res[1] = A.foo(20)
            res[2] = A.foo(a_kwarg=30)

        Z = np.arange(5)
        res = np.zeros(3)
        bar[1, 1](Z, res)
        self.assertEqual(res[0], 10)
        self.assertEqual(res[1], 20)
        self.assertEqual(res[2], 30)

    def test_overload_method_literal_unpack(self):
        # Issue #3683
        @overload_method(types.Array, "litfoo")
        def litfoo(arr, val):
            # Must be an integer
            if isinstance(val, types.Integer):
                # Must not be literal
                if not isinstance(val, types.Literal):

                    def impl(arr, val):
                        return val

                    return impl

        extending.refresh_registries()

        @cuda.jit
        def bar(A, res):
            res[0] = A.litfoo(0xCAFE)

        A = np.zeros(1)
        res = np.zeros(1)
        bar[1, 1](A, res)
        self.assertEqual(res[0], 0xCAFE)


def _assert_cache_stats(cfunc, expect_hit, expect_misses):
    hit = cfunc._cache_hits[cfunc.signatures[0]]
    if hit != expect_hit:
        raise AssertionError("cache not used")
    miss = cfunc._cache_misses[cfunc.signatures[0]]
    if miss != expect_misses:
        raise AssertionError("cache not used")


class TestIntrinsic(NumbaCUDATestCase):
    @pytest.mark.xfail(True, reason="ICE")
    def test_void_return(self):
        """
        Verify that returning a None from codegen function is handled
        automatically for void functions, otherwise raise exception.
        """

        @intrinsic
        def void_func(typingctx, a):
            sig = types.void(types.int32)

            def codegen(context, builder, signature, args):
                pass  # do nothing, return None, should be turned into
                # dummy value

            return sig, codegen

        @intrinsic
        def non_void_func(typingctx, a):
            sig = types.int32(types.int32)

            def codegen(context, builder, signature, args):
                pass  # oops, should be returning a value here, raise exception

            return sig, codegen

        @cuda.jit
        def call_void_func():
            void_func(1)

        @cuda.jit
        def call_non_void_func():
            non_void_func(1)

        # void func should work
        self.assertEqual(call_void_func[1, 1](), None)
        # not void function should raise exception
        with self.assertRaises(LoweringError) as e:
            call_non_void_func[1, 1]()
        self.assertIn("non-void function returns None", e.exception.msg)

    @pytest.mark.xfail(True, reason="ICE")
    def test_serialization(self):
        """
        Test serialization of intrinsic objects
        """

        # define a intrinsic
        @intrinsic
        def identity(context, x):
            def codegen(context, builder, signature, args):
                return args[0]

            sig = x(x)
            return sig, codegen

        # use in a cuda.jit function
        @cuda.jit
        def foo(x):
            identity(x)

        self.assertEqual(foo[1, 1](1), None)

        # get serialization memo
        memo = _Intrinsic._memo
        memo_size = len(memo)

        # pickle foo and check memo size
        serialized_foo = pickle.dumps(foo)
        # increases the memo size
        memo_size += 1
        self.assertEqual(memo_size, len(memo))
        # unpickle
        foo_rebuilt = pickle.loads(serialized_foo)
        self.assertEqual(memo_size, len(memo))
        # check rebuilt foo

        self.assertEqual(foo[1, 1](1), foo_rebuilt[1, 1](1))

        # pickle identity directly
        serialized_identity = pickle.dumps(identity)
        # memo size unchanged
        self.assertEqual(memo_size, len(memo))
        # unpickle
        identity_rebuilt = pickle.loads(serialized_identity)
        # must be the same object
        self.assertIs(identity, identity_rebuilt)
        # memo size unchanged
        self.assertEqual(memo_size, len(memo))

    def test_deserialization(self):
        """
        Test deserialization of intrinsic
        """

        def defn(context, x):
            def codegen(context, builder, signature, args):
                return args[0]

            return x(x), codegen

        memo = _Intrinsic._memo
        memo_size = len(memo)
        # invoke _Intrinsic indirectly to avoid registration which keeps an
        # internal reference inside the compiler
        original = _Intrinsic("foo", defn)
        self.assertIs(original._defn, defn)
        pickled = pickle.dumps(original)
        # by pickling, a new memo entry is created
        memo_size += 1
        self.assertEqual(memo_size, len(memo))
        del original  # remove original before unpickling

        # by deleting, the memo entry is NOT removed due to recent
        # function queue
        self.assertEqual(memo_size, len(memo))

        # Manually force clear of _recent queue
        _Intrinsic._recent.clear()
        memo_size -= 1
        self.assertEqual(memo_size, len(memo))

        rebuilt = pickle.loads(pickled)
        # verify that the rebuilt object is different
        self.assertIsNot(rebuilt._defn, defn)

        # the second rebuilt object is the same as the first
        second = pickle.loads(pickled)
        self.assertIs(rebuilt._defn, second._defn)

    def test_docstring(self):
        @intrinsic
        def void_func(typingctx, a: int):
            """void_func docstring"""
            sig = types.void(types.int32)

            def codegen(context, builder, signature, args):
                pass  # do nothing, return None, should be turned into
                # dummy value

            return sig, codegen

        self.assertEqual("numba_cuda_tests.cudapy.test_extending", void_func.__module__)
        self.assertEqual("void_func", void_func.__name__)
        self.assertEqual(
            "TestIntrinsic.test_docstring.<locals>.void_func",
            void_func.__qualname__,
        )
        self.assertDictEqual({"a": int}, inspect.get_annotations(void_func))
        self.assertEqual("void_func docstring", void_func.__doc__)


class TestRegisterJitable(NumbaCUDATestCase):
    def test_no_flags(self):
        @register_jitable
        def foo(x, y):
            x[0] += y

        extending.refresh_registries()

        def bar(x, y):
            foo(x, y)
            x[0] += x[0]

        cbar = cuda.jit(bar)

        x = np.array([1, 2])
        bar(x, 2)
        self.assertEqual(x[0], 6)
        cbar[1, 1](x, 2)
        self.assertEqual(x[0], 16)


class TestOverloadPreferLiteral(NumbaCUDATestCase):
    @pytest.mark.xfail(True, reason="ICE")
    def test_overload(self):
        def prefer_lit(x):
            pass

        def non_lit(x):
            pass

        def ov(x):
            if isinstance(x, types.IntegerLiteral):
                # With prefer_literal=False, this branch will not be reached.
                if x.literal_value == 1:

                    def impl(x):
                        return 0xCAFE

                    return impl
                else:
                    raise errors.TypingError("literal value")
            else:

                def impl(x):
                    return x * 100

                return impl

        overload(prefer_lit, prefer_literal=True)(ov)
        overload(non_lit)(ov)

        @cuda.jit
        def check_prefer_lit(x, res):
            res[0] = prefer_lit(1)
            res[1] = prefer_lit(2)
            res[2] = prefer_lit(x)

        res = np.zeros(3)
        check_prefer_lit[1, 1](3, res)
        a, b, c = res
        self.assertEqual(a, 0xCAFE)
        self.assertEqual(b, 200)
        self.assertEqual(c, 300)

        @cuda.jit
        def check_non_lit(x, res):
            res[0] = non_lit(1)
            res[1] = non_lit(2)
            res[2] = non_lit(x)

        check_non_lit[1, 1](3, res)
        a, b, c = res
        self.assertEqual(a, 100)
        self.assertEqual(b, 200)
        self.assertEqual(c, 300)


def test_overload_array_return():
    def slice_a(a):
        pass

    @overload(slice_a, inline="always")
    def slice_a_overload(a):
        def impl(a):
            return a[:, 0]

        return impl

    extending.refresh_registries()

    @numba_cuda_mlir.cuda.jit
    def add(a):
        s = slice_a(a)
        s[0] += 1.0

    a = np.array([[0.0], [1.0]], dtype=np.float32)
    a = cuda.to_device(a)
    add[1, 1](a)
    a = a.copy_to_host()

    assert a[0, 0] == 1


if __name__ == "__main__":
    unittest.main()
