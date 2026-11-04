# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest

from numba_cuda_mlir import cuda, extending, testing, types
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.mlir.dialect_exts import llvm
from numba_cuda_mlir.extending import StructProxy
from numba_cuda_mlir.models import StructModel, register_model


class PairType(types.Type):
    def __init__(self):
        super().__init__(name="StructProxyTestPair")


pair_type = PairType()


@register_model(PairType)
class PairModel(StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("x", types.int32), ("y", types.float32)])


def test_wraps_existing_value():
    @extending.intrinsic
    def roundtrip_x(typingctx, x):
        def codegen(builder, target, args, kwargs):
            struct_type = builder.get_mlir_type(pair_type)
            raw = llvm.mlir_zero(res=struct_type)
            raw = llvm.insertvalue(raw, builder.load_var(args[0]), [0])

            p = StructProxy(pair_type, raw)

            builder.store_var(target, llvm.extractvalue(T.i32(), p.value, [0]))

        return types.int32(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(a):
        a[1] = roundtrip_x(a[0])

    a = np.array([42, 0], dtype=np.int32)
    k[1, 1](a)

    assert a[1] == 42


def test_rejects_non_struct_model_type():
    @extending.intrinsic
    def wrap_int(typingctx, x):
        def codegen(builder, target, args, kwargs):
            StructProxy(types.int32, builder.load_var(args[0]))

        return types.void(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(a):
        wrap_int(a[0])

    with pytest.raises(TypeError, match=r"StructModel.*int32"):
        k[1, 1](np.zeros(1, dtype=np.int32))


def test_rejects_value_of_wrong_type():
    @extending.intrinsic
    def wrap_wrong_value(typingctx, x):
        def codegen(builder, target, args, kwargs):
            StructProxy(pair_type, builder.load_var(args[0]))

        return types.void(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(a):
        wrap_wrong_value(a[0])

    with pytest.raises(TypeError, match=r"StructProxyTestPair.*i32"):
        k[1, 1](np.zeros(1, dtype=np.int32))


def test_new_struct_is_zero_by_default():
    @extending.intrinsic
    def zero_pair_y(typingctx, x):
        def codegen(builder, target, args, kwargs):
            p = StructProxy(pair_type)

            builder.store_var(target, llvm.extractvalue(T.f32(), p.value, [1]))

        return types.float32(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(a):
        a[0] = zero_pair_y(a[0])

    a = np.array([5.0], dtype=np.float32)
    k[1, 1](a)

    assert a[0] == 0.0

    mlir = next(iter(k.inspect_mlir().values()))
    # CHECK: llvm.mlir.zero : !llvm.struct<(i32, f32)>
    testing.filecheck_with_comments(mlir)


def test_new_struct_from_undef():
    @extending.intrinsic
    def undef_pair(typingctx, x):
        def codegen(builder, target, args, kwargs):
            StructProxy(pair_type, init="undef")

        return types.void(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(a):
        undef_pair(a[0])

    k[1, 1](np.zeros(1, dtype=np.int32))

    mlir = next(iter(k.inspect_mlir().values()))
    # CHECK: llvm.mlir.undef : !llvm.struct<(i32, f32)>
    testing.filecheck_with_comments(mlir)


@pytest.mark.parametrize(
    "init",
    [
        pytest.param("garbage", id="unknown"),
        pytest.param("", id="empty"),
    ],
)
def test_rejects_unknown_init(init):
    @extending.intrinsic
    def bad_init(typingctx, x):
        def codegen(builder, target, args, kwargs):
            StructProxy(pair_type, init=init)

        return types.void(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(a):
        bad_init(a[0])

    with pytest.raises(ValueError, match=rf"init.*{init!r}"):
        k[1, 1](np.zeros(1, dtype=np.int32))


def test_rejects_init_with_existing_value():
    @extending.intrinsic
    def init_and_value(typingctx, x):
        def codegen(builder, target, args, kwargs):
            existing = StructProxy(pair_type).value
            StructProxy(pair_type, existing, init="undef")

        return types.void(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(a):
        init_and_value(a[0])

    with pytest.raises(ValueError, match=r"init"):
        k[1, 1](np.zeros(1, dtype=np.int32))


def test_read_fields_by_name():
    @extending.intrinsic
    def pair_x(typingctx, x, y):
        def codegen(builder, target, args, kwargs):
            raw = llvm.mlir_zero(res=builder.get_mlir_type(pair_type))
            raw = llvm.insertvalue(raw, builder.load_var(args[0]), [0])
            raw = llvm.insertvalue(raw, builder.load_var(args[1]), [1])

            builder.store_var(target, StructProxy(pair_type, raw).x)

        return types.int32(x, y), codegen

    @extending.intrinsic
    def pair_y(typingctx, x, y):
        def codegen(builder, target, args, kwargs):
            raw = llvm.mlir_zero(res=builder.get_mlir_type(pair_type))
            raw = llvm.insertvalue(raw, builder.load_var(args[0]), [0])
            raw = llvm.insertvalue(raw, builder.load_var(args[1]), [1])

            builder.store_var(target, StructProxy(pair_type, raw).y)

        return types.float32(x, y), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(xs, ys):
        xs[1] = pair_x(xs[0], ys[0])
        ys[1] = pair_y(xs[0], ys[0])

    xs = np.array([42, 0], dtype=np.int32)
    ys = np.array([2.5, 0.0], dtype=np.float32)
    k[1, 1](xs, ys)

    assert xs[1] == 42
    assert ys[1] == np.float32(2.5)

    mlir = next(iter(k.inspect_mlir().values()))
    # CHECK: llvm.extractvalue {{.*}}[0] : !llvm.struct<(i32, f32)>
    # CHECK: llvm.extractvalue {{.*}}[1] : !llvm.struct<(i32, f32)>
    testing.filecheck_with_comments(mlir)


def test_read_fields_of_new_zero_struct():
    @extending.intrinsic
    def zero_x(typingctx, x):
        def codegen(builder, target, args, kwargs):
            builder.store_var(target, StructProxy(pair_type, init="zero").x)

        return types.int32(x), codegen

    @extending.intrinsic
    def zero_y(typingctx, x):
        def codegen(builder, target, args, kwargs):
            builder.store_var(target, StructProxy(pair_type, init="zero").y)

        return types.float32(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(xs, ys):
        xs[0] = zero_x(xs[0])
        ys[0] = zero_y(ys[0])

    xs = np.array([7], dtype=np.int32)
    ys = np.array([7.0], dtype=np.float32)
    k[1, 1](xs, ys)

    assert xs[0] == 0
    assert ys[0] == 0.0


def test_unknown_field_lists_valid_fields():
    @extending.intrinsic
    def read_z(typingctx, x):
        def codegen(builder, target, args, kwargs):
            StructProxy(pair_type).z

        return types.void(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(a):
        read_z(a[0])

    with pytest.raises(AttributeError, match=r"'z'.*\('x', 'y'\)"):
        k[1, 1](np.zeros(1, dtype=np.int32))


def test_private_names_are_not_fields():
    @extending.intrinsic
    def read_private(typingctx, x):
        def codegen(builder, target, args, kwargs):
            StructProxy(pair_type)._missing

        return types.void(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(a):
        read_private(a[0])

    with pytest.raises(AttributeError, match=r"_missing"):
        k[1, 1](np.zeros(1, dtype=np.int32))


def test_build_struct_with_replace():
    @extending.intrinsic
    def make_pair(typingctx, x, y):
        def codegen(builder, target, args, kwargs):
            x, y = builder.load_vars(args)
            p = StructProxy(pair_type).replace(x=x, y=y)

            builder.store_var(target, p.value)

        return pair_type(x, y), codegen

    @extending.intrinsic
    def pair_x(typingctx, pair):
        def codegen(builder, target, args, kwargs):
            builder.store_var(target, StructProxy(pair_type, builder.load_var(args[0])).x)

        return types.int32(pair), codegen

    @extending.intrinsic
    def pair_y(typingctx, pair):
        def codegen(builder, target, args, kwargs):
            builder.store_var(target, StructProxy(pair_type, builder.load_var(args[0])).y)

        return types.float32(pair), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(xs, ys):
        p = make_pair(xs[0], ys[0])
        xs[1] = pair_x(p)
        ys[1] = pair_y(p)

    xs = np.array([42, 0], dtype=np.int32)
    ys = np.array([2.5, 0.0], dtype=np.float32)
    k[1, 1](xs, ys)

    assert xs[1] == 42
    assert ys[1] == np.float32(2.5)

    mlir = next(iter(k.inspect_mlir().values()))
    # CHECK: llvm.mlir.zero : !llvm.struct<(i32, f32)>
    # CHECK: llvm.insertvalue {{.*}}[0] : !llvm.struct<(i32, f32)>
    # CHECK: llvm.insertvalue {{.*}}[1] : !llvm.struct<(i32, f32)>
    testing.filecheck_with_comments(mlir)


def test_build_struct_from_undef_with_replace():
    @extending.intrinsic
    def make_pair(typingctx, x, y):
        def codegen(builder, target, args, kwargs):
            x, y = builder.load_vars(args)
            p = StructProxy(pair_type, init="undef").replace(x=x, y=y)

            builder.store_var(target, p.value)

        return pair_type(x, y), codegen

    @extending.intrinsic
    def pair_x(typingctx, pair):
        def codegen(builder, target, args, kwargs):
            builder.store_var(target, StructProxy(pair_type, builder.load_var(args[0])).x)

        return types.int32(pair), codegen

    @extending.intrinsic
    def pair_y(typingctx, pair):
        def codegen(builder, target, args, kwargs):
            builder.store_var(target, StructProxy(pair_type, builder.load_var(args[0])).y)

        return types.float32(pair), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(xs, ys):
        p = make_pair(xs[0], ys[0])
        xs[1] = pair_x(p)
        ys[1] = pair_y(p)

    xs = np.array([42, 0], dtype=np.int32)
    ys = np.array([2.5, 0.0], dtype=np.float32)
    k[1, 1](xs, ys)

    assert xs[1] == 42
    assert ys[1] == np.float32(2.5)

    mlir = next(iter(k.inspect_mlir().values()))
    # CHECK: llvm.mlir.undef : !llvm.struct<(i32, f32)>
    # CHECK: llvm.insertvalue {{.*}}[0] : !llvm.struct<(i32, f32)>
    # CHECK: llvm.insertvalue {{.*}}[1] : !llvm.struct<(i32, f32)>
    testing.filecheck_with_comments(mlir)


def test_replace_leaves_original_unchanged():
    @extending.intrinsic
    def original_x(typingctx, x):
        def codegen(builder, target, args, kwargs):
            p = StructProxy(pair_type)
            p.replace(x=builder.load_var(args[0]))

            builder.store_var(target, p.x)

        return types.int32(x), codegen

    @extending.intrinsic
    def replaced_x(typingctx, x):
        def codegen(builder, target, args, kwargs):
            p = StructProxy(pair_type)
            q = p.replace(x=builder.load_var(args[0]))

            builder.store_var(target, q.x)

        return types.int32(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(a):
        a[1] = original_x(a[0])
        a[2] = replaced_x(a[0])

    a = np.array([42, 7, 7], dtype=np.int32)
    k[1, 1](a)

    assert a[1] == 0
    assert a[2] == 42


def test_replace_rejects_unknown_field():
    @extending.intrinsic
    def replace_z(typingctx, x):
        def codegen(builder, target, args, kwargs):
            StructProxy(pair_type).replace(z=builder.load_var(args[0]))

        return types.void(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(a):
        replace_z(a[0])

    with pytest.raises(AttributeError, match=r"'z'.*\('x', 'y'\)"):
        k[1, 1](np.zeros(1, dtype=np.int32))


def test_replace_rejects_mismatched_field_type():
    @extending.intrinsic
    def replace_x_with_float(typingctx, x):
        def codegen(builder, target, args, kwargs):
            StructProxy(pair_type).replace(x=builder.load_var(args[0]))

        return types.void(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(a):
        replace_x_with_float(a[0])

    with pytest.raises(TypeError, match=r"field 'x'.*i32.*f32"):
        k[1, 1](np.zeros(1, dtype=np.float32))


def test_replace_rejects_non_mlir_value():
    @extending.intrinsic
    def replace_x_with_int(typingctx, x):
        def codegen(builder, target, args, kwargs):
            StructProxy(pair_type).replace(x=7)

        return types.void(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(a):
        replace_x_with_int(a[0])

    with pytest.raises(TypeError, match=r"field 'x'.*MLIR value.*int"):
        k[1, 1](np.zeros(1, dtype=np.int32))


def test_setting_field_is_rejected_with_replace_hint():
    @extending.intrinsic
    def set_x(typingctx, x):
        def codegen(builder, target, args, kwargs):
            p = StructProxy(pair_type)
            p.x = builder.load_var(args[0])

        return types.void(x), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(a):
        set_x(a[0])

    with pytest.raises(AttributeError, match=r"immutable.*replace\(x="):
        k[1, 1](np.zeros(1, dtype=np.int32))


class OddNamesType(types.Type):
    def __init__(self):
        super().__init__(name="StructProxyTestOddNames")


odd_names_type = OddNamesType()


@register_model(OddNamesType)
class OddNamesModel(StructModel):
    def __init__(self, dmm, fe_type):
        members = [("value", types.int32), ("self", types.int32), ("_hidden", types.float32)]
        super().__init__(dmm, fe_type, members)


def test_read_and_replace_fields_shadowed_by_api_names():
    @extending.intrinsic
    def make_odd(typingctx, v, s, h):
        def codegen(builder, target, args, kwargs):
            v, s, h = builder.load_vars(args)
            p = StructProxy(odd_names_type).replace(value=v, self=s, _hidden=h)

            builder.store_var(target, p.value)

        return odd_names_type(v, s, h), codegen

    @extending.intrinsic
    def read_value(typingctx, odd):
        def codegen(builder, target, args, kwargs):
            p = StructProxy(odd_names_type, builder.load_var(args[0]))

            builder.store_var(target, p.read("value"))

        return types.int32(odd), codegen

    @extending.intrinsic
    def read_self(typingctx, odd):
        def codegen(builder, target, args, kwargs):
            p = StructProxy(odd_names_type, builder.load_var(args[0]))

            builder.store_var(target, p.read("self"))

        return types.int32(odd), codegen

    @extending.intrinsic
    def read_hidden(typingctx, odd):
        def codegen(builder, target, args, kwargs):
            p = StructProxy(odd_names_type, builder.load_var(args[0]))

            builder.store_var(target, p.read("_hidden"))

        return types.float32(odd), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(ints, floats):
        p = make_odd(ints[0], ints[1], floats[0])
        ints[2] = read_value(p)
        ints[3] = read_self(p)
        floats[1] = read_hidden(p)

    ints = np.array([42, 7, 0, 0], dtype=np.int32)
    floats = np.array([2.5, 0.0], dtype=np.float32)
    k[1, 1](ints, floats)

    assert ints[2] == 42
    assert ints[3] == 7
    assert floats[1] == np.float32(2.5)


class FlagType(types.Type):
    def __init__(self):
        super().__init__(name="StructProxyTestFlag")


flag_type = FlagType()


@register_model(FlagType)
class FlagModel(StructModel):
    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, [("flag", types.boolean), ("n", types.int32)])


def test_bool_field_uses_value_type():
    @extending.intrinsic
    def make_flag(typingctx, flag, n):
        def codegen(builder, target, args, kwargs):
            flag, n = builder.load_vars(args)
            p = StructProxy(flag_type).replace(flag=flag, n=n)

            builder.store_var(target, p.value)

        return flag_type(flag, n), codegen

    @extending.intrinsic
    def flag_of(typingctx, f):
        def codegen(builder, target, args, kwargs):
            builder.store_var(target, StructProxy(flag_type, builder.load_var(args[0])).flag)

        return types.boolean(f), codegen

    @extending.intrinsic
    def n_of(typingctx, f):
        def codegen(builder, target, args, kwargs):
            builder.store_var(target, StructProxy(flag_type, builder.load_var(args[0])).n)

        return types.int32(f), codegen

    extending.refresh_registries()

    @cuda.jit
    def k(flags, ns):
        p = make_flag(flags[0], ns[0])
        flags[1] = flag_of(p)
        ns[1] = n_of(p)

    flags = np.array([True, False])
    ns = np.array([42, 0], dtype=np.int32)
    k[1, 1](flags, ns)

    assert flags[1]
    assert ns[1] == 42

    mlir = next(iter(k.inspect_mlir().values()))
    # CHECK: llvm.mlir.zero : !llvm.struct<(i1, i32)>
    # CHECK: llvm.insertvalue {{.*}}[0] : !llvm.struct<(i1, i32)>
    testing.filecheck_with_comments(mlir)
