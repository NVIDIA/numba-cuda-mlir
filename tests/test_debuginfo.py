# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from enum import IntEnum

import pytest
from numba_cuda_mlir import cuda
from numba_cuda_mlir import types, compiler, testing
from numba_cuda_mlir.numba_cuda.types.ext_types import bfloat16
from numba_cuda_mlir.numba_cuda.np import numpy_support


def k_scalar_add(out, a, b):
    out[0] = a + b


def k_local_var(out, a, b):
    c = a + b
    out[0] = c


def k_bool_kernel(out, flag):
    if flag:
        out[0] = 1
    else:
        out[0] = 0


def k_array_2d(a):
    i = cuda.threadIdx.x
    a[0, i] = a[0, i] + 1.0


def test_mlir_emission_kind_full():
    """debug=True must set emissionKind = Full (not LineTablesOnly)."""
    mlir = compiler.compile_mlir(
        k_scalar_add,
        types.void(types.int32[:], types.int32, types.int32),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: emissionKind = Full
        CHECK-NOT: emissionKind = LineTablesOnly
        """,
        mlir,
    )


def test_mlir_emission_kind_debug_directives_only():
    """lineinfo=True must set emissionKind = DebugDirectivesOnly (not Full)."""
    sig = types.void(types.int32[:], types.int32, types.int32)
    mlir = compiler.compile_mlir(k_scalar_add, sig, lineinfo=True, opt=False)
    testing.filecheck(
        """
        CHECK: emissionKind = DebugDirectivesOnly
        CHECK-NOT: emissionKind = Full
        """,
        mlir,
    )


def test_mlir_no_debug_info_default():
    """Default compilation (no debug/lineinfo) must not emit DI attributes or dbg.value."""
    sig = types.void(types.int32[:], types.int32, types.int32)
    mlir = compiler.compile_mlir(k_scalar_add, sig, opt=False)
    testing.filecheck(
        """
        CHECK-NOT: emissionKind = Full
        CHECK-NOT: llvm.intr.dbg.value
        CHECK-NOT: di_local_variable
        """,
        mlir,
    )


def test_mlir_di_basic_type_int():
    """debug=True emits DIBasicType with int32 for int32 arguments."""
    mlir = compiler.compile_mlir(
        k_scalar_add,
        types.void(types.int32[:], types.int32, types.int32),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: name = "int32"
        CHECK-SAME: sizeInBits = 32
        CHECK-SAME: encoding = DW_ATE_signed
        """,
        mlir,
    )


def test_mlir_di_basic_type_float():
    """debug=True emits DIBasicType with float64 for float64 arguments."""
    mlir = compiler.compile_mlir(
        k_scalar_add,
        types.void(types.float64[:], types.float64, types.float64),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: name = "float64"
        CHECK-SAME: sizeInBits = 64
        CHECK-SAME: encoding = DW_ATE_float
        """,
        mlir,
    )


def test_mlir_di_basic_type_bool():
    """debug=True emits DIBasicType with bool for boolean arguments."""
    mlir = compiler.compile_mlir(
        k_bool_kernel,
        types.void(types.int32[:], types.boolean),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: name = "bool"
        CHECK-SAME: sizeInBits = 8
        CHECK-SAME: encoding = DW_ATE_boolean
        """,
        mlir,
    )


def test_mlir_di_local_variable_args():
    """Function arguments appear as DILocalVariable with arg = N."""
    mlir = compiler.compile_mlir(
        k_scalar_add,
        types.void(types.int32[:], types.int32, types.int32),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: name = "a"
        CHECK-SAME: arg = 2
        CHECK: name = "b"
        CHECK-SAME: arg = 3
        """,
        mlir,
    )


def test_mlir_di_local_variable_locals():
    """Assigned local variables appear as DILocalVariable without arg."""
    mlir = compiler.compile_mlir(
        k_local_var,
        types.void(types.int32[:], types.int32, types.int32),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: di_local_variable<{{.*}}name = "c"
        CHECK-NOT: arg =
        CHECK-SAME: type =
        """,
        mlir,
    )


def test_mlir_dbg_value_ops():
    """debug=True emits llvm.intr.dbg.value for local variables."""
    mlir = compiler.compile_mlir(
        k_local_var,
        types.void(types.int32[:], types.int32, types.int32),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: llvm.intr.dbg.value
        """,
        mlir,
    )


def test_mlir_dbg_declare_ops():
    """Scalar (non-boolean) args use dbg.declare on a stack alloca."""
    mlir = compiler.compile_mlir(
        k_scalar_add,
        types.void(types.int32[:], types.int32, types.int32),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: llvm.intr.dbg.declare
        """,
        mlir,
    )


def test_mlir_dbg_value_bool_arg():
    """Boolean args use dbg.value (not dbg.declare) due to NVVM workaround."""
    mlir = compiler.compile_mlir(
        k_bool_kernel,
        types.void(types.int32[:], types.boolean),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: #[[FLAG_VAR:di_local_variable[0-9]*]] = #llvm.di_local_variable<{{.*}}name = "flag"
        CHECK: llvm.intr.dbg.value #[[FLAG_VAR]] = %{{.*}} : i1
        """,
        mlir,
    )


def test_mlir_grid_group_type():
    """Emits GridGroup as an opaque 64-bit unsigned DI basic type."""

    def k_grid_group_sync(out):
        grid = cuda.cg.this_grid()
        out[0] = grid.sync()

    mlir = compiler.compile_mlir(
        k_grid_group_sync,
        types.void(types.int32[:]),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: di_basic_type<tag = DW_TAG_base_type, name = "GridGroup"
        CHECK-SAME: sizeInBits = 64
        CHECK-SAME: encoding = DW_ATE_unsigned
        """,
        mlir,
    )


def test_mlir_cpointer_type():
    """Emits pointer DI with int32 pointee for CPointer(int32)."""

    def k_cpointer(p):
        i = cuda.threadIdx.x
        p[i] = i

    mlir = compiler.compile_mlir(
        k_cpointer,
        types.void(types.CPointer(types.int32)),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: di_basic_type<tag = DW_TAG_base_type, name = "int32"
        CHECK-SAME: sizeInBits = 32
        CHECK-SAME: encoding = DW_ATE_signed
        CHECK: di_derived_type<tag = DW_TAG_pointer_type
        CHECK-SAME: baseType = #di_basic_type
        CHECK-SAME: sizeInBits = 64
        """,
        mlir,
    )


def test_mlir_enummember_type():
    """debug=True emits stable scalar DI for EnumMember locals."""

    class Color(IntEnum):
        RED = 0
        GREEN = 1
        BLUE = 2

    def k_enum(out):
        i = cuda.threadIdx.x
        c = Color.GREEN
        out[i] = c.value

    mlir = compiler.compile_mlir(
        k_enum,
        types.void(types.int32[::1]),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: di_basic_type<tag = DW_TAG_base_type, name = "IntEnum<int64>(Color)"
        CHECK-SAME: sizeInBits = 64
        CHECK-SAME: encoding = DW_ATE_signed
        """,
        mlir,
    )


@pytest.mark.parametrize(
    "arg_type, expected_name",
    [
        (types.NPDatetime("ms")[::1], "datetime64[ms]"),
        (types.NPTimedelta("ms")[::1], "timedelta64[ms]"),
    ],
)
def test_mlir_named_scalar_type(arg_type, expected_name):
    """Emits signed 64-bit DI basic type for datetime64/timedelta64 units."""

    def k_named_scalar(arg):
        i = cuda.threadIdx.x
        x = arg[i]  # noqa: F841

    mlir = compiler.compile_mlir(
        k_named_scalar,
        types.void(arg_type),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        f"""
        CHECK: di_basic_type<tag = DW_TAG_base_type, name = "{expected_name}"
        CHECK-SAME: sizeInBits = 64
        CHECK-SAME: encoding = DW_ATE_signed
        """,
        mlir,
    )


def test_mlir_bfloat16_type():
    """Emits __nv_bfloat16 as a 16-bit float DI type."""

    def k_bfloat16(a, b, out):
        i = cuda.threadIdx.x
        c = a[i] + b[i]
        out[i] = c

    mlir = compiler.compile_mlir(
        k_bfloat16,
        types.void(bfloat16[::1], bfloat16[::1], bfloat16[::1]),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: di_basic_type<tag = DW_TAG_base_type, name = "__nv_bfloat16"
        CHECK-SAME: sizeInBits = 16
        CHECK-SAME: encoding = DW_ATE_float
        """,
        mlir,
    )


def k_complex_add(a, b):
    c = a + b
    return c


def test_mlir_fusedloc_tags_complex():
    """Complex vars are tagged for deferred dbg.declare emission."""
    mlir = compiler.compile_mlir(
        k_complex_add,
        (types.complex64, types.complex64),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: loc("dbg_var:a")
        CHECK: loc("dbg_var:b")
        CHECK: loc("dbg_var:c")
        """,
        mlir,
    )


def test_mlir_deferred_dbg_declare_complex():
    """Deferred pass emits dbg.declare and complex DI type."""
    optimized_mlir = compiler.compile_mlir(
        k_complex_add,
        (types.complex128, types.complex128),
        optimized=True,
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: loc(
        CHECK-NOT: loc("dbg_var:
        CHECK: di_derived_type<tag = DW_TAG_member, name = "real"{{.*}}sizeInBits = 64
        CHECK: di_derived_type<tag = DW_TAG_member, name = "imag"{{.*}}sizeInBits = 64, offsetInBits = 64
        CHECK: di_composite_type<tag = DW_TAG_structure_type, name = "complex128", sizeInBits = 128
        CHECK: di_local_variable<{{.*}}name = "a"{{.*}}type = #di_composite_type
        CHECK: di_local_variable<{{.*}}name = "b"{{.*}}type = #di_composite_type
        CHECK: di_local_variable<{{.*}}name = "c"{{.*}}type = #di_composite_type
        CHECK-COUNT-3: llvm.intr.dbg.declare
        """,
        optimized_mlir,
    )


def test_mlir_mixed_complex_scalar():
    """Regular and deferred emission paths must share same di_subprogram scope."""

    def k_mixed(out, scale, z1, z2, flag):
        result = z1 + z2
        scaled_real = scale * result.real
        if flag:
            out[0] = scaled_real

    optimized_mlir = compiler.compile_mlir(
        k_mixed,
        (
            types.float32[:],
            types.int32,
            types.complex64,
            types.complex64,
            types.boolean,
        ),
        optimized=True,
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK-COUNT-1: = #llvm.di_subprogram<
        CHECK-DAG: name = "scale"
        CHECK-DAG: name = "z1"
        CHECK-DAG: name = "z2"
        CHECK-DAG: name = "flag"
        CHECK-DAG: name = "result"
        """,
        optimized_mlir,
    )


def test_mlir_unituple_type():
    """UniTuple local uses dbg.declare and llvm.di_composite_type with DW_TAG_array_type."""

    def k_tuple_uniform(out, a, b, c):
        i = cuda.threadIdx.x
        t = (a, b, c)
        out[i] = t[0] + t[1] + t[2]

    mlir = compiler.compile_mlir(
        k_tuple_uniform,
        types.void(types.int64[::1], types.int64, types.int64, types.int64),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: #[[TUPLE_TYPE:di_composite_type[0-9]*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, name = "UniTuple(int64 x 3) ([3 x i64])"
        CHECK-SAME: elements = #llvm.di_subrange<count = 3 : i64>
        CHECK: #[[TUPLE_VAR:di_local_variable[0-9]*]] = #llvm.di_local_variable<{{.*}}name = "t"
        CHECK-SAME: type = #[[TUPLE_TYPE]]
        CHECK: llvm.intr.dbg.declare #[[TUPLE_VAR]] = %{{[0-9]+}} : !llvm.ptr
        """,
        mlir,
    )


def test_mlir_basetuple_type():
    """Base tuple local uses dbg.declare and llvm.di_composite_type with DW_TAG_structure_type."""

    def k_tuple_hetero(out, a, b):
        i = cuda.threadIdx.x
        t = (a, b)
        out[i] = t[0] + int(t[1])

    mlir = compiler.compile_mlir(
        k_tuple_hetero,
        types.void(types.int64[::1], types.int64, types.float64),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: #[[TUPLE_MEMBER0:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "f0"
        CHECK-SAME: sizeInBits = 64
        CHECK: #[[TUPLE_MEMBER1:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "f1"
        CHECK-SAME: sizeInBits = 64
        CHECK-SAME: offsetInBits = 64
        CHECK: #[[TUPLE_TYPE:di_composite_type[0-9]*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "Tuple(int64, float64) ({i64, double})"
        CHECK-SAME: elements = #[[TUPLE_MEMBER0]], #[[TUPLE_MEMBER1]]
        CHECK: #[[TUPLE_VAR:di_local_variable[0-9]*]] = #llvm.di_local_variable<{{.*}}name = "t"
        CHECK-SAME: type = #[[TUPLE_TYPE]]
        CHECK: llvm.intr.dbg.declare #[[TUPLE_VAR]] = %{{[0-9]+}} : !llvm.ptr
        """,
        mlir,
    )


def test_mlir_basetuple_type_with_alignment_padding():
    """Base tuple size and member offsets include LLVM struct alignment padding."""

    def k_tuple_padded(out, a, b, c):
        i = cuda.threadIdx.x
        t = (a, b, c)
        out[i] = t[0] + int(t[1]) + int(t[2])

    mlir = compiler.compile_mlir(
        k_tuple_padded,
        types.void(types.int64[::1], types.int32, types.float64, types.boolean),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: #[[TUPLE_MEMBER0:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "f0"
        CHECK-SAME: sizeInBits = 32
        CHECK: #[[TUPLE_MEMBER1:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "f1"
        CHECK-SAME: sizeInBits = 64
        CHECK-SAME: offsetInBits = 64
        CHECK: #[[TUPLE_MEMBER2:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "f2"
        CHECK-SAME: sizeInBits = 8
        CHECK-SAME: offsetInBits = 128
        CHECK: #[[TUPLE_TYPE:di_composite_type[0-9]*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "Tuple(int32, float64, bool) ({i32, double, i8})"
        CHECK-SAME: sizeInBits = 192
        CHECK-SAME: elements = #[[TUPLE_MEMBER0]], #[[TUPLE_MEMBER1]], #[[TUPLE_MEMBER2]]
        CHECK: #[[TUPLE_VAR:di_local_variable[0-9]*]] = #llvm.di_local_variable<{{.*}}name = "t"
        CHECK-SAME: type = #[[TUPLE_TYPE]]
        CHECK: llvm.intr.dbg.declare #[[TUPLE_VAR]] = %{{[0-9]+}} : !llvm.ptr
        """,
        mlir,
    )


def test_mlir_record_type():
    """Record local uses dbg.declare and llvm.di_composite_type with DW_TAG_structure_type."""

    record_dtype = np.dtype([("a", np.int32), ("b", np.float64)], align=True)
    record_type = numpy_support.from_dtype(record_dtype)

    def k_record_local(records, out):
        i = cuda.threadIdx.x
        r = records[i]
        out[i] = r.a + int(r.b)

    mlir = compiler.compile_mlir(
        k_record_local,
        types.void(types.Array(record_type, 1, "C"), types.int64[::1]),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: #[[RECORD_TYPE:di_composite_type[0-9]*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "Record(a[type=int32;offset=0],b[type=float64;offset=8];16;True)"
        CHECK: #[[RECORD_VAR:di_local_variable[0-9]*]] = #llvm.di_local_variable<{{.*}}name = "r"
        CHECK-SAME: type = #[[RECORD_TYPE]]
        CHECK: llvm.intr.dbg.declare #[[RECORD_VAR]] = %{{[0-9]+}} : !llvm.ptr
        """,
        mlir,
    )


def test_mlir_array_1d_kernel_arg_type():
    """Array kernel args emit dbg.declare with descriptor struct DI."""

    def k_array_1d(a):
        i = cuda.threadIdx.x
        a[i] = a[i] + 1.0

    mlir = compiler.compile_mlir(
        k_array_1d,
        types.void(types.float32[::1]),
        optimized=True,
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: #[[TUPLE_TYPE:di_composite_type[0-9]*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, name = "UniTuple(int64 x 1) ([1 x i64])"
        CHECK-SAME: sizeInBits = 64
        CHECK: #[[ITEMSIZE:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "itemsize"
        CHECK-SAME: offsetInBits = 192
        CHECK: #[[DATA:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "data"
        CHECK-SAME: offsetInBits = 256
        CHECK: #[[SHAPE:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "shape"
        CHECK-SAME: baseType = #[[TUPLE_TYPE]]
        CHECK-SAME: offsetInBits = 320
        CHECK: #[[STRIDES:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "strides"
        CHECK-SAME: baseType = #[[TUPLE_TYPE]]
        CHECK-SAME: offsetInBits = 384
        CHECK: #[[ARRAY_TYPE:di_composite_type[0-9]*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "array(float32, 1d, C) ({i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]})"
        CHECK-SAME: sizeInBits = 448
        CHECK-SAME: elements = {{.*}}#[[ITEMSIZE]], #[[DATA]], #[[SHAPE]], #[[STRIDES]]
        CHECK: #[[ARRAY_VAR:di_local_variable[0-9]*]] = #llvm.di_local_variable<{{.*}}name = "a"
        CHECK-SAME: type = #[[ARRAY_TYPE]]
        CHECK: llvm.intr.dbg.declare #[[ARRAY_VAR]] = %{{[0-9]+}} : !llvm.ptr
        """,
        mlir,
    )


def test_mlir_array_2d_c_order_kernel_arg_type():
    """C-order 2d array kernel args size shape/strides by ndim."""

    mlir = compiler.compile_mlir(
        k_array_2d,
        types.void(types.float32[:, ::1]),
        optimized=True,
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: #[[TUPLE_TYPE:di_composite_type[0-9]*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, name = "UniTuple(int64 x 2) ([2 x i64])"
        CHECK-SAME: sizeInBits = 128
        CHECK: #[[SHAPE:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "shape"
        CHECK-SAME: baseType = #[[TUPLE_TYPE]]
        CHECK-SAME: sizeInBits = 128
        CHECK-SAME: offsetInBits = 320
        CHECK: #[[STRIDES:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "strides"
        CHECK-SAME: baseType = #[[TUPLE_TYPE]]
        CHECK-SAME: sizeInBits = 128
        CHECK-SAME: offsetInBits = 448
        CHECK: #[[ARRAY_TYPE:di_composite_type[0-9]*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "array(float32, 2d, C) ({i8*, i8*, i64, i64, float*, [2 x i64], [2 x i64]})"
        CHECK-SAME: sizeInBits = 576
        CHECK: #[[ARRAY_VAR:di_local_variable[0-9]*]] = #llvm.di_local_variable<{{.*}}name = "a"
        CHECK-SAME: type = #[[ARRAY_TYPE]]
        CHECK: %[[ITEMSIZE:[0-9]+]] = llvm.mlir.constant(4 : i64) : i64
        CHECK: %[[STRIDE0_BYTES:[0-9]+]] = llvm.mul %arg5, %[[ITEMSIZE]] : i64
        CHECK: %[[STRIDES0:[0-9]+]] = llvm.insertvalue %[[STRIDE0_BYTES]], %{{[0-9]+}}[0] : !llvm.array<2 x i64>
        CHECK: %[[STRIDE1_BYTES:[0-9]+]] = llvm.mul %arg6, %[[ITEMSIZE]] : i64
        CHECK: %[[STRIDES:[0-9]+]] = llvm.insertvalue %[[STRIDE1_BYTES]], %[[STRIDES0]][1] : !llvm.array<2 x i64>
        CHECK: %[[DESC:[0-9]+]] = llvm.insertvalue %[[STRIDES]], %{{[0-9]+}}[6] : !llvm.struct<(ptr, ptr, i64, i64, ptr, array<2 x i64>, array<2 x i64>)>
        CHECK: llvm.intr.dbg.declare #[[ARRAY_VAR]] = %{{[0-9]+}} : !llvm.ptr
        """,
        mlir,
    )


def test_mlir_array_2d_f_order_kernel_arg_type():
    """F-order 2d array kernel args size shape/strides by ndim."""

    mlir = compiler.compile_mlir(
        k_array_2d,
        types.void(types.float32[::1, :]),
        optimized=True,
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: #[[TUPLE_TYPE:di_composite_type[0-9]*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, name = "UniTuple(int64 x 2) ([2 x i64])"
        CHECK-SAME: sizeInBits = 128
        CHECK: #[[SHAPE:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "shape"
        CHECK-SAME: baseType = #[[TUPLE_TYPE]]
        CHECK-SAME: sizeInBits = 128
        CHECK-SAME: offsetInBits = 320
        CHECK: #[[STRIDES:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "strides"
        CHECK-SAME: baseType = #[[TUPLE_TYPE]]
        CHECK-SAME: sizeInBits = 128
        CHECK-SAME: offsetInBits = 448
        CHECK: #[[ARRAY_TYPE:di_composite_type[0-9]*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "array(float32, 2d, F) ({i8*, i8*, i64, i64, float*, [2 x i64], [2 x i64]})"
        CHECK-SAME: sizeInBits = 576
        CHECK: #[[ARRAY_VAR:di_local_variable[0-9]*]] = #llvm.di_local_variable<{{.*}}name = "a"
        CHECK-SAME: type = #[[ARRAY_TYPE]]
        CHECK: %[[ITEMSIZE:[0-9]+]] = llvm.mlir.constant(4 : i64) : i64
        CHECK: %[[STRIDE0_BYTES:[0-9]+]] = llvm.mul %arg5, %[[ITEMSIZE]] : i64
        CHECK: %[[STRIDES0:[0-9]+]] = llvm.insertvalue %[[STRIDE0_BYTES]], %{{[0-9]+}}[0] : !llvm.array<2 x i64>
        CHECK: %[[STRIDE1_BYTES:[0-9]+]] = llvm.mul %arg6, %[[ITEMSIZE]] : i64
        CHECK: %[[STRIDES:[0-9]+]] = llvm.insertvalue %[[STRIDE1_BYTES]], %[[STRIDES0]][1] : !llvm.array<2 x i64>
        CHECK: %[[DESC:[0-9]+]] = llvm.insertvalue %[[STRIDES]], %{{[0-9]+}}[6] : !llvm.struct<(ptr, ptr, i64, i64, ptr, array<2 x i64>, array<2 x i64>)>
        CHECK: llvm.intr.dbg.declare #[[ARRAY_VAR]] = %{{[0-9]+}} : !llvm.ptr
        """,
        mlir,
    )


def test_mlir_local_array_type():
    """cuda.local.array locals use descriptor struct DI over a debug descriptor slot."""

    def k_local_array(out):
        tmp = cuda.local.array((4,), types.float32)
        i = cuda.threadIdx.x
        tmp[0] = out[i]
        out[i] = tmp[0]

    mlir = compiler.compile_mlir(
        k_local_array,
        types.void(types.float32[::1]),
        optimized=True,
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK: #[[TUPLE_TYPE:di_composite_type[0-9]*]] = #llvm.di_composite_type<tag = DW_TAG_array_type, name = "UniTuple(int64 x 1) ([1 x i64])"
        CHECK-SAME: sizeInBits = 64
        CHECK: #[[ITEMSIZE:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "itemsize"
        CHECK-SAME: offsetInBits = 192
        CHECK: #[[DATA:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "data"
        CHECK-SAME: offsetInBits = 256
        CHECK: #[[SHAPE:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "shape"
        CHECK-SAME: baseType = #[[TUPLE_TYPE]]
        CHECK-SAME: offsetInBits = 320
        CHECK: #[[STRIDES:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "strides"
        CHECK-SAME: baseType = #[[TUPLE_TYPE]]
        CHECK-SAME: offsetInBits = 384
        CHECK: #[[ARRAY_TYPE:di_composite_type[0-9]*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "array(float32, 1d, C) ({i8*, i8*, i64, i64, float*, [1 x i64], [1 x i64]})"
        CHECK-SAME: sizeInBits = 448
        CHECK: #[[LOCAL_VAR:di_local_variable[0-9]*]] = #llvm.di_local_variable<{{.*}}name = "tmp"
        CHECK-SAME: type = #[[ARRAY_TYPE]]
        CHECK: llvm.intr.dbg.declare #[[LOCAL_VAR]] = %{{[0-9]+}} : !llvm.ptr
        """,
        mlir,
    )


def test_mlir_poly_scalar_var():
    """Polymorphic scalar locals use discriminated-union DI and dbg.declare."""

    def k_poly_scalar_var(out, flag1, flag2):
        x = 1
        if flag1:
            x = 2.5
        elif flag2:
            x = np.uint32(3)
        else:
            x = True
        out[0] = x

    mlir = compiler.compile_mlir(
        k_poly_scalar_var,
        types.void(types.float64[:], types.boolean, types.boolean),
        debug=True,
        opt=False,
    )
    testing.filecheck(
        """
        CHECK-DAG: #[[DISCRIM:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "discriminator-bool.float64.int64.uint32"{{.*}}flags = Artificial
        CHECK-DAG: #[[BOOL_MEMBER:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "_bool"{{.*}}sizeInBits = 8{{.*}}extraData = 0 : i8
        CHECK-DAG: #[[FLOAT_MEMBER:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "_float64"{{.*}}sizeInBits = 64{{.*}}extraData = 1 : i8
        CHECK-DAG: #[[INT_MEMBER:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "_int64"{{.*}}sizeInBits = 64{{.*}}extraData = 2 : i8
        CHECK-DAG: #[[UINT_MEMBER:di_derived_type[0-9]*]] = #llvm.di_derived_type<tag = DW_TAG_member, name = "_uint32"{{.*}}sizeInBits = 32{{.*}}extraData = 3 : i8
        CHECK-DAG: #[[VARIANT_PART:di_composite_type[0-9]*]] = #llvm.di_composite_type<tag = DW_TAG_variant_part, name = "variant_part"{{.*}}discriminator = #[[DISCRIM]]{{.*}}elements = #[[BOOL_MEMBER]], #[[FLOAT_MEMBER]], #[[INT_MEMBER]], #[[UINT_MEMBER]]
        CHECK-DAG: #[[WRAPPER_TYPE:di_composite_type[0-9]*]] = #llvm.di_composite_type<tag = DW_TAG_structure_type, name = "variant_wrapper_struct"{{.*}}elements = #[[DISCRIM]], #[[VARIANT_PART]]
        CHECK: di_local_variable<{{.*}}name = "x"
        CHECK-SAME: type = #[[WRAPPER_TYPE]]
        CHECK: %[[POLY_SLOT:[0-9]+]] = llvm.alloca
        CHECK-SAME: !llvm.array<2 x i64>
        CHECK: llvm.intr.dbg.declare {{.*}} = %[[POLY_SLOT]] : !llvm.ptr
        """,
        mlir,
    )


def test_mlir_poly_scalar_var_runtime_debug():
    """Polymorphic scalar locals preserve runtime values under debug=True."""

    def k_poly_scalar_var_runtime(out, use_float, use_uint, use_bool):
        x = 1
        if use_float:
            x = 2.5
        elif use_uint:
            x = np.uint32(3)
        elif use_bool:
            x = True
        else:
            x = x + 10
        out[0] = x

    kernel = cuda.jit(
        types.void(types.float64[:], types.boolean, types.boolean, types.boolean),
        debug=True,
        opt=False,
    )(k_poly_scalar_var_runtime)

    for flags, expected in [
        ((False, False, False), 11.0),
        ((True, False, False), 2.5),
        ((False, True, False), 3.0),
        ((False, False, True), 1.0),
    ]:
        out = cuda.device_array(1, dtype=np.float64)
        kernel[1, 1](out, *flags)
        assert out.copy_to_host()[0] == expected
