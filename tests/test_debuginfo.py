# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from enum import IntEnum

import pytest
import cuda.simt as cuda
from cuda.simt import types, compiler, testing
from cusimt.numba_cuda.types.ext_types import bfloat16
from cusimt.numba_cuda.np import numpy_support


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
        CHECK-NOT: llvm.intr.dbg.declare
        CHECK: llvm.intr.dbg.value
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
