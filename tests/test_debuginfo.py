# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Validates that debug=True produces MLIR with correct DI attributes:
emissionKind=Full, DILocalVariable, DIBasicType, dbg.declare for
scalar args, dbg.value for locals.
"""

from cuda.simt import types, compiler, testing


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
