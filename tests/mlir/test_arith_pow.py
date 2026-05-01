# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest
from numba_cuda_mlir.mlir.context import mlir_mod_ctx
from numba_cuda_mlir.mlir.dialect_exts import func
from numba_cuda_mlir import testing
from numba_cuda_mlir._mlir.extras import types as T


def _resolve_type(name):
    return {"f32": T.f32, "f64": T.f64, "i32": T.i32, "i64": T.i64}[name]()


def _build_pow_module(lhs_type, rhs_type):
    with mlir_mod_ctx() as ctx:

        @func.func
        def pow_func(a: lhs_type, b: rhs_type):
            return a**b

        pow_func.emit()
    return str(ctx.module)


@pytest.mark.parametrize("fty", ["f32", "f64"])
def test_powf(fty):
    with mlir_mod_ctx():
        mlir = _build_pow_module(_resolve_type(fty), _resolve_type(fty))
    testing.filecheck(
        """
        CHECK: math.powf
        """,
        mlir,
    )


@pytest.mark.parametrize(
    "fty,ity",
    [("f32", "i32"), ("f64", "i32"), ("f32", "i64"), ("f64", "i64")],
)
def test_fpowi(fty, ity):
    with mlir_mod_ctx():
        mlir = _build_pow_module(_resolve_type(fty), _resolve_type(ity))
    testing.filecheck(
        """
        CHECK: math.fpowi
        """,
        mlir,
    )


@pytest.mark.parametrize("ity", ["i32", "i64"])
def test_ipowi(ity):
    with mlir_mod_ctx():
        mlir = _build_pow_module(_resolve_type(ity), _resolve_type(ity))
    testing.filecheck(
        """
        CHECK: cf.assert
        CHECK-SAME: negative exponent not supported
        CHECK: scf.while
        """,
        mlir,
    )


def test_powf_f32_literal_rhs():
    with mlir_mod_ctx() as ctx:

        @func.func
        def pow_literal(a: T.f32()):
            return a**2.0

        pow_literal.emit()
    testing.filecheck(
        """
        CHECK: arith.constant 2.0{{.*}} : f32
        CHECK: math.powf
        """,
        str(ctx.module),
    )


def test_rpow():
    with mlir_mod_ctx() as ctx:

        @func.func
        def rpow_func(a: T.f32()):
            return 2.0**a

        rpow_func.emit()
    testing.filecheck(
        """
        CHECK: arith.constant 2.0{{.*}} : f32
        CHECK: math.powf
        """,
        str(ctx.module),
    )
