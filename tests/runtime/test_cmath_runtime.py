# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for cmath runtime intrinsics MLIR generation.

These tests verify the structure of the generated MLIR for cmath intrinsics.
"""

import pytest
from numba_cuda_mlir.runtime.cmath_intrinsics import get_cmath_intrinsics_module
from numba_cuda_mlir import testing


@pytest.fixture(scope="module")
def cmath_module():
    """Get the generated cmath MLIR module."""
    return get_cmath_intrinsics_module()


class TestCmathExp:
    def test_exp_f32(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_exp_type_f32(
            CHECK-SAME: %{{.*}}: f32, %{{.*}}: f32) -> (f32, f32)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isfinite %{{.*}} : f32
            CHECK: math.isnan %{{.*}} : f32
            CHECK: math.exp %{{.*}} : f32
            CHECK: math.cos %{{.*}} : f32
            CHECK: math.sin %{{.*}} : f32
            CHECK: return %{{.*}}, %{{.*}} : f32, f32
            """,
            cmath_module,
        )

    def test_exp_f64(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_exp_type_f64(
            CHECK-SAME: %{{.*}}: f64, %{{.*}}: f64) -> (f64, f64)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isfinite %{{.*}} : f64
            CHECK: math.isnan %{{.*}} : f64
            CHECK: math.exp %{{.*}} : f64
            CHECK: math.cos %{{.*}} : f64
            CHECK: math.sin %{{.*}} : f64
            CHECK: return %{{.*}}, %{{.*}} : f64, f64
            """,
            cmath_module,
        )


class TestCmathSinh:
    def test_sinh_f32(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_sinh_type_f32(
            CHECK-SAME: %{{.*}}: f32, %{{.*}}: f32) -> (f32, f32)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f32
            CHECK: math.isnan %{{.*}} : f32
            CHECK: math.sinh %{{.*}} : f32
            CHECK: math.cosh %{{.*}} : f32
            CHECK: return %{{.*}}, %{{.*}} : f32, f32
            """,
            cmath_module,
        )

    def test_sinh_f64(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_sinh_type_f64(
            CHECK-SAME: %{{.*}}: f64, %{{.*}}: f64) -> (f64, f64)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f64
            CHECK: math.isnan %{{.*}} : f64
            CHECK: math.sinh %{{.*}} : f64
            CHECK: math.cosh %{{.*}} : f64
            CHECK: return %{{.*}}, %{{.*}} : f64, f64
            """,
            cmath_module,
        )


class TestCmathCosh:
    def test_cosh_f32(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_cosh_type_f32(
            CHECK-SAME: %{{.*}}: f32, %{{.*}}: f32) -> (f32, f32)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f32
            CHECK: math.isnan %{{.*}} : f32
            CHECK: math.cosh %{{.*}} : f32
            CHECK: math.sinh %{{.*}} : f32
            CHECK: return %{{.*}}, %{{.*}} : f32, f32
            """,
            cmath_module,
        )

    def test_cosh_f64(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_cosh_type_f64(
            CHECK-SAME: %{{.*}}: f64, %{{.*}}: f64) -> (f64, f64)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f64
            CHECK: math.isnan %{{.*}} : f64
            CHECK: math.cosh %{{.*}} : f64
            CHECK: math.sinh %{{.*}} : f64
            CHECK: return %{{.*}}, %{{.*}} : f64, f64
            """,
            cmath_module,
        )


class TestCmathSin:
    def test_sin_f32(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_sin_type_f32(
            CHECK-SAME: %{{.*}}: f32, %{{.*}}: f32) -> (f32, f32)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f32
            CHECK: math.isnan %{{.*}} : f32
            CHECK: math.sin %{{.*}} : f32
            CHECK: math.cos %{{.*}} : f32
            CHECK: math.sinh %{{.*}} : f32
            CHECK: math.cosh %{{.*}} : f32
            CHECK: return %{{.*}}, %{{.*}} : f32, f32
            """,
            cmath_module,
        )

    def test_sin_f64(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_sin_type_f64(
            CHECK-SAME: %{{.*}}: f64, %{{.*}}: f64) -> (f64, f64)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f64
            CHECK: math.isnan %{{.*}} : f64
            CHECK: math.sin %{{.*}} : f64
            CHECK: math.cos %{{.*}} : f64
            CHECK: math.sinh %{{.*}} : f64
            CHECK: math.cosh %{{.*}} : f64
            CHECK: return %{{.*}}, %{{.*}} : f64, f64
            """,
            cmath_module,
        )


class TestCmathCos:
    def test_cos_f32(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_cos_type_f32(
            CHECK-SAME: %{{.*}}: f32, %{{.*}}: f32) -> (f32, f32)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isfinite %{{.*}} : f32
            CHECK: math.cos %{{.*}} : f32
            CHECK: math.sin %{{.*}} : f32
            CHECK: math.cosh %{{.*}} : f32
            CHECK: math.sinh %{{.*}} : f32
            CHECK: return %{{.*}}, %{{.*}} : f32, f32
            """,
            cmath_module,
        )

    def test_cos_f64(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_cos_type_f64(
            CHECK-SAME: %{{.*}}: f64, %{{.*}}: f64) -> (f64, f64)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isfinite %{{.*}} : f64
            CHECK: math.cos %{{.*}} : f64
            CHECK: math.sin %{{.*}} : f64
            CHECK: math.cosh %{{.*}} : f64
            CHECK: math.sinh %{{.*}} : f64
            CHECK: return %{{.*}}, %{{.*}} : f64, f64
            """,
            cmath_module,
        )


class TestCmathRect:
    def test_rect_f32(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_rect_type_f32(
            CHECK-SAME: %{{.*}}: f32, %{{.*}}: f32) -> (f32, f32)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f32
            CHECK: math.isfinite %{{.*}} : f32
            CHECK: math.cos %{{.*}} : f32
            CHECK: math.sin %{{.*}} : f32
            CHECK: return %{{.*}}, %{{.*}} : f32, f32
            """,
            cmath_module,
        )

    def test_rect_f64(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_rect_type_f64(
            CHECK-SAME: %{{.*}}: f64, %{{.*}}: f64) -> (f64, f64)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f64
            CHECK: math.isfinite %{{.*}} : f64
            CHECK: math.cos %{{.*}} : f64
            CHECK: math.sin %{{.*}} : f64
            CHECK: return %{{.*}}, %{{.*}} : f64, f64
            """,
            cmath_module,
        )


class TestCmathSqrt:
    def test_sqrt_f32(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_sqrt_type_f32(
            CHECK-SAME: %{{.*}}: f32, %{{.*}}: f32) -> (f32, f32)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f32
            CHECK: math.isnan %{{.*}} : f32
            CHECK: math.sqrt %{{.*}} : f32
            CHECK: math.copysign %{{.*}}, %{{.*}} : f32
            CHECK: return %{{.*}}, %{{.*}} : f32, f32
            """,
            cmath_module,
        )

    def test_sqrt_f64(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_sqrt_type_f64(
            CHECK-SAME: %{{.*}}: f64, %{{.*}}: f64) -> (f64, f64)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f64
            CHECK: math.isnan %{{.*}} : f64
            CHECK: math.sqrt %{{.*}} : f64
            CHECK: math.copysign %{{.*}}, %{{.*}} : f64
            CHECK: return %{{.*}}, %{{.*}} : f64, f64
            """,
            cmath_module,
        )


class TestCmathAcos:
    def test_acos_f32(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_acos_type_f32(
            CHECK-SAME: %{{.*}}: f32, %{{.*}}: f32) -> (f32, f32)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f32
            CHECK: math.isnan %{{.*}} : f32
            CHECK: math.atan2 %{{.*}}, %{{.*}} : f32
            CHECK: return %{{.*}}, %{{.*}} : f32, f32
            """,
            cmath_module,
        )

    def test_acos_f64(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_acos_type_f64(
            CHECK-SAME: %{{.*}}: f64, %{{.*}}: f64) -> (f64, f64)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f64
            CHECK: math.isnan %{{.*}} : f64
            CHECK: math.atan2 %{{.*}}, %{{.*}} : f64
            CHECK: return %{{.*}}, %{{.*}} : f64, f64
            """,
            cmath_module,
        )


class TestCmathAcosh:
    def test_acosh_f32(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_acosh_type_f32(
            CHECK-SAME: %{{.*}}: f32, %{{.*}}: f32) -> (f32, f32)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f32
            CHECK: math.isnan %{{.*}} : f32
            CHECK: math.atan2 %{{.*}}, %{{.*}} : f32
            CHECK: return %{{.*}}, %{{.*}} : f32, f32
            """,
            cmath_module,
        )

    def test_acosh_f64(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_acosh_type_f64(
            CHECK-SAME: %{{.*}}: f64, %{{.*}}: f64) -> (f64, f64)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f64
            CHECK: math.isnan %{{.*}} : f64
            CHECK: math.atan2 %{{.*}}, %{{.*}} : f64
            CHECK: return %{{.*}}, %{{.*}} : f64, f64
            """,
            cmath_module,
        )


class TestCmathAtan:
    def test_atan_f32(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_atan_type_f32(
            CHECK-SAME: %{{.*}}: f32, %{{.*}}: f32) -> (f32, f32)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f32
            CHECK: math.isnan %{{.*}} : f32
            CHECK: math.atan2 %{{.*}}, %{{.*}} : f32
            CHECK: return %{{.*}}, %{{.*}} : f32, f32
            """,
            cmath_module,
        )

    def test_atan_f64(self, cmath_module):
        testing.filecheck(
            """
            CHECK-LABEL: func.func private @cmath_atan_type_f64(
            CHECK-SAME: %{{.*}}: f64, %{{.*}}: f64) -> (f64, f64)
            CHECK-SAME: attributes {alwaysinline}
            CHECK: math.isinf %{{.*}} : f64
            CHECK: math.isnan %{{.*}} : f64
            CHECK: math.atan2 %{{.*}}, %{{.*}} : f64
            CHECK: return %{{.*}}, %{{.*}} : f64, f64
            """,
            cmath_module,
        )


class TestAllFunctionsPresent:
    @pytest.mark.parametrize(
        "func_name",
        [
            "cmath_exp_type_f32",
            "cmath_exp_type_f64",
            "cmath_sinh_type_f32",
            "cmath_sinh_type_f64",
            "cmath_cosh_type_f32",
            "cmath_cosh_type_f64",
            "cmath_sin_type_f32",
            "cmath_sin_type_f64",
            "cmath_cos_type_f32",
            "cmath_cos_type_f64",
            "cmath_rect_type_f32",
            "cmath_rect_type_f64",
            "cmath_sqrt_type_f32",
            "cmath_sqrt_type_f64",
            "cmath_acos_type_f32",
            "cmath_acos_type_f64",
            "cmath_acosh_type_f32",
            "cmath_acosh_type_f64",
            "cmath_atan_type_f32",
            "cmath_atan_type_f64",
        ],
    )
    def test_function_exists(self, cmath_module, func_name):
        assert f"@{func_name}" in cmath_module
