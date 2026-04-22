# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for numba-cuda compatibility attributes and type conversions."""

import pytest
import cusimt._mlir.ir as ir
from cusimt._mlir.dialects import arith
from cusimt.mlir.dialect_exts.arith import ScalarValue


class TestMLIRDispatcherType:
    """Test MLIRDispatcherType compatibility attributes."""

    def test_templates_property(self):
        """MLIRDispatcherType should have a templates property for numba-cuda compat."""
        from cusimt.descriptor import MLIRDispatcherType

        class DummyDispatcher:
            pass

        dt = MLIRDispatcherType(DummyDispatcher())
        assert hasattr(dt, "templates")
        assert dt.templates == []


class TestCompileResult:
    """Test CompileResult compatibility attributes."""

    def test_codelibrary_attribute_without_cubin(self):
        """CompileResult._codelibrary should be None when no cubin in metadata."""
        from cusimt.compiler import CompileResult

        class MockCres:
            metadata = {}

        cr = CompileResult(MockCres())
        assert cr._codelibrary is None

    def test_codelibrary_attribute_with_cubin(self):
        """CompileResult._codelibrary should be CodeLibrary when cubin present."""
        from cusimt.compiler import CompileResult, CodeLibrary

        class MockCres:
            metadata = {"cubin": b"fake_cubin", "func_name": "test_func"}

        cr = CompileResult(MockCres())
        assert cr._codelibrary is not None
        assert isinstance(cr._codelibrary, CodeLibrary)
        assert hasattr(cr._codelibrary, "get_cufunc")

    def test_type_annotation_attribute(self):
        """CompileResult should have _type_annotation attribute."""
        from cusimt.compiler import CompileResult

        class MockCres:
            metadata = {}
            type_annotation = "test_annotation"

        cr = CompileResult(MockCres())
        assert cr._type_annotation == "test_annotation"

    def test_cooperative_attribute(self):
        """CompileResult should have cooperative attribute."""
        from cusimt.compiler import CompileResult

        class MockCres:
            metadata = {}
            use_cooperative = True

        cr = CompileResult(MockCres())
        assert cr.cooperative is True

    def test_cooperative_default_false(self):
        """CompileResult.cooperative should default to False."""
        from cusimt.compiler import CompileResult

        class MockCres:
            metadata = {}

        cr = CompileResult(MockCres())
        assert cr.cooperative is False

    def test_metadata_access(self):
        """CompileResult should allow access to metadata keys as attributes."""
        from cusimt.compiler import CompileResult

        class MockCres:
            metadata = {"cubin": b"test_cubin", "func_name": "test_func"}

        cr = CompileResult(MockCres())
        assert cr.cubin == b"test_cubin"
        assert cr.func_name == "test_func"

    def test_delegated_attribute(self):
        """CompileResult should delegate to underlying cres for other attributes."""
        from cusimt.compiler import CompileResult

        class MockCres:
            metadata = {}
            signature = "test_signature"

        cr = CompileResult(MockCres())
        assert cr.signature == "test_signature"


class TestScalarValueConversion:
    """Test ScalarValue to Numba type conversion."""

    def test_scalar_value_i32(self):
        """ScalarValue with i32 should convert to int32."""
        from cusimt.lowering_utilities.type_conversions import to_numba_type
        from cusimt import types

        with ir.Context(), ir.Location.unknown():
            i32 = ir.IntegerType.get_signless(32)
            c = arith.constant(i32, 42)
            sv = ScalarValue(c)
            result = to_numba_type(sv)
            assert result == types.int32

    def test_scalar_value_i64(self):
        """ScalarValue with i64 should convert to int64."""
        from cusimt.lowering_utilities.type_conversions import to_numba_type
        from cusimt import types

        with ir.Context(), ir.Location.unknown():
            i64 = ir.IntegerType.get_signless(64)
            c = arith.constant(i64, 42)
            sv = ScalarValue(c)
            result = to_numba_type(sv)
            assert result == types.int64

    def test_scalar_value_f32(self):
        """ScalarValue with f32 should convert to float32."""
        from cusimt.lowering_utilities.type_conversions import to_numba_type
        from cusimt import types

        with ir.Context(), ir.Location.unknown():
            f32 = ir.F32Type.get()
            c = arith.constant(f32, 3.14)
            sv = ScalarValue(c)
            result = to_numba_type(sv)
            assert result == types.float32

    def test_scalar_value_f64(self):
        """ScalarValue with f64 should convert to float64."""
        from cusimt.lowering_utilities.type_conversions import to_numba_type
        from cusimt import types

        with ir.Context(), ir.Location.unknown():
            f64 = ir.F64Type.get()
            c = arith.constant(f64, 3.14)
            sv = ScalarValue(c)
            result = to_numba_type(sv)
            assert result == types.float64

    def test_ir_value_conversion(self):
        """ir.Value should convert via its type attribute."""
        from cusimt.lowering_utilities.type_conversions import to_numba_type
        from cusimt import types

        with ir.Context(), ir.Location.unknown():
            i32 = ir.IntegerType.get_signless(32)
            c = arith.constant(i32, 42)
            # c is an ir.Value (OpResult)
            result = to_numba_type(c)
            assert result == types.int32


class TestToMlirTypeScalarValue:
    """Test to_mlir_type handling of ScalarValue."""

    def test_scalar_value_to_mlir_type(self):
        """ScalarValue should convert to its underlying MLIR type."""
        from cusimt.lowering_utilities.type_conversions import to_mlir_type

        with ir.Context(), ir.Location.unknown():
            i32 = ir.IntegerType.get_signless(32)
            c = arith.constant(i32, 42)
            sv = ScalarValue(c)
            result = to_mlir_type(sv)
            assert result == i32

    def test_ir_value_to_mlir_type(self):
        """ir.Value should convert to its underlying MLIR type."""
        from cusimt.lowering_utilities.type_conversions import to_mlir_type

        with ir.Context(), ir.Location.unknown():
            f64 = ir.F64Type.get()
            c = arith.constant(f64, 3.14)
            result = to_mlir_type(c)
            assert result == f64
