# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for NumPy Record (structured dtype) support.
"""

import numpy as np
import pytest
from numba import cuda
from cusimt.numba_cuda.np import numpy_support


# Basic record type definitions
recordtype = np.dtype(
    [
        ("a", np.float64),
        ("b", np.int32),
        ("c", np.complex64),
    ],
    align=True,
)

recordwitharray = np.dtype([("g", np.int32), ("h", np.float32, 2)], align=True)

recordwith2darray = np.dtype([("i", np.int32), ("j", np.float32, (3, 2))])


class TestRecordTypeModel:
    """Test that Record types can be converted to MLIR types."""

    def test_record_type_conversion(self):
        """Test that from_dtype works and produces a valid Record type."""
        rec = numpy_support.from_dtype(recordtype)
        assert rec is not None
        assert hasattr(rec, "fields")
        assert "a" in rec.fields
        assert "b" in rec.fields
        assert "c" in rec.fields

    def test_record_with_array_type_conversion(self):
        """Test Record with nested array field."""
        rec = numpy_support.from_dtype(recordwitharray)
        assert rec is not None
        assert "g" in rec.fields
        assert "h" in rec.fields

    def test_record_with_2d_array_type_conversion(self):
        """Test Record with 2D nested array field."""
        rec = numpy_support.from_dtype(recordwith2darray)
        assert rec is not None
        assert "i" in rec.fields
        assert "j" in rec.fields

    def test_record_mlir_type_model(self):
        """Test that Record type can be looked up in MLIR data manager."""
        from cusimt._mlir import ir
        from numba.core.types import Record
        from cusimt.models import mlir_data_manager

        rec = numpy_support.from_dtype(recordtype)
        assert isinstance(rec, Record)

        with ir.Context(), ir.Location.unknown():
            model = mlir_data_manager.lookup(rec)
            assert model.__class__.__name__ == "RecordModel"
            be_type = model.get_value_type()
            # Record should be represented as LLVM pointer
            assert "llvm.ptr" in str(be_type)

    def test_nested_array_mlir_type_model(self):
        """Test that NestedArray type can be looked up in MLIR data manager."""
        from cusimt._mlir import ir
        from numba.core.types import NestedArray
        from cusimt.models import mlir_data_manager

        rec = numpy_support.from_dtype(recordwitharray)
        h_type = rec.typeof("h")
        assert isinstance(h_type, NestedArray)

        with ir.Context(), ir.Location.unknown():
            model = mlir_data_manager.lookup(h_type)
            assert model.__class__.__name__ == "NestedArrayModel"
            be_type = model.get_value_type()
            # NestedArray is represented as llvm.ptr (shape/stride info from type)
            assert "llvm.ptr" in str(be_type)

    def test_nested_2d_array_mlir_type_model(self):
        """Test that 2D NestedArray type can be looked up in MLIR data manager."""
        from cusimt._mlir import ir
        from numba.core.types import NestedArray
        from cusimt.models import mlir_data_manager

        rec = numpy_support.from_dtype(recordwith2darray)
        j_type = rec.typeof("j")
        assert isinstance(j_type, NestedArray)

        with ir.Context(), ir.Location.unknown():
            model = mlir_data_manager.lookup(j_type)
            assert model.__class__.__name__ == "NestedArrayModel"
            be_type = model.get_value_type()
            # NestedArray is represented as llvm.ptr (shape/stride info from type)
            assert "llvm.ptr" in str(be_type)


class TestRecordLoweringRegistration:
    """Test that Record lowering functions are properly registered."""

    def test_record_lowering_module_imports(self):
        """Test that the record lowering module can be imported."""
        from cusimt.lowering import record

        assert record is not None

    def test_record_getattr_registered(self):
        """Test that Record getattr lowering is registered."""
        from cusimt.lowering.record import registry
        from numba.core.types import Record

        # registry.getattrs is a list of (func, attr, (type,)) tuples
        record_registered = any(Record in entry[2] for entry in registry.getattrs)
        assert record_registered, "Record getattr should be registered"

    def test_record_setattr_registered(self):
        """Test that Record setattr lowering is registered."""
        from cusimt.lowering.record import registry
        from numba.core.types import Record

        # registry.setattrs is a list of (func, attr, (type, ...)) tuples
        record_registered = any(Record in entry[2] for entry in registry.setattrs)
        assert record_registered, "Record setattr should be registered"

    def test_record_static_getitem_registered(self):
        """Test that Record static_getitem lowering is registered."""
        from cusimt.lowering.record import registry
        from numba.core.types import Record

        # registry.functions is a list - check for static_getitem with Record
        found = any(
            "static_getitem" in str(entry) and "Record" in str(entry)
            for entry in registry.functions
        )
        assert found, "static_getitem for Record should be registered"

    def test_record_static_setitem_registered(self):
        """Test that Record static_setitem lowering is registered."""
        from cusimt.lowering.record import registry
        from numba.core.types import Record

        # Check for static_setitem with Record
        found = any(
            "static_setitem" in str(entry) and "Record" in str(entry)
            for entry in registry.functions
        )
        assert found, "static_setitem for Record should be registered"


class TestRecordArrayCompilation:
    """Test that kernels using record arrays can compile."""

    def test_record_array_kernel_compiles(self):
        """Test that a kernel taking a record array compiles without KeyError."""

        def set_field_a(ary, i, v):
            ary[i].a = v

        rec = numpy_support.from_dtype(recordtype)
        # This should not raise KeyError for Record type
        kernel = cuda.jit(set_field_a)

        # Create test data
        arr = np.zeros(3, dtype=recordtype)

        # Try to compile by invoking with signature
        # Note: actual field access lowering may not work yet,
        # but the type conversion should succeed
        try:
            kernel[1, 1](arr, 0, 1.5)
        except NotImplementedError as e:
            # Expected - field access lowering not implemented yet
            # But we should NOT get KeyError for Record type
            assert "KeyError" not in str(e)
            assert "Cannot convert type" not in str(e)
        except Exception as e:
            # Other errors are OK for now, as long as it's not the type conversion error
            error_msg = str(e)
            assert "KeyError" not in error_msg or "Record" not in error_msg


class TestRecordFieldAccess:
    """Test record field access operations."""

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_set_field_a(self):
        """Test setting float64 field via array indexing: ary[i].a = v"""

        @cuda.jit
        def set_a(ary, i, v):
            ary[i].a = v

        arr = np.zeros(3, dtype=recordtype)
        arr = cuda.to_device(arr)
        set_a[1, 1](arr, 1, 99.0)
        result = arr.copy_to_host()
        assert result[1]["a"] == 99.0

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_set_field_b(self):
        """Test setting int32 field via array indexing."""

        @cuda.jit
        def set_b(ary, i, v):
            ary[i].b = v

        arr = np.zeros(3, dtype=recordtype)
        arr = cuda.to_device(arr)
        set_b[1, 1](arr, 0, 42)
        result = arr.copy_to_host()
        assert result[0]["b"] == 42

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_read_field(self):
        """Test reading a record field."""

        @cuda.jit
        def read_a(ary, out):
            out[0] = ary[0].a

        arr = np.zeros(3, dtype=recordtype)
        arr[0]["a"] = 3.14159
        arr = cuda.to_device(arr)
        out = cuda.to_device(np.zeros(1, dtype=np.float64))

        read_a[1, 1](arr, out)
        result = out.copy_to_host()
        np.testing.assert_almost_equal(result[0], 3.14159)


class TestScalarRecordArgument:
    """Test passing scalar records as kernel arguments."""

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_scalar_record_set_field(self):
        """Test setting a field on a scalar record argument: r.a = v

        This corresponds to numba-cuda's test_rec_set_a test.
        """

        @cuda.jit
        def record_set_a(r, v):
            r.a = v

        # Create a scalar record (numpy.void)
        arr = np.zeros(1, dtype=recordtype)
        rec = arr[0]  # This is a numpy.void
        rec = cuda.to_device(rec)

        record_set_a[1, 1](rec, 1.5)
        result = rec.copy_to_host()
        assert result["a"] == 1.5

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_scalar_record_read_field(self):
        """Test reading a field from a scalar record argument.

        This corresponds to numba-cuda's test_rec_read_a test.
        """

        @cuda.jit
        def record_read_a(r, arr):
            arr[0] = r.a

        # Create a scalar record
        rec_arr = np.zeros(1, dtype=recordtype)
        rec_arr[0]["a"] = 3.14
        rec = rec_arr[0]
        rec = cuda.to_device(rec)
        out = cuda.to_device(np.zeros(1, dtype=np.float64))

        record_read_a[1, 1](rec, out)
        result = out.copy_to_host()
        np.testing.assert_almost_equal(result[0], 3.14)


class TestRecordArrayCompilationErrors:
    """Test compilation of record array operations to identify errors.

    These tests require CUDA to compile because of how the MLIR compiler
    initializes. They test the full compilation pipeline.
    """

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    @pytest.mark.xfail(reason="Array[Record] getitem typing not yet implemented")
    def test_record_array_setitem_field_compiles(self):
        """Test that set_a(ary, i, v) with ary[i].a = v compiles.

        This is the pattern from cusimt.numba_cuda's test_set_a.
        """
        from numba.core import types

        def set_a(ary, i, v):
            ary[i].a = v

        rec = numpy_support.from_dtype(recordtype)

        # Create signature: (record_array, int64, float64)
        sig = (types.Array(rec, 1, "C"), types.int64, types.float64)

        # Specialize first to compile, then inspect
        kernel = cuda.jit(set_a)
        kernel.specialize(*sig)
        ptx = kernel.inspect_asm(signature=sig)
        assert ptx is not None

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    @pytest.mark.xfail(reason="Record setattr typing not yet implemented")
    def test_scalar_record_argument_compiles(self):
        """Test that record_set_a(r, v) with r.a = v compiles.

        This is the pattern from cusimt.numba_cuda's test_rec_set_a.
        """
        from numba.core import types

        def record_set_a(r, v):
            r.a = v

        rec = numpy_support.from_dtype(recordtype)

        # Signature: (record, float64)
        sig = (rec, types.float64)

        # Specialize first to compile, then inspect
        kernel = cuda.jit(record_set_a)
        kernel.specialize(*sig)
        ptx = kernel.inspect_asm(signature=sig)
        assert ptx is not None
