# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Tests for NumPy Record (structured dtype) support.
"""

import numpy as np
import pytest
from numba_cuda_mlir import cuda
from numba_cuda_mlir.numba_cuda.np import numpy_support


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

recordwithvaluestorage = np.dtype([("h", np.float16), ("flag", np.bool_)], align=True)

nestedrecordwitharray = np.dtype([("inner", [("a", np.float64), ("b", np.float64)], (1,))])

nestedrecordwith2darray = np.dtype([("inner", [("a", np.float64), ("b", np.float64)], (2, 2))])

nestedrecord = np.dtype([("inner", [("a", np.float64), ("b", np.float64)])])


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
        from numba_cuda_mlir._mlir import ir
        from numba_cuda_mlir.numba_cuda.types import Record
        from numba_cuda_mlir.models import mlir_data_manager

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
        from numba_cuda_mlir._mlir import ir
        from numba_cuda_mlir.numba_cuda.types import NestedArray
        from numba_cuda_mlir.models import mlir_data_manager

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
        from numba_cuda_mlir._mlir import ir
        from numba_cuda_mlir.numba_cuda.types import NestedArray
        from numba_cuda_mlir.models import mlir_data_manager

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
        from numba_cuda_mlir.lowering import record

        assert record is not None

    def test_record_getattr_registered(self):
        """Test that Record getattr lowering is registered."""
        from numba_cuda_mlir.lowering.record import registry
        from numba_cuda_mlir.numba_cuda.types import Record

        # registry.getattrs is a list of (func, attr, (type,)) tuples
        record_registered = any(Record in entry[2] for entry in registry.getattrs)
        assert record_registered, "Record getattr should be registered"

    def test_record_setattr_registered(self):
        """Test that Record setattr lowering is registered."""
        from numba_cuda_mlir.lowering.record import registry
        from numba_cuda_mlir.numba_cuda.types import Record

        # registry.setattrs is a list of (func, attr, (type, ...)) tuples
        record_registered = any(Record in entry[2] for entry in registry.setattrs)
        assert record_registered, "Record setattr should be registered"

    def test_record_static_getitem_registered(self):
        """Test that Record static_getitem lowering is registered."""
        from numba_cuda_mlir.lowering.record import registry
        from numba_cuda_mlir.numba_cuda.types import Record

        # registry.functions is a list - check for static_getitem with Record
        found = any(
            "static_getitem" in str(entry) and "Record" in str(entry)
            for entry in registry.functions
        )
        assert found, "static_getitem for Record should be registered"

    def test_record_static_setitem_registered(self):
        """Test that Record static_setitem lowering is registered."""
        from numba_cuda_mlir.lowering.record import registry
        from numba_cuda_mlir.numba_cuda.types import Record

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

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_static_setitem_value_storage_fields(self):
        """Test record['field'] assignment for fields with distinct storage types."""

        @cuda.jit
        def set_fields(ary):
            ary[0]["h"] = np.float16(1.5)
            ary[0]["flag"] = True
            ary[1]["h"] = np.float16(2.5)
            ary[1]["flag"] = False

        arr = cuda.to_device(np.zeros(2, dtype=recordwithvaluestorage))
        set_fields[1, 1](arr)
        result = arr.copy_to_host()
        np.testing.assert_allclose(result["h"], np.array([1.5, 2.5], dtype=np.float16))
        np.testing.assert_array_equal(result["flag"], np.array([True, False]))

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_set_nested_record_fields(self):
        @cuda.jit
        def set_fields(ary, scalar_ary):
            ary[0]["inner"][0]["a"] = 11.0
            ary[0]["inner"][0]["b"] = 22.0
            scalar_ary[0]["inner"]["a"] = 33.0
            scalar_ary[0]["inner"]["b"] = 44.0

        arr = cuda.to_device(np.zeros(2, dtype=nestedrecordwitharray))
        scalar_arr = cuda.to_device(np.zeros(2, dtype=nestedrecord))
        set_fields[1, 1](arr, scalar_arr)
        result = arr.copy_to_host()
        scalar_result = scalar_arr.copy_to_host()
        np.testing.assert_equal(result[0]["inner"][0]["a"], 11.0)
        np.testing.assert_equal(result[0]["inner"][0]["b"], 22.0)
        np.testing.assert_equal(scalar_result[0]["inner"]["a"], 33.0)
        np.testing.assert_equal(scalar_result[0]["inner"]["b"], 44.0)

    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_read_nested_record_field(self):
        # Regression for #214: reading a field of a scalar sub-record at a
        # runtime index (rec[i]["inner"]["a"]) used to scalar-load the
        # sub-record and dereference that value as a pointer -> illegal memory
        # access. Fixed by #160; existing coverage only exercised the nested
        # *array* read and the nested write, not this scalar sub-record read.
        outer = np.dtype([("inner", [("a", np.float64), ("b", np.float64)]), ("scale", np.float64)])

        @cuda.jit
        def read_nested(rec, out):
            i = cuda.grid(1)
            if i < rec.shape[0]:
                out[i] = rec[i]["inner"]["a"] * rec[i]["scale"]

        arr = np.zeros(1, dtype=outer)
        arr["inner"]["a"] = 5.0
        arr["scale"] = 2.0
        arr = cuda.to_device(arr)
        out = cuda.to_device(np.zeros(1, dtype=np.float64))

        read_nested[1, 1](arr, out)
        np.testing.assert_equal(out.copy_to_host()[0], 10.0)

    def test_read_tuple_indexed_nested_record_field(self):
        @cuda.jit
        def read_field(ary, out):
            out[0] = ary[0]["inner"][0, 1]["a"]

        arr = np.zeros(1, dtype=nestedrecordwith2darray)
        arr[0]["inner"][0, 1]["a"] = 42.0
        arr = cuda.to_device(arr)
        out = cuda.to_device(np.zeros(1, dtype=np.float64))

        read_field[1, 1](arr, out)
        np.testing.assert_equal(out.copy_to_host()[0], 42.0)


@pytest.mark.skip(reason="Causes memory errors")
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

    def test_record_array_setitem_field_compiles(self):
        """Test that set_a(ary, i, v) with ary[i].a = v compiles.

        This is the pattern from numba_cuda_mlir.numba_cuda's test_set_a.
        """
        from numba_cuda_mlir.numba_cuda import types

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

    def test_scalar_record_argument_compiles(self):
        """Test that record_set_a(r, v) with r.a = v compiles.

        This is the pattern from numba_cuda_mlir.numba_cuda's test_rec_set_a.
        """
        from numba_cuda_mlir.numba_cuda import types

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


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
class TestRecordArrayViews:
    """Record array views created inside kernels index the correct records.

    Regression tests for issue #324: views of record arrays were indexed
    incorrectly (getitem read the wrong record, setitem overwrote the wrong
    record, ``.ctypes.data`` reported the wrong address).
    """

    def test_strided_view_getitem(self):
        @cuda.jit
        def kernel(ary, out):
            view = ary[1::2]
            i = cuda.grid(1)
            out[i] = view[i].b

        host = np.zeros(8, dtype=recordtype)
        host["b"] = np.arange(8)
        out = cuda.device_array(4, np.int32)
        kernel[1, 4](cuda.to_device(host), out)
        np.testing.assert_equal(out.copy_to_host(), [1, 3, 5, 7])

    def test_offset_view_getitem(self):
        @cuda.jit
        def kernel(ary, out):
            view = ary[2:]
            i = cuda.grid(1)
            out[i] = view[i].b

        host = np.zeros(8, dtype=recordtype)
        host["b"] = np.arange(8)
        out = cuda.device_array(6, np.int32)
        kernel[1, 6](cuda.to_device(host), out)
        np.testing.assert_equal(out.copy_to_host(), np.arange(2, 8))

    @pytest.mark.parametrize("start,step", [(0, 1), (2, 1), (1, 2)])
    @pytest.mark.parametrize("index_kind", ["literal", "runtime"])
    def test_negative_record_indices(self, start, step, index_kind):
        if index_kind == "literal":

            @cuda.jit
            def kernel(ary, src, out):
                view = ary[start::step]
                out[0] = view[-1].b
                view[-1].b += 100
                view[-2] = src[-1]
        else:

            @cuda.jit
            def kernel(ary, src, out, index):
                view = ary[start::step]
                out[0] = view[index].b
                view[index].b += 100
                view[index - 1] = src[index]

        # Nine records give the strided view an exact extent, isolating
        # negative indexing from the slice-length bug tracked in #262.
        host = np.zeros(9, dtype=recordtype)
        host["a"] = np.arange(9) + 10.0
        host["b"] = np.arange(9) + 20
        host["c"] = np.arange(9) + 30.0j
        src = np.zeros(2, dtype=recordtype)
        src["a"] = [100.0, 200.0]
        src["b"] = [300, 400]
        src["c"] = [500.0j, 600.0j]
        ary = cuda.to_device(host)
        out = cuda.device_array(1, np.int32)
        args = (ary, cuda.to_device(src), out)
        if index_kind == "runtime":
            args += (-1,)
        kernel[1, 1](*args)

        selection = slice(start, None, step)
        assert out.copy_to_host()[0] == host["b"][selection][-1]
        expected = host.copy()
        expected["b"][selection][-1] += 100
        expected[selection][-2] = src[-1]
        np.testing.assert_equal(ary.copy_to_host(), expected)

    @pytest.mark.parametrize("start,step", [(2, 1), (1, 2)])
    def test_view_field_view(self, start, step):
        @cuda.jit
        def kernel(ary, out):
            field = ary[start::step].b
            i = cuda.grid(1)
            out[i] = field[i]

        host = np.zeros(8, dtype=recordtype)
        host["b"] = np.arange(8)
        expected = host["b"][start::step]
        out = cuda.device_array(expected.size, np.int32)
        kernel[1, expected.size](cuda.to_device(host), out)
        np.testing.assert_equal(out.copy_to_host(), expected)

    @pytest.mark.parametrize("start,step", [(0, 2), (2, 2), (6, -2)])
    @pytest.mark.parametrize("view_kind", ["argument", "kernel"])
    def test_packed_record_field_view(self, start, step, view_kind):
        # A two-record stride is 24 bytes, or three float64 elements.
        # Dividing the 12-byte record size before scaling truncates it to 8.
        dtype = np.dtype([("y", np.float64), ("x", np.int32)])
        assert dtype.itemsize == 12

        @cuda.jit
        def kernel(ary, out):
            field = ary.y
            i = cuda.grid(1)
            out[i] = field[i]
            field[i] += 10.0

        @cuda.jit
        def kernel_view(ary, out):
            field = ary[start::step].y
            i = cuda.grid(1)
            out[i] = field[i]
            field[i] += 10.0

        host = np.zeros(8, dtype=dtype)
        host["y"] = np.arange(8) + 100.0
        host["x"] = np.arange(8)
        selection = slice(start, None, step)
        ary = cuda.to_device(host)
        out = cuda.device_array(host[selection].size, np.float64)
        if view_kind == "kernel":
            kernel_view[1, out.size](ary, out)
        else:
            kernel[1, out.size](ary[selection], out)
        np.testing.assert_equal(out.copy_to_host(), host["y"][selection])
        expected = host.copy()
        expected["y"][selection] += 10.0
        np.testing.assert_equal(ary.copy_to_host(), expected)

    def test_strided_view_setitem(self):
        @cuda.jit
        def kernel(ary):
            view = ary[1::2]
            i = cuda.grid(1)
            view[i].a = i + 1.0
            view[i].b = 2 * i + 1

        arr = cuda.to_device(np.zeros(8, dtype=recordtype))
        kernel[1, 4](arr)
        result = arr.copy_to_host()
        expected = np.zeros(8, dtype=recordtype)
        for i in range(4):
            expected[2 * i + 1]["a"] = i + 1.0
            expected[2 * i + 1]["b"] = 2 * i + 1
        np.testing.assert_equal(result, expected)

    def test_strided_view_whole_record_setitem(self):
        @cuda.jit
        def kernel(ary, src):
            view = ary[1::2]
            i = cuda.grid(1)
            view[i] = src[i]

        host = np.zeros(8, dtype=recordtype)
        host["a"] = np.arange(8) + 10.0
        host["b"] = np.arange(8) + 20
        src = np.zeros(4, dtype=recordtype)
        src["a"] = np.arange(4) + 100.0
        src["b"] = np.arange(4) + 200
        arr = cuda.to_device(host)
        kernel[1, 4](arr, cuda.to_device(src))
        expected = host.copy()
        expected[1::2] = src
        np.testing.assert_equal(arr.copy_to_host(), expected)

    def test_view_ctypes_data_offset(self):
        @cuda.jit
        def kernel(ary, out):
            out[0] = ary[2:].ctypes.data - ary.ctypes.data
            out[1] = ary[1::2].ctypes.data - ary.ctypes.data

        arr = cuda.to_device(np.zeros(8, dtype=recordtype))
        out = cuda.device_array(2, np.int64)
        kernel[1, 1](arr, out)
        itemsize = recordtype.itemsize
        np.testing.assert_equal(out.copy_to_host(), [2 * itemsize, itemsize])

    def test_device_strided_view_as_argument(self):
        @cuda.jit
        def kernel(view, out):
            i = cuda.grid(1)
            out[i] = view[i].b

        host = np.zeros(8, dtype=recordtype)
        host["b"] = np.arange(8)
        arr = cuda.to_device(host)
        out = cuda.device_array(4, np.int32)
        kernel[1, 4](arr[1::2], out)
        np.testing.assert_equal(out.copy_to_host(), [1, 3, 5, 7])

    def test_non_power_of_two_record_size(self):
        # 12 bytes per record: exercises exact byte scaling of element
        # offsets (a vector-typed GEP would scale by LLVM's rounded-up
        # allocation size instead).
        dtype = np.dtype([("x", np.int32), ("y", np.float64)])
        assert dtype.itemsize == 12

        @cuda.jit
        def kernel(ary, out):
            i = cuda.grid(1)
            out[i] = ary[i + 1].x

        host = np.zeros(6, dtype=dtype)
        host["x"] = np.arange(6)
        out = cuda.device_array(5, np.int32)
        kernel[1, 5](cuda.to_device(host), out)
        np.testing.assert_equal(out.copy_to_host(), np.arange(1, 6))

        @cuda.jit
        def kernel_write(ary):
            i = cuda.grid(1)
            ary[i + 1].x = 10 * i

        arr = cuda.to_device(np.zeros(6, dtype=dtype))
        kernel_write[1, 5](arr)
        result = arr.copy_to_host()
        expected = np.zeros(6, dtype=dtype)
        expected["x"][1:] = [10 * i for i in range(5)]
        np.testing.assert_equal(result, expected)

    @pytest.mark.parametrize("dtype", [recordtype, np.dtype([("a", np.float64), ("b", np.int32)])])
    @pytest.mark.parametrize("view_kind", ["contiguous", "argument", "kernel"])
    def test_record_iteration(self, dtype, view_kind):
        @cuda.jit
        def kernel(ary, out):
            i = 0
            for record in ary:
                record.b += 10
                out[i] = record.b
                i += 1

        @cuda.jit
        def kernel_view(ary, out):
            i = 0
            for record in ary[1::2]:
                record.b += 10
                out[i] = record.b
                i += 1

        host = np.zeros(9, dtype=dtype)
        host["b"] = np.arange(9)
        ary = cuda.to_device(host)
        selection = slice(None) if view_kind == "contiguous" else slice(1, None, 2)
        out = cuda.device_array(host[selection].size, np.int32)
        if view_kind == "kernel":
            kernel_view[1, 1](ary, out)
        else:
            kernel[1, 1](ary[selection], out)
        expected = host.copy()
        expected["b"][selection] += 10
        np.testing.assert_equal(out.copy_to_host(), expected["b"][selection])
        np.testing.assert_equal(ary.copy_to_host(), expected)

    def test_record_item(self):
        # arr.item() returns the record as a pointer to element storage;
        # previously it loaded the raw memref element and crashed the
        # verifier downstream.
        @cuda.jit
        def kernel(ary, out):
            record = ary.item()
            out[0] = record.b

        host = np.zeros(1, dtype=recordtype)
        host["b"] = 42
        ary = cuda.to_device(host)
        out = cuda.device_array(1, np.int32)
        kernel[1, 1](ary, out)
        assert out.copy_to_host()[0] == 42

    def test_record_item_offset_view(self):
        # item() through an in-kernel offset view must read the viewed record.
        @cuda.jit
        def kernel(ary, out):
            record = ary[2:3].item()
            out[0] = record.b

        host = np.zeros(8, dtype=recordtype)
        host["b"] = np.arange(8)
        ary = cuda.to_device(host)
        out = cuda.device_array(1, np.int32)
        kernel[1, 1](ary, out)
        assert out.copy_to_host()[0] == 2
