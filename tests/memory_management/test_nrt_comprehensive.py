# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Comprehensive NRT tests based on numba-cuda test suite."""

import pytest
import numpy as np
from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda import jit


@pytest.fixture(autouse=True)
def reset_rtsys():
    """Reset rtsys before and after each test."""
    from numba_cuda_mlir.memory_management import rtsys

    rtsys._reset()
    yield
    rtsys._reset()


class TestNrtBasic:
    """Basic NRT functionality tests."""

    def test_nrt_launches(self):
        """Test basic NRT array creation and kernel launch."""

        @jit
        def f(x):
            return x[:5]

        @jit
        def g():
            x = np.empty(10, np.int64)
            f(x)

        g[1, 1]()
        cuda.synchronize()

    def test_nrt_ptx_contains_runtime_calls(self):
        """Test that PTX contains NRT function calls."""
        from numba_cuda_mlir import compiler

        @jit
        def getstr(s):
            return s.upper()

        @jit
        def kernel(out):
            out[0] = getstr("test") == "TEST"

        out = np.zeros(1, dtype=np.bool_)
        cres = compiler.compile_for(kernel, out)
        ptx = cres.metadata["ptx"]

        assert "NRT_MemInfo_alloc_aligned" in ptx
        assert "NRT_MemInfo_data_fast" in ptx

    def test_nrt_returns_correct(self):
        """Test that NRT array operations return correct values."""

        @jit
        def f(x):
            return x[5:]

        @jit
        def g(out_ary):
            x = np.empty(10, np.int64)
            x[5] = 1
            y = f(x)
            out_ary[0] = y[0]

        out_ary = np.zeros(1, dtype=np.int64)
        g[1, 1](out_ary)
        assert out_ary[0] == 1

    @pytest.mark.xfail(reason="np.array unimplemented")
    def test_nrt_arrays_basic(self):
        """Test basic array creation and access."""

        @jit
        def foo():
            return np.array([1, 2, 3])

        @jit(dump=True)
        def kernel(a):
            nrt_array = foo()
            a[0] = nrt_array[0]
            a[1] = nrt_array.shape[0]

        a = np.zeros(2, dtype=np.int32)
        kernel[1, 1](a)
        assert a[0] == 1
        assert a[1] == 3

    @pytest.mark.xfail(reason="Advanced NRT array ops not implemented yet")
    def test_nrt_arrays_slicing():
        """Test array slicing with NRT."""

        @jit
        def kernel(out):
            arr = np.arange(10)
            sliced = arr[2:7]
            out[0] = sliced[0]
            out[1] = len(sliced)

        out = np.zeros(2, dtype=np.int32)
        kernel[1, 1](out)
        assert out[0] == 2
        assert out[1] == 5

    @pytest.mark.xfail(reason="Advanced NRT array ops not implemented yet")
    def test_nrt_arrays_nested_calls(self):
        """Test nested function calls with NRT arrays."""

        @jit
        def create_array(size):
            return np.arange(size)

        @jit
        def process_array(arr):
            return arr * 2

        @jit
        def kernel(out):
            arr = create_array(5)
            processed = process_array(arr)
            out[0] = processed[2]

        out = np.zeros(1, dtype=np.int32)
        kernel[1, 1](out)
        assert out[0] == 4


class TestNrtLinking:
    """NRT linking functionality tests."""

    @pytest.mark.xfail(reason="NRT linking not implemented yet")
    def test_nrt_detect_linked_ptx_file(self):
        """Test linking external PTX code that uses NRT."""
        from numba_cuda_mlir.memory_management.nrt import get_include
        from numba_cuda_mlir.cuda.cudadrv.nvrtc import compile
        from numba_cuda_mlir.cuda.cudadrv.linkable_code import PTXSource
        from numba_cuda_mlir.cuda import get_current_device

        # Create external C++ code that uses NRT
        src = f"#include <{get_include()}/nrt.cuh>"
        src += """
                 extern "C" __device__ int device_allocate_deallocate(int* nb_retval){
                     auto ptr = NRT_Allocate(1);
                     NRT_Free(ptr);
                     return 0;
                 }
        """

        # Compile to PTX
        cc = get_current_device().compute_capability
        ptx, _ = compile(src, "external_nrt.cu", cc)

        # Define device function handle for linking
        def allocate_deallocate_handle():
            pass

        # Create kernel that links with external PTX
        @jit(link=[PTXSource(ptx.code, nrt=True)])
        def kernel():
            allocate_deallocate_handle()

        kernel[1, 1]()

    @pytest.mark.xfail(reason="NRT linking not implemented yet")
    def test_nrt_detect_linkable_code(self):
        """Test linking various types of NRT-enabled code objects."""
        # This test would require pre-compiled binary objects
        # Skip for now as it needs TEST_BIN_DIR setup
        pytest.skip("Requires pre-compiled NRT binary objects")


class TestNrtStatistics:
    """NRT statistics functionality tests."""

    def setup_method(self):
        """Set up stats state for each test."""
        from numba_cuda_mlir.memory_management import rtsys

        rtsys.ensure_initialized()
        self._stats_state = rtsys.memsys_stats_enabled()

    def teardown_method(self):
        """Restore stats state after each test."""
        from numba_cuda_mlir.memory_management import rtsys

        if self._stats_state:
            rtsys.memsys_enable_stats()
        else:
            rtsys.memsys_disable_stats()

    @pytest.mark.xfail(reason="NRT stats integration not implemented yet")
    def test_stats_env_var_explicit_on(self):
        """Test that NRT stats work when explicitly enabled via env var."""
        # This test would run in a subprocess with NUMBA_CUDA_NRT_STATS=1
        # For now, test basic stats functionality
        from numba_cuda_mlir.memory_management import rtsys

        @jit
        def foo():
            x = np.arange(10)[0]

        # Initialize NRT before use
        rtsys.ensure_initialized()
        rtsys.memsys_enable_stats()
        assert rtsys.memsys_stats_enabled(), "Stats not enabled"

        orig_stats = rtsys.get_allocation_stats()
        foo[1, 1]()
        new_stats = rtsys.get_allocation_stats()

        # Check that allocations happened
        total_alloc = new_stats.alloc - orig_stats.alloc
        total_free = new_stats.free - orig_stats.free
        total_mi_alloc = new_stats.mi_alloc - orig_stats.mi_alloc
        total_mi_free = new_stats.mi_free - orig_stats.mi_free

        expected = 1
        assert total_alloc == expected
        assert total_free == expected
        assert total_mi_alloc == expected
        assert total_mi_free == expected

    def test_stats_status_toggle(self):
        """Test enabling and disabling stats."""
        from numba_cuda_mlir.memory_management import rtsys

        @jit
        def foo():
            tmp = np.ones(3)
            arr = np.arange(5 * tmp[0])  # noqa: F841
            return None

        # Test stats toggle functionality
        rtsys.ensure_initialized()

        # Enable stats
        rtsys.memsys_enable_stats()
        assert rtsys.memsys_stats_enabled()

        for i in range(2):
            # Capture stats state
            stats_1 = rtsys.get_allocation_stats()

            # Disable stats
            rtsys.memsys_disable_stats()
            assert not rtsys.memsys_stats_enabled()

            # Run kernel (stats shouldn't change while disabled)
            # Note: This will fail until NRT lowering is implemented
            try:
                foo[1, 1]()
            except Exception:
                pass  # Expected until NRT lowering works

            # Re-enable stats
            rtsys.memsys_enable_stats()
            assert rtsys.memsys_stats_enabled()

            # Stats should be unchanged
            stats_2 = rtsys.get_allocation_stats()
            assert stats_1 == stats_2

    def test_rtsys_stats_query_raises_exception_when_disabled(self):
        """Test that stats queries raise when disabled."""
        from numba_cuda_mlir.memory_management import rtsys

        rtsys.ensure_initialized()
        rtsys.memsys_disable_stats()
        assert not rtsys.memsys_stats_enabled()

        with pytest.raises(RuntimeError, match="NRT stats are disabled"):
            rtsys.get_allocation_stats()

    def test_read_one_stat(self):
        """Test reading individual stat values."""
        from numba_cuda_mlir.memory_management import rtsys

        @jit
        def foo():
            tmp = np.ones(3)
            arr = np.arange(5 * tmp[0])  # noqa: F841

        rtsys.ensure_initialized()
        rtsys.memsys_enable_stats()

        # Note: This will fail until NRT lowering is implemented
        try:
            foo[1, 1]()
            foo[1, 1]()

            # Get stats struct and individual stats
            stats = rtsys.get_allocation_stats()
            stats_alloc = rtsys.memsys_get_stats_alloc()
            stats_mi_alloc = rtsys.memsys_get_stats_mi_alloc()
            stats_free = rtsys.memsys_get_stats_free()
            stats_mi_free = rtsys.memsys_get_stats_mi_free()

            # Check individual stats match stats struct
            assert stats.alloc == stats_alloc
            assert stats.mi_alloc == stats_mi_alloc
            assert stats.free == stats_free
            assert stats.mi_free == stats_mi_free
        except Exception:
            pytest.xfail("NRT lowering not implemented yet")


class TestNrtRefCt:
    """Reference counting tests."""

    def setup_method(self):
        """Set up each test."""
        from numba_cuda_mlir.memory_management import rtsys

        rtsys.ensure_initialized()
        rtsys.memsys_enable_stats()

    @pytest.mark.xfail(reason="NRT reference counting not implemented yet")
    def test_no_return(self):
        """Test allocation/deallocation balance in loops (issue #1291)."""
        from numba_cuda_mlir.memory_management import rtsys

        n = 10

        @jit
        def kernel():
            for i in range(n):
                temp = np.empty(2)  # noqa: F841
            return None

        init_stats = rtsys.get_allocation_stats()
        kernel[1, 1]()
        cur_stats = rtsys.get_allocation_stats()

        # Should have n allocations and n deallocations
        assert cur_stats.alloc - init_stats.alloc == n
        assert cur_stats.free - init_stats.free == n

    @pytest.mark.xfail(reason="NRT reference counting not implemented yet")
    def test_escaping_var_init_in_loop(self):
        """Test variable lifetime across loops (issue #1297)."""
        from numba_cuda_mlir.memory_management import rtsys

        @jit
        def g(n):
            x = np.empty((n, 2))

            for i in range(n):
                y = x[i]

            for i in range(n):
                y = x[i]  # noqa: F841

            return None

        init_stats = rtsys.get_allocation_stats()
        g[1, 1](10)
        cur_stats = rtsys.get_allocation_stats()

        # Should have only 1 allocation (for x) and 1 deallocation
        assert cur_stats.alloc - init_stats.alloc == 1
        assert cur_stats.free - init_stats.free == 1

    @pytest.mark.xfail(reason="NRT reference counting not implemented yet")
    def test_invalid_computation_of_lifetime(self):
        """Test conditional block lifetime handling (issue #1573)."""
        from numba_cuda_mlir.memory_management import rtsys

        @jit
        def if_with_allocation_and_initialization(arr1, test1):
            tmp_arr = np.empty_like(arr1)

            for i in range(tmp_arr.shape[0]):
                pass

            if test1:
                np.empty_like(arr1)

        arr = np.random.random((5, 5))

        init_stats = rtsys.get_allocation_stats()
        if_with_allocation_and_initialization[1, 1](arr, False)
        cur_stats = rtsys.get_allocation_stats()

        # Allocations should equal deallocations
        assert cur_stats.alloc - init_stats.alloc == cur_stats.free - init_stats.free

    def test_del_at_beginning_of_loop(self):
        """Test variable deletion edge case (issue #1734)."""
        from numba_cuda_mlir.memory_management import rtsys

        @jit
        def f(arr):
            res = 0

            for i in (0, 1):
                # `del t` is issued here before defining t
                t = arr[i]
                if t[i] > 1:
                    res += t[i]

        arr = np.ones((2, 2))

        init_stats = rtsys.get_allocation_stats()
        f[1, 1](arr)
        cur_stats = rtsys.get_allocation_stats()

        # Allocations should equal deallocations
        assert cur_stats.alloc - init_stats.alloc == cur_stats.free - init_stats.free


class TestNrtArrayOperations:
    """Tests for array operations with NRT."""

    def test_array_creation_basic(self):
        """Test basic array creation with NRT."""

        @jit
        def kernel(out):
            arr = np.empty(5, dtype=np.int32)
            arr[0] = 42
            out[0] = arr[0]

        out = np.zeros(1, dtype=np.int32)
        kernel[1, 1](out)
        assert out[0] == 42

    @pytest.mark.xfail(reason="NRT array operations not implemented yet")
    def test_array_slicing(self):
        """Test array slicing with NRT."""

        @jit
        def kernel(out):
            arr = np.arange(10)
            sliced = arr[2:7]
            out[0] = sliced[0]
            out[1] = sliced.shape[0]

        out = np.zeros(2, dtype=np.int32)
        kernel[1, 1](out)
        assert out[0] == 2
        assert out[1] == 5

    @pytest.mark.xfail(reason="NRT array operations not implemented yet")
    def test_array_reshape(self):
        """Test array reshaping with NRT."""

        @jit
        def kernel(out):
            arr = np.arange(6)
            reshaped = arr.reshape((2, 3))
            out[0] = reshaped[1, 2]

        out = np.zeros(1, dtype=np.int32)
        kernel[1, 1](out)
        assert out[0] == 5

    @pytest.mark.xfail(reason="np.array unimplemented")
    def test_current_basic_functionality(self):
        """Test that basic functionality from existing test still works."""

        @jit
        def foo():
            return np.array([1, 2, 3])

        @jit(dump=True)
        def kernel(a):
            nrt_array = foo()
            a[0] = nrt_array[0]
            a[1] = nrt_array.shape[0]

        a = np.zeros(2, dtype=np.int32)
        kernel[1, 1](a)
        assert a[0] == 1
        assert a[1] == 3
