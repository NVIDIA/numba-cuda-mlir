# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for cusimt caching infrastructure."""

import tempfile
import os
import pytest
import numpy as np
from pathlib import Path

import cuda.simt as cuda
from cusimt.caching import MLIRCache, NullCache, CachedCompileResult
from cusimt.tools import get_gpu_compute_capability, get_cuda_runtime_version


class TestCachingUtilities:
    """Test caching utility functions."""

    def test_get_cuda_runtime_version(self):
        """Test that we can get the CUDA runtime version."""
        version = get_cuda_runtime_version()
        assert isinstance(version, tuple)
        assert len(version) == 2
        major, minor = version
        assert isinstance(major, int)
        assert isinstance(minor, int)
        assert major >= 11  # Minimum supported CUDA version

    def test_null_cache(self):
        """Test NullCache does nothing."""
        cache = NullCache()
        assert cache.cache_path is None
        assert cache.load_overload(None, None) is None
        cache.save_overload(None, None)  # Should not raise
        cache.flush()  # Should not raise


class TestCachingIntegration:
    """Integration tests for caching with actual kernels."""

    def test_cache_enabled_creates_files(self, tmp_path):
        """Test that cache=True creates cache files."""
        # Create a test module file in tmp_path
        test_file = tmp_path / "test_kernel.py"
        test_file.write_text(
            """
import cuda.simt as cuda

@cuda.jit(cache=True)
def add_kernel(a, b, out):
    idx = cuda.grid(1)
    if idx < out.shape[0]:
        out[idx] = a[idx] + b[idx]
"""
        )

        # Import and run the kernel
        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            import test_kernel

            a = cuda.to_device(np.array([1, 2, 3], dtype=np.float32))
            b = cuda.to_device(np.array([4, 5, 6], dtype=np.float32))
            out = cuda.to_device(np.zeros(3, dtype=np.float32))

            test_kernel.add_kernel[1, 3](a, b, out)
            result = out.copy_to_host()

            np.testing.assert_array_equal(result, [5, 7, 9])

            # Check that cache files were created
            pycache = tmp_path / "__pycache__"
            if pycache.exists():
                cache_files = list(pycache.glob("*.nbi")) + list(pycache.glob("*.nbc"))
                assert (
                    len(cache_files) >= 2
                ), f"Expected cache files, found: {cache_files}"
        finally:
            sys.path.remove(str(tmp_path))
            if "test_kernel" in sys.modules:
                del sys.modules["test_kernel"]

    def test_cache_disabled_no_files(self, tmp_path):
        """Test that cache=False does not create cache files."""
        test_file = tmp_path / "test_kernel_nocache.py"
        test_file.write_text(
            """
import cuda.simt as cuda

@cuda.jit(cache=False)
def add_kernel_nocache(a, b, out):
    idx = cuda.grid(1)
    if idx < out.shape[0]:
        out[idx] = a[idx] + b[idx]
"""
        )

        import sys

        sys.path.insert(0, str(tmp_path))
        try:
            import test_kernel_nocache

            a = cuda.to_device(np.array([1, 2, 3], dtype=np.float32))
            b = cuda.to_device(np.array([4, 5, 6], dtype=np.float32))
            out = cuda.to_device(np.zeros(3, dtype=np.float32))

            test_kernel_nocache.add_kernel_nocache[1, 3](a, b, out)
            result = out.copy_to_host()

            np.testing.assert_array_equal(result, [5, 7, 9])

            # Check that no cache files were created
            pycache = tmp_path / "__pycache__"
            if pycache.exists():
                cache_files = list(pycache.glob("*test_kernel_nocache*.nbi"))
                assert len(cache_files) == 0, f"Unexpected cache files: {cache_files}"
        finally:
            sys.path.remove(str(tmp_path))
            if "test_kernel_nocache" in sys.modules:
                del sys.modules["test_kernel_nocache"]

    def test_dispatcher_stats(self):
        """Test that dispatcher tracks cache hits/misses."""

        @cuda.jit
        def simple_kernel(arr):
            idx = cuda.grid(1)
            if idx < arr.shape[0]:
                arr[idx] = idx

        arr = cuda.to_device(np.zeros(10, dtype=np.int32))
        simple_kernel[1, 10](arr)

        # Check stats are available
        stats = simple_kernel.stats
        assert hasattr(stats, "cache_path")
        assert hasattr(stats, "cache_hits")
        assert hasattr(stats, "cache_misses")

    def test_enable_caching_method(self):
        """Test that enable_caching() method works."""

        @cuda.jit
        def another_kernel(arr):
            idx = cuda.grid(1)
            if idx < arr.shape[0]:
                arr[idx] = idx * 2

        # Initially uses NullCache
        assert another_kernel._cache.cache_path is None

        # Enable caching - this may fail if no source file, which is fine
        try:
            another_kernel.enable_caching()
            assert another_kernel._cache.cache_path is not None
        except RuntimeError as e:
            # Expected for functions defined in <string> or similar
            assert "no locator available" in str(e)

    def test_cannot_cache_global_device_array(self):
        """Test that kernels capturing global device arrays cannot be cached.

        This replicates numba-cuda's test_caching.py::CUDACachingTest::test_cannot_cache_global_device_array
        """
        from pickle import PicklingError

        host_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        global_device_array = cuda.to_device(host_data)

        with pytest.raises(PicklingError, match="global device arrays"):

            @cuda.jit(cache=True)
            def cached_kernel_global(output):
                i = cuda.grid(1)
                if i < output.size:
                    output[i] = global_device_array[i] * 2.0

            output = cuda.device_array(3, dtype=np.float32)
            cached_kernel_global[1, 3](output)

    def test_closure_caching(self):
        """Test that closures can be cached without warnings.

        This replicates numba-cuda's test_caching.py::CUDACachingTest::test_closure
        """
        import warnings
        from cusimt.numba_cuda.core.errors import NumbaPerformanceWarning, NumbaWarning

        # Define closures with different captured values
        def make_closure(val):
            @cuda.jit(cache=True)
            def closure_kernel(arr):
                i = cuda.grid(1)
                if i < arr.shape[0]:
                    arr[i] = arr[i] + val

            return closure_kernel

        closure1 = make_closure(3)
        closure2 = make_closure(5)

        arr1 = cuda.to_device(np.array([3], dtype=np.int32))
        arr2 = cuda.to_device(np.array([3], dtype=np.int32))

        with warnings.catch_warnings():
            warnings.simplefilter("error", NumbaWarning)
            warnings.simplefilter("ignore", NumbaPerformanceWarning)
            closure1[1, 1](arr1)
            closure2[1, 1](arr2)

        result1 = arr1.copy_to_host()
        result2 = arr2.copy_to_host()

        assert result1[0] == 6  # 3 + 3
        assert result2[0] == 8  # 3 + 5

    def test_cache_reuse(self, tmp_path):
        """Test that cached kernels are reused on subsequent imports.

        This replicates numba-cuda's test_caching.py::CUDACachingTest::test_cache_reuse
        but without Record types.
        """
        import sys

        test_file = tmp_path / "cache_reuse_module.py"
        test_file.write_text(
            """
import cuda.simt as cuda
import numpy as np

Z = 1

@cuda.jit(cache=True)
def add_kernel(out, a, b):
    idx = cuda.grid(1)
    if idx < out.shape[0]:
        out[idx] = (a[idx] + b[idx]) * Z
"""
        )

        sys.path.insert(0, str(tmp_path))
        try:
            # First import - compiles and caches
            import cache_reuse_module as mod

            a = cuda.to_device(np.array([2, 3], dtype=np.int32))
            b = cuda.to_device(np.array([3, 4], dtype=np.int32))
            out = cuda.to_device(np.zeros(2, dtype=np.int32))
            mod.add_kernel[1, 2](out, a, b)
            result1 = out.copy_to_host()
            np.testing.assert_array_equal(result1, [5, 7])

            # Run with floats (second signature)
            a_f = cuda.to_device(np.array([2.5, 3.5], dtype=np.float32))
            b_f = cuda.to_device(np.array([3.0, 4.0], dtype=np.float32))
            out_f = cuda.to_device(np.zeros(2, dtype=np.float32))
            mod.add_kernel[1, 2](out_f, a_f, b_f)
            result2 = out_f.copy_to_host()
            np.testing.assert_array_equal(result2, [5.5, 7.5])

            # Check that cache files were created
            pycache = tmp_path / "__pycache__"
            assert pycache.exists(), "Cache directory should exist"
            cache_files = list(pycache.glob("*.nbi")) + list(pycache.glob("*.nbc"))
            assert len(cache_files) >= 2, f"Expected cache files, found: {cache_files}"

            # Check initial cache misses
            stats = mod.add_kernel.stats
            initial_misses = sum(stats.cache_misses.values())
            assert initial_misses == 2, f"Expected 2 cache misses, got {initial_misses}"

            # Remove module from cache to force reimport
            del sys.modules["cache_reuse_module"]

            # Second import - should load from cache
            import cache_reuse_module as mod2

            a2 = cuda.to_device(np.array([2, 3], dtype=np.int32))
            b2 = cuda.to_device(np.array([3, 4], dtype=np.int32))
            out2 = cuda.to_device(np.zeros(2, dtype=np.int32))
            mod2.add_kernel[1, 2](out2, a2, b2)
            result3 = out2.copy_to_host()
            np.testing.assert_array_equal(result3, [5, 7])

            # Check cache hit
            stats2 = mod2.add_kernel.stats
            hits = sum(stats2.cache_hits.values())
            assert hits >= 1, f"Expected at least 1 cache hit, got {hits}"

        finally:
            sys.path.remove(str(tmp_path))
            if "cache_reuse_module" in sys.modules:
                del sys.modules["cache_reuse_module"]

    def test_multiple_signatures_caching(self, tmp_path):
        """Test that multiple signatures are cached correctly.

        This tests the cache's ability to handle multiple overloads of the same function.
        """
        import sys

        test_file = tmp_path / "multi_sig_module.py"
        test_file.write_text(
            """
import cuda.simt as cuda

@cuda.jit(cache=True)
def typed_kernel(arr):
    idx = cuda.grid(1)
    if idx < arr.shape[0]:
        arr[idx] = arr[idx] * 2
"""
        )

        sys.path.insert(0, str(tmp_path))
        try:
            import multi_sig_module as mod

            # Run with int32
            arr_int = cuda.to_device(np.array([1, 2, 3], dtype=np.int32))
            mod.typed_kernel[1, 3](arr_int)
            result_int = arr_int.copy_to_host()
            np.testing.assert_array_equal(result_int, [2, 4, 6])

            # Run with float32
            arr_float = cuda.to_device(np.array([1.0, 2.0, 3.0], dtype=np.float32))
            mod.typed_kernel[1, 3](arr_float)
            result_float = arr_float.copy_to_host()
            np.testing.assert_array_equal(result_float, [2.0, 4.0, 6.0])

            # Run with float64
            arr_double = cuda.to_device(np.array([1.0, 2.0, 3.0], dtype=np.float64))
            mod.typed_kernel[1, 3](arr_double)
            result_double = arr_double.copy_to_host()
            np.testing.assert_array_equal(result_double, [2.0, 4.0, 6.0])

            # Check cache misses for all 3 signatures
            stats = mod.typed_kernel.stats
            misses = sum(stats.cache_misses.values())
            assert misses == 3, f"Expected 3 cache misses, got {misses}"

            # Check cache files were created
            pycache = tmp_path / "__pycache__"
            cache_files = list(pycache.glob("*.nbi")) + list(pycache.glob("*.nbc"))
            # Should have 1 nbi and at least 3 nbc files (one per signature)
            assert (
                len(cache_files) >= 4
            ), f"Expected at least 4 cache files, found: {cache_files}"

        finally:
            sys.path.remove(str(tmp_path))
            if "multi_sig_module" in sys.modules:
                del sys.modules["multi_sig_module"]

    def test_recompile(self, tmp_path):
        """Test that recompile() picks up updated global values.

        This replicates numba-cuda's test_caching.py::CUDACachingTest::test_recompile
        """
        import sys

        test_file = tmp_path / "recompile_module.py"
        test_file.write_text(
            """
import cuda.simt as cuda

Z = 1

@cuda.jit
def f(x, y, out):
    i = cuda.grid(1)
    if i < out.shape[0]:
        out[i] = x[i] + y[i] + Z
"""
        )

        sys.path.insert(0, str(tmp_path))
        try:
            import recompile_module as mod

            x = cuda.to_device(np.array([2], dtype=np.int32))
            y = cuda.to_device(np.array([3], dtype=np.int32))
            out = cuda.to_device(np.zeros(1, dtype=np.int32))

            # First call with Z=1: 2+3+1=6
            mod.f[1, 1](x, y, out)
            result1 = out.copy_to_host()
            assert result1[0] == 6, f"Expected 6, got {result1[0]}"

            # Change Z to 10
            mod.Z = 10

            # Call again - should still use cached version with Z=1
            out = cuda.to_device(np.zeros(1, dtype=np.int32))
            mod.f[1, 1](x, y, out)
            result2 = out.copy_to_host()
            assert result2[0] == 6, f"Expected 6 (cached), got {result2[0]}"

            # Recompile to pick up the new Z value
            mod.f.recompile()

            # Now should use Z=10: 2+3+10=15
            out = cuda.to_device(np.zeros(1, dtype=np.int32))
            mod.f[1, 1](x, y, out)
            result3 = out.copy_to_host()
            assert result3[0] == 15, f"Expected 15 (recompiled), got {result3[0]}"

        finally:
            sys.path.remove(str(tmp_path))
            if "recompile_module" in sys.modules:
                del sys.modules["recompile_module"]
