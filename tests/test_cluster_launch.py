# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for cluster launch support.

Cluster launch requires compute capability 9.0+ (Hopper or newer).
This feature allows launching kernels with cooperative thread arrays (CTAs)
organized into clusters that can synchronize and share data.
"""

import pytest
import numpy as np
import cuda.simt as cuda
from gpu_utils import require_cc_min


def simple_kernel_for_cluster(out):
    """A simple kernel that writes thread/block info to verify cluster launch works."""
    i = cuda.grid(1)
    if i < out.size:
        out[i] = cuda.blockIdx.x


def cluster_sync_kernel(out):
    """A kernel that uses cluster synchronization primitives."""
    i = cuda.grid(1)
    if i < out.size:
        # Write block index before sync
        out[i] = cuda.blockIdx.x

        # Cluster synchronization
        cuda.intrin.cluster_arrive_relaxed()
        cuda.intrin.cluster_wait()


def cluster_idx_kernel(out):
    """A kernel that queries which cluster this block belongs to."""
    i = cuda.grid(1)
    if i < out.size:
        # Note: block_idx_in_cluster actually returns the cluster ID (clusterid.x)
        out[i] = cuda.intrin.block_idx_in_cluster()


@pytest.mark.requires_cc_min((9, 0), "Cluster launch")
class TestClusterLaunch:
    """Tests for cluster launch functionality."""

    def test_cluster_launch_1x1x1(self):
        """Test cluster launch with minimal 1x1x1 cluster (effectively no clustering)."""
        kernel = cuda.jit(simple_kernel_for_cluster)

        nblocks = 4
        nthreads = 32
        out = np.zeros(nblocks * nthreads, dtype=np.int32)
        out_dev = cuda.to_device(out)

        # Launch with 1x1x1 cluster
        kernel[nblocks, nthreads, None, None, (1, 1, 1)](out_dev)

        result = out_dev.copy_to_host()
        # Each block writes its index
        expected = np.repeat(np.arange(nblocks, dtype=np.int32), nthreads)
        np.testing.assert_array_equal(result, expected)

    def test_cluster_launch_2x1x1(self):
        """Test cluster launch with 2x1x1 cluster (2 CTAs per cluster)."""
        kernel = cuda.jit(simple_kernel_for_cluster)

        nblocks = 4  # Must be multiple of cluster size
        nthreads = 32
        out = np.zeros(nblocks * nthreads, dtype=np.int32)
        out_dev = cuda.to_device(out)

        # Launch with 2x1x1 cluster
        kernel[nblocks, nthreads, None, None, (2, 1, 1)](out_dev)

        result = out_dev.copy_to_host()
        # Each block writes its index
        expected = np.repeat(np.arange(nblocks, dtype=np.int32), nthreads)
        np.testing.assert_array_equal(result, expected)

    def test_cluster_launch_tuple_syntax(self):
        """Test cluster launch with tuple as cluster dimension."""
        kernel = cuda.jit(simple_kernel_for_cluster)

        nblocks = 2
        nthreads = 64
        out = np.zeros(nblocks * nthreads, dtype=np.int32)
        out_dev = cuda.to_device(out)

        # Launch with tuple cluster dims
        kernel[(nblocks,), (nthreads,), None, None, (2,)](out_dev)

        result = out_dev.copy_to_host()
        expected = np.repeat(np.arange(nblocks, dtype=np.int32), nthreads)
        np.testing.assert_array_equal(result, expected)

    def test_cluster_launch_configure_method(self):
        """Test cluster launch using the configure method."""
        kernel = cuda.jit(simple_kernel_for_cluster)

        nblocks = 4
        nthreads = 32
        out = np.zeros(nblocks * nthreads, dtype=np.int32)
        out_dev = cuda.to_device(out)

        # Use configure method with cluster parameter
        configured = kernel.configure(
            griddim=(nblocks, 1, 1),
            blockdim=(nthreads, 1, 1),
            cluster=(2, 1, 1),
        )
        configured(out_dev)

        result = out_dev.copy_to_host()
        expected = np.repeat(np.arange(nblocks, dtype=np.int32), nthreads)
        np.testing.assert_array_equal(result, expected)

    def test_cluster_sync_primitives(self):
        """Test that cluster synchronization primitives work."""
        kernel = cuda.jit(cluster_sync_kernel)

        nblocks = 4
        nthreads = 32
        out = np.zeros(nblocks * nthreads, dtype=np.int32)
        out_dev = cuda.to_device(out)

        # Launch with 2x1x1 cluster to enable cluster sync
        kernel[nblocks, nthreads, None, None, (2, 1, 1)](out_dev)

        result = out_dev.copy_to_host()
        expected = np.repeat(np.arange(nblocks, dtype=np.int32), nthreads)
        np.testing.assert_array_equal(result, expected)

    def test_cluster_idx(self):
        """Test that block_idx_in_cluster returns the cluster ID."""
        kernel = cuda.jit(cluster_idx_kernel)

        cluster_size = 2
        nblocks = 4  # 2 clusters of 2 blocks each
        nthreads = 32
        out = np.zeros(nblocks * nthreads, dtype=np.int32)
        out_dev = cuda.to_device(out)

        kernel[nblocks, nthreads, None, None, (cluster_size, 1, 1)](out_dev)

        result = out_dev.copy_to_host()
        # block_idx_in_cluster returns the cluster ID (clusterid.x)
        # With 4 blocks and cluster_size=2: cluster 0 has blocks 0-1, cluster 1 has blocks 2-3
        # Each cluster's blocks report the same cluster ID
        num_clusters = nblocks // cluster_size
        expected = np.repeat(
            np.repeat(np.arange(num_clusters, dtype=np.int32), cluster_size), nthreads
        )
        np.testing.assert_array_equal(result, expected)


@pytest.mark.requires_cc_min((9, 0), "Cluster launch")
def test_cluster_launch_no_cluster_param():
    """Test that kernel launch without cluster param still works on CC 9.0+."""
    kernel = cuda.jit(simple_kernel_for_cluster)

    nblocks = 4
    nthreads = 32
    out = np.zeros(nblocks * nthreads, dtype=np.int32)
    out_dev = cuda.to_device(out)

    # Standard launch without cluster parameter
    kernel[nblocks, nthreads](out_dev)

    result = out_dev.copy_to_host()
    expected = np.repeat(np.arange(nblocks, dtype=np.int32), nthreads)
    np.testing.assert_array_equal(result, expected)


if __name__ == "__main__":
    require_cc_min((9, 0), "Cluster launch")

    test = TestClusterLaunch()
    test.test_cluster_launch_1x1x1()
    print("test_cluster_launch_1x1x1 PASSED")

    test.test_cluster_launch_2x1x1()
    print("test_cluster_launch_2x1x1 PASSED")

    test.test_cluster_launch_tuple_syntax()
    print("test_cluster_launch_tuple_syntax PASSED")

    test.test_cluster_launch_configure_method()
    print("test_cluster_launch_configure_method PASSED")

    test.test_cluster_sync_primitives()
    print("test_cluster_sync_primitives PASSED")

    test.test_cluster_idx()
    print("test_cluster_idx PASSED")

    test_cluster_launch_no_cluster_param()
    print("test_cluster_launch_no_cluster_param PASSED")

    print("\nAll tests PASSED!")
