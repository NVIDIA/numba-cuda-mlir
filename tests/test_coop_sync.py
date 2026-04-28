# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import cuda.simt as cuda
from cuda.simt import int32, float32


def useless_syncthreads(ary):
    i = cuda.grid(1)
    cuda.syncthreads()
    ary[i] = i


def useless_syncwarp(ary):
    i = cuda.grid(1)
    cuda.syncwarp()
    ary[i] = i


def useless_syncwarp_with_mask(ary):
    i = cuda.grid(1)
    cuda.syncwarp(0xFFFF)
    ary[i] = i


def coop_syncwarp(res):
    sm = cuda.shared_array(32, int32)
    i = cuda.grid(1)

    sm[i] = i
    cuda.syncwarp()

    if i < 16:
        sm[i] = sm[i] + sm[i + 16]
        cuda.syncwarp(0xFFFF)

    if i < 8:
        sm[i] = sm[i] + sm[i + 8]
        cuda.syncwarp(0xFF)

    if i < 4:
        sm[i] = sm[i] + sm[i + 4]
        cuda.syncwarp(0xF)

    if i < 2:
        sm[i] = sm[i] + sm[i + 2]
        cuda.syncwarp(0x3)

    if i == 0:
        res[0] = sm[0] + sm[1]


def simple_smem(ary):
    N = 100
    sm = cuda.shared_array(N, int32)
    i = cuda.grid(1)
    if i == 0:
        for j in range(N):
            sm[j] = j
    cuda.syncthreads()
    ary[i] = sm[i]


def coop_smem2d(ary):
    i, j = cuda.grid(2)
    sm = cuda.shared_array((10, 20), float32)
    sm[i, j] = (i + 1) / (j + 1)
    cuda.syncthreads()
    ary[i, j] = sm[i, j]


def dyn_shared_memory(ary):
    i = cuda.grid(1)
    sm = cuda.shared_array(0, float32)
    sm[i] = i * 2
    cuda.syncthreads()
    ary[i] = sm[i]


def use_threadfence(ary):
    ary[0] += 123
    cuda.threadfence()
    ary[0] += 321


def use_threadfence_block(ary):
    ary[0] += 123
    cuda.threadfence_block()
    ary[0] += 321


def use_threadfence_system(ary):
    ary[0] += 123
    cuda.threadfence_system()
    ary[0] += 321


def use_syncthreads_count(ary_in, ary_out):
    i = cuda.grid(1)
    ary_out[i] = cuda.syncthreads_count(ary_in[i])


def use_syncthreads_and(ary_in, ary_out):
    i = cuda.grid(1)
    ary_out[i] = cuda.syncthreads_and(ary_in[i])


def use_syncthreads_or(ary_in, ary_out):
    i = cuda.grid(1)
    ary_out[i] = cuda.syncthreads_or(ary_in[i])


def _safe_cc_check(cc):
    return cuda.get_current_device().compute_capability >= cc


def _test_useless(kernel):
    compiled = cuda.jit("void(int32[::1])")(kernel)
    nelem = 10
    ary = np.empty(nelem, dtype=np.int32)
    exp = np.arange(nelem, dtype=np.int32)
    ary = cuda.to_device(ary)
    compiled[1, nelem](ary)
    ary = ary.copy_to_host()
    np.testing.assert_equal(ary, exp)


def test_useless_syncthreads():
    _test_useless(useless_syncthreads)


def test_useless_syncwarp():
    _test_useless(useless_syncwarp)


@pytest.mark.skipif(not _safe_cc_check((7, 0)), reason="Partial masks require CC 7.0 or greater")
def test_useless_syncwarp_with_mask():
    _test_useless(useless_syncwarp_with_mask)


@pytest.mark.skipif(not _safe_cc_check((7, 0)), reason="Partial masks require CC 7.0 or greater")
def test_coop_syncwarp():
    # coop_syncwarp computes the sum of all integers from 0 to 31 (496)
    # using a single warp
    expected = 496
    nthreads = 32
    nblocks = 1

    compiled = cuda.jit("void(int32[::1])")(coop_syncwarp)
    res = np.zeros(1, dtype=np.int32)
    res = cuda.to_device(res)
    compiled[nblocks, nthreads](res)
    res = res.copy_to_host()
    np.testing.assert_equal(expected, res[0])


def test_simple_smem():
    compiled = cuda.jit("void(int32[::1])")(simple_smem)
    nelem = 100
    ary = np.empty(nelem, dtype=np.int32)
    ary = cuda.to_device(ary)
    compiled[1, nelem](ary)
    ary = ary.copy_to_host()
    assert np.all(ary == np.arange(nelem, dtype=np.int32))


def test_coop_smem2d():
    compiled = cuda.jit("void(float32[:,::1])")(coop_smem2d)
    shape = 10, 20
    ary = np.empty(shape, dtype=np.float32)
    ary = cuda.to_device(ary)
    compiled[1, shape](ary)
    ary = ary.copy_to_host()
    exp = np.empty_like(ary)
    for i in range(ary.shape[0]):
        for j in range(ary.shape[1]):
            exp[i, j] = (i + 1) / (j + 1)
    assert np.allclose(ary, exp)


@pytest.mark.skip()
def test_dyn_shared_memory():
    compiled = cuda.jit("void(float32[::1])")(dyn_shared_memory)
    shape = 10
    ary = np.empty(shape, dtype=np.float32)
    ary = cuda.to_device(ary)
    compiled[1, shape, 0, ary.size * 4](ary)
    ary = ary.copy_to_host()
    assert np.all(ary == 2 * np.arange(ary.size, dtype=np.int32))


@pytest.mark.xfail()
def test_threadfence_codegen():
    # Does not test runtime behavior, just the code generation.
    sig = (int32[:],)
    compiled = cuda.jit(sig)(use_threadfence)
    ary = np.zeros(10, dtype=np.int32)
    ary = cuda.to_device(ary)
    compiled[1, 1](ary)
    ary = ary.copy_to_host()
    assert 123 + 321 == ary[0]
    assert "membar.gl;" in compiled.inspect_asm(sig)


@pytest.mark.xfail()
def test_threadfence_block_codegen():
    # Does not test runtime behavior, just the code generation.
    sig = (int32[:],)
    compiled = cuda.jit(sig)(use_threadfence_block)
    ary = np.zeros(10, dtype=np.int32)
    ary = cuda.to_device(ary)
    compiled[1, 1](ary)
    ary = ary.copy_to_host()
    assert 123 + 321 == ary[0]
    assert "membar.cta;" in compiled.inspect_asm(sig)


@pytest.mark.xfail()
def test_threadfence_system_codegen():
    # Does not test runtime behavior, just the code generation.
    sig = (int32[:],)
    compiled = cuda.jit(sig)(use_threadfence_system)
    ary = np.zeros(10, dtype=np.int32)
    ary = cuda.to_device(ary)
    compiled[1, 1](ary)
    ary = ary.copy_to_host()
    assert 123 + 321 == ary[0]
    assert "membar.sys;" in compiled.inspect_asm(sig)


def _test_syncthreads_count(in_dtype):
    compiled = cuda.jit(use_syncthreads_count)
    ary_in = np.ones(72, dtype=in_dtype)
    ary_out = np.zeros(72, dtype=np.int32)
    ary_in[31] = 0
    ary_in[42] = 0
    ary_in = cuda.to_device(ary_in)
    ary_out = cuda.to_device(ary_out)
    compiled[1, 72](ary_in, ary_out)
    ary_out = ary_out.copy_to_host()
    assert np.all(ary_out == 70)


def test_syncthreads_count():
    _test_syncthreads_count(np.int32)


def test_syncthreads_count_upcast():
    _test_syncthreads_count(np.int16)


def test_syncthreads_count_downcast():
    _test_syncthreads_count(np.int64)


def _test_syncthreads_and(in_dtype):
    compiled = cuda.jit(use_syncthreads_and)
    nelem = 100
    ary_in = np.ones(nelem, dtype=in_dtype)
    ary_out = np.zeros(nelem, dtype=np.int32)
    ary_in = cuda.to_device(ary_in)
    ary_out = cuda.to_device(ary_out)
    compiled[1, nelem](ary_in, ary_out)
    ary_out = ary_out.copy_to_host()
    assert np.all(ary_out == 1)
    ary_in[31] = 0
    ary_out = cuda.to_device(ary_out)
    compiled[1, nelem](ary_in, ary_out)
    ary_out = ary_out.copy_to_host()
    assert np.all(ary_out == 0)


def test_syncthreads_and():
    _test_syncthreads_and(np.int32)


def test_syncthreads_and_upcast():
    _test_syncthreads_and(np.int16)


def test_syncthreads_and_downcast():
    _test_syncthreads_and(np.int64)


def _test_syncthreads_or(in_dtype):
    compiled = cuda.jit(use_syncthreads_or)
    nelem = 100
    ary_in = np.zeros(nelem, dtype=in_dtype)
    ary_out = np.zeros(nelem, dtype=np.int32)
    ary_in = cuda.to_device(ary_in)
    ary_out = cuda.to_device(ary_out)
    compiled[1, nelem](ary_in, ary_out)
    ary_out = ary_out.copy_to_host()
    ary_in = ary_in.copy_to_host()
    assert np.all(ary_out == 0)
    ary_in[31] = 1
    ary_in = cuda.to_device(ary_in)
    ary_out = cuda.to_device(ary_out)
    compiled[1, nelem](ary_in, ary_out)
    ary_out = ary_out.copy_to_host()
    assert np.all(ary_out == 1)


def test_syncthreads_or():
    _test_syncthreads_or(np.int32)


def test_syncthreads_or_upcast():
    _test_syncthreads_or(np.int16)


def test_syncthreads_or_downcast():
    _test_syncthreads_or(np.int64)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.DEBUG)
    # test_syncthreads_and_downcast()
    # test_syncthreads_or_downcast()
    # test_syncthreads_count_downcast()
    # test_coop_syncwarp()
    # test_syncthreads_and()
    # test_syncthreads_or()
    # test_syncthreads_count()
    # test_useless_syncthreads()
    # test_useless_syncwarp()
    # test_useless_syncwarp_with_mask()
    test_simple_smem()
    # test_coop_smem2d()
    # test_dyn_shared_memory()
    # test_threadfence_codegen()
    # test_threadfence_block_codegen()
    # test_threadfence_system_codegen()
