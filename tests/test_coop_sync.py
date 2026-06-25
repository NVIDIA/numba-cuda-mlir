# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
from numba_cuda_mlir import cuda
from numba_cuda_mlir.cuda import int32, float32
from numba_cuda_mlir.numba_cuda.testing import cc_X_or_above


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


def dyn_shared_memory_aligned(ary):
    i = cuda.grid(1)
    sm = cuda.shared.array(0, float32, 16)
    sm[i] = i * 2
    cuda.syncthreads()
    ary[i] = sm[i]


def dyn_shared_memory_row_view(ary):
    i, j = cuda.grid(2)
    sm = cuda.shared_array((ary.shape[0], 4), float32)
    row = sm[i]
    row[j] = ary[i, j]
    cuda.syncthreads()
    ary[i, j] = row[j]


def dyn_shared_memory_ravel_view(ary):
    i = cuda.grid(1)
    sm = cuda.shared_array((ary.shape[0], 4), float32)
    flat = sm.ravel()
    flat[i] = i * 2
    cuda.syncthreads()
    ary[i] = flat[i]


def dyn_shared_memory_reshape_tuple_view(ary):
    i = cuda.grid(1)
    sm = cuda.shared_array((ary.shape[0], 1), float32)
    reshaped = sm.reshape((ary.shape[0], 1))
    reshaped[i, 0] = i * 5
    cuda.syncthreads()
    ary[i] = reshaped[i, 0]


def local_memory_reshape_tuple_view(ary):
    i = cuda.grid(1)
    local = cuda.local.array((16, 1), float32)
    local[i, 0] = ary[i]
    reshaped = local.reshape((8, 2))
    row = i // 2
    col = i - row * 2
    ary[i] = reshaped[row, col] + 1


def local_memory_reshape_multiassign_tuple_view(ary):
    i = cuda.grid(1)
    local = cuda.local.array((16, 1), float32)
    local[i, 0] = ary[i]
    reshaped = local.reshape((8, 2))
    if i == 0:
        reshaped = local.reshape((8, 2))
    row = i // 2
    col = i - row * 2
    ary[i] = reshaped[row, col] + 1


def dyn_shared_memory_dtype_view(ary):
    i = cuda.grid(1)
    sm = cuda.shared_array(0, float32)
    view = sm.view(np.int32)
    view[i] = i * 2
    cuda.syncthreads()
    ary[i] = view[i]


def dyn_shared_memory_slice_view(ary):
    i = cuda.grid(1)
    sm = cuda.shared_array(0, float32)
    window = sm[1:]
    window[i] = i * 3
    cuda.syncthreads()
    ary[i] = window[i]


def dyn_shared_memory_slice_assign(ary):
    i = cuda.grid(1)
    sm = cuda.shared_array(0, float32)
    if i == 0:
        sm[0 : ary.shape[0]] = 7
    cuda.syncthreads()
    ary[i] = sm[i]


def dyn_shared_memory_complex_real_imag_view(ary):
    i = cuda.grid(1)
    sm = cuda.shared_array(0, np.complex64)
    real = sm.real
    imag = sm.imag
    real[i] = i
    imag[i] = i + 1
    cuda.syncthreads()
    ary[i] = real[i] + imag[i]


def dyn_shared_memory_complex_element(src, ary):
    i = cuda.grid(1)
    sm = cuda.shared_array(0, np.complex64)
    sm[i] = src[i]
    cuda.syncthreads()
    value = sm[i]
    ary[i] = value.real + value.imag


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
    return cc_X_or_above(*cc)


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


def test_dynamic_shared_memory_aligned_executes():
    compiled = cuda.jit("void(float32[::1])")(dyn_shared_memory_aligned)
    ary = np.zeros(16, dtype=np.float32)
    ary_dev = cuda.to_device(ary)

    compiled[1, ary.size, 0, ary.size * ary.dtype.itemsize](ary_dev)

    expected = 2 * np.arange(ary.size, dtype=np.float32)
    np.testing.assert_allclose(ary_dev.copy_to_host(), expected)


def test_dynamic_shared_memory_row_view_executes():
    compiled = cuda.jit("void(float32[:, ::1])")(dyn_shared_memory_row_view)
    ary = np.arange(16, dtype=np.float32).reshape(4, 4)
    ary_dev = cuda.to_device(ary.copy())

    compiled[(1, 1), ary.shape, 0, ary.size * ary.dtype.itemsize](ary_dev)

    np.testing.assert_allclose(ary_dev.copy_to_host(), ary)


def test_dynamic_shared_memory_ravel_view_executes():
    compiled = cuda.jit("void(float32[::1])")(dyn_shared_memory_ravel_view)
    ary = np.zeros(16, dtype=np.float32)
    ary_dev = cuda.to_device(ary)
    shared_bytes = ary.size * 4 * ary.dtype.itemsize

    compiled[1, ary.size, 0, shared_bytes](ary_dev)

    expected = 2 * np.arange(ary.size, dtype=np.float32)
    np.testing.assert_allclose(ary_dev.copy_to_host(), expected)


def test_dynamic_shared_memory_reshape_tuple_view_executes():
    compiled = cuda.jit("void(float32[::1])")(dyn_shared_memory_reshape_tuple_view)
    ary = np.zeros(16, dtype=np.float32)
    ary_dev = cuda.to_device(ary)

    compiled[1, ary.size, 0, ary.size * ary.dtype.itemsize](ary_dev)

    expected = 5 * np.arange(ary.size, dtype=np.float32)
    np.testing.assert_allclose(ary_dev.copy_to_host(), expected)


def test_local_memory_reshape_tuple_view_executes():
    compiled = cuda.jit("void(float32[::1])")(local_memory_reshape_tuple_view)
    ary = np.arange(16, dtype=np.float32)
    ary_dev = cuda.to_device(ary)

    compiled[1, ary.size](ary_dev)

    np.testing.assert_allclose(ary_dev.copy_to_host(), ary + 1)


def test_local_memory_reshape_multiassign_tuple_view_executes():
    compiled = cuda.jit("void(float32[::1])")(local_memory_reshape_multiassign_tuple_view)
    ary = np.arange(16, dtype=np.float32)
    ary_dev = cuda.to_device(ary)

    compiled[1, ary.size](ary_dev)

    np.testing.assert_allclose(ary_dev.copy_to_host(), ary + 1)


def test_dynamic_shared_memory_dtype_view_executes():
    compiled = cuda.jit("void(int32[::1])")(dyn_shared_memory_dtype_view)
    ary = np.zeros(16, dtype=np.int32)
    ary_dev = cuda.to_device(ary)

    compiled[1, ary.size, 0, ary.size * ary.dtype.itemsize](ary_dev)

    expected = 2 * np.arange(ary.size, dtype=np.int32)
    np.testing.assert_equal(ary_dev.copy_to_host(), expected)


def test_dynamic_shared_memory_slice_view_executes():
    compiled = cuda.jit("void(float32[::1])")(dyn_shared_memory_slice_view)
    ary = np.zeros(16, dtype=np.float32)
    ary_dev = cuda.to_device(ary)
    shared_bytes = (ary.size + 1) * ary.dtype.itemsize

    compiled[1, ary.size, 0, shared_bytes](ary_dev)

    expected = 3 * np.arange(ary.size, dtype=np.float32)
    np.testing.assert_allclose(ary_dev.copy_to_host(), expected)


def test_dynamic_shared_memory_slice_assign_executes():
    compiled = cuda.jit("void(float32[::1])")(dyn_shared_memory_slice_assign)
    ary = np.zeros(16, dtype=np.float32)
    ary_dev = cuda.to_device(ary)

    compiled[1, ary.size, 0, ary.size * ary.dtype.itemsize](ary_dev)

    np.testing.assert_allclose(ary_dev.copy_to_host(), np.full_like(ary, 7))


def test_dynamic_shared_memory_complex_real_imag_views_execute():
    compiled = cuda.jit("void(float32[::1])")(dyn_shared_memory_complex_real_imag_view)
    ary = np.zeros(16, dtype=np.float32)
    ary_dev = cuda.to_device(ary)
    shared_bytes = ary.size * np.dtype(np.complex64).itemsize

    compiled[1, ary.size, 0, shared_bytes](ary_dev)

    expected = 2 * np.arange(ary.size, dtype=np.float32) + 1
    np.testing.assert_allclose(ary_dev.copy_to_host(), expected)


def test_dynamic_shared_memory_complex_element_executes():
    compiled = cuda.jit("void(complex64[::1], float32[::1])")(dyn_shared_memory_complex_element)
    src = (np.arange(16, dtype=np.float32) + 1j * (np.arange(16, dtype=np.float32) + 1)).astype(
        np.complex64
    )
    ary = np.zeros(src.shape, dtype=np.float32)
    src_dev = cuda.to_device(src)
    ary_dev = cuda.to_device(ary)
    shared_bytes = src.size * src.dtype.itemsize

    compiled[1, src.size, 0, shared_bytes](src_dev, ary_dev)

    expected = src.real + src.imag
    np.testing.assert_allclose(ary_dev.copy_to_host(), expected)


@pytest.mark.parametrize(
    ("kernel", "sig"),
    [
        (dyn_shared_memory, "void(float32[:])"),
        (dyn_shared_memory_aligned, "void(float32[:])"),
        (dyn_shared_memory_row_view, "void(float32[:, :])"),
        (dyn_shared_memory_ravel_view, "void(float32[:])"),
        (dyn_shared_memory_reshape_tuple_view, "void(float32[:])"),
        (dyn_shared_memory_dtype_view, "void(int32[:])"),
        (dyn_shared_memory_slice_view, "void(float32[:])"),
        (dyn_shared_memory_slice_assign, "void(float32[:])"),
        (dyn_shared_memory_complex_real_imag_view, "void(float32[:])"),
        (dyn_shared_memory_complex_element, "void(complex64[:], float32[:])"),
    ],
)
def test_dynamic_shared_memory_gep_has_no_no_wrap_flags(monkeypatch, kernel, sig):
    from numba_cuda_mlir import compiler, tools

    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (10, 0))

    mlir = compiler.compile_mlir(
        cuda.jit(chip="sm_100")(kernel),
        sig,
        optimized=False,
        chip="sm_100",
    )

    shared_geps = [
        line
        for line in mlir.splitlines()
        if "llvm.getelementptr" in line and "!llvm.ptr<3>" in line
    ]
    assert shared_geps
    assert all("inbounds" not in line and "nuw" not in line for line in shared_geps)


def test_dynamic_shared_memory_assume_alignment_result_is_used(monkeypatch):
    from numba_cuda_mlir import compiler, tools

    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (10, 0))

    mlir = compiler.compile_mlir(
        cuda.jit(chip="sm_100")(dyn_shared_memory_aligned),
        "void(float32[:])",
        optimized=False,
        chip="sm_100",
    )

    lines = mlir.splitlines()
    assume_line = next(line for line in lines if "memref.assume_alignment" in line)
    assume_result = assume_line.split("=", 1)[0].strip()
    assume_uses = [
        line for line in lines if assume_result in line and "memref.assume_alignment" not in line
    ]

    assert assume_uses
    assert any("memref.extract_strided_metadata" in line for line in assume_uses)


def test_dynamic_shared_memory_marker_ignores_unexpected_values():
    from numba_cuda_mlir.mlir_lowering import MLIRLower

    lower = MLIRLower.__new__(MLIRLower)
    lower._dynamic_shared_memory_values = set()
    value = []

    assert lower._mark_dynamic_shared_memory(value) is value
    assert not lower._is_dynamic_shared_memory(value)
    assert not lower._dynamic_shared_memory_values


def test_dynamic_shared_memory_marker_handles_tuples_and_non_memrefs():
    from numba_cuda_mlir._mlir import ir
    from numba_cuda_mlir._mlir.dialects import arith, memref
    from numba_cuda_mlir._mlir.extras import types as T
    from numba_cuda_mlir.mlir_lowering import MLIRLower

    lower = MLIRLower.__new__(MLIRLower)
    lower._dynamic_shared_memory_values = set()

    with ir.Context(), ir.Location.unknown():
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            scalar = arith.constant(result=T.i32(), value=1)
            memref_type = ir.MemRefType.get(
                [1],
                T.i32(),
                memory_space=ir.Attribute.parse("#gpu.address_space<workgroup>"),
            )
            shared = memref.alloca(memref=memref_type, dynamic_sizes=[], symbol_operands=[])
            slot_shared = memref.alloca(memref=memref_type, dynamic_sizes=[], symbol_operands=[])
            slot_scalar = memref.alloca(memref=memref_type, dynamic_sizes=[], symbol_operands=[])
            loaded_shared = memref.alloca(memref=memref_type, dynamic_sizes=[], symbol_operands=[])
            loaded_scalar = arith.constant(result=T.i32(), value=2)
            mismatched_slot = memref.alloca(
                memref=memref_type, dynamic_sizes=[], symbol_operands=[]
            )
            mismatched_loaded = memref.alloca(
                memref=memref_type, dynamic_sizes=[], symbol_operands=[]
            )

            lower._mark_dynamic_shared_memory(shared)
            lower._mark_dynamic_shared_memory_slot((slot_shared, slot_scalar), (shared, scalar))
            lower._mark_dynamic_shared_memory_alias(
                (loaded_shared, loaded_scalar), (slot_shared, slot_scalar)
            )

            assert lower._is_dynamic_shared_memory(slot_shared)
            assert not lower._is_dynamic_shared_memory(slot_scalar)
            assert lower._is_dynamic_shared_memory(loaded_shared)
            assert not lower._is_dynamic_shared_memory(loaded_scalar)

            lower._mark_dynamic_shared_memory(scalar)
            assert lower._dynamic_shared_memory_memref_type(shared) is not None
            assert lower._dynamic_shared_memory_memref_type(scalar) is None
            lower._mark_dynamic_shared_memory_slot((mismatched_slot,), (shared, scalar))
            lower._mark_dynamic_shared_memory_alias(
                (mismatched_loaded,), (slot_shared, slot_scalar)
            )
            assert not lower._is_dynamic_shared_memory(mismatched_slot)
            assert not lower._is_dynamic_shared_memory(mismatched_loaded)


@pytest.mark.parametrize(
    ("kernel", "sig"),
    [
        (dyn_shared_memory_row_view, "void(float32[:, :])"),
        (dyn_shared_memory_ravel_view, "void(float32[:])"),
        (dyn_shared_memory_reshape_tuple_view, "void(float32[:])"),
        (dyn_shared_memory_dtype_view, "void(int32[:])"),
        (dyn_shared_memory_slice_view, "void(float32[:])"),
        (dyn_shared_memory_slice_assign, "void(float32[:])"),
        (dyn_shared_memory_complex_real_imag_view, "void(float32[:])"),
        (dyn_shared_memory_complex_element, "void(complex64[:], float32[:])"),
    ],
)
def test_dynamic_shared_memory_aliases_use_shared_load_store_before_cleanup(
    monkeypatch, kernel, sig
):
    from numba_cuda_mlir import compiler, tools

    monkeypatch.setattr(tools, "get_gpu_compute_capability", lambda tuple=False: (10, 0))

    mlir = compiler.compile_mlir(
        cuda.jit(chip="sm_100")(kernel),
        sig,
        optimized=False,
        chip="sm_100",
    )

    shared_loads = [
        line for line in mlir.splitlines() if "llvm.load" in line and "!llvm.ptr<3>" in line
    ]
    shared_stores = [
        line for line in mlir.splitlines() if "llvm.store" in line and "!llvm.ptr<3>" in line
    ]
    assert shared_loads
    assert shared_stores


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
