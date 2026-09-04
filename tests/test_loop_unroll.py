# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re

import numpy as np
import pytest

from numba_cuda_mlir import cuda, testing
from numba_cuda_mlir.numba_cuda.core.errors import TypingError

N = 16
LATCH = "llvm.br ^bb{{[0-9]+}} {loop_annotation = #loop_annotation}"


def _optimized_mlir(kernel):
    (mlir_text,) = kernel.inspect_mlir_optimized().values()
    return mlir_text


def _llvm_ir(kernel):
    (llvm_ir,) = kernel.inspect_llvm().values()
    return llvm_ir


def _run_sum(kernel):
    x = cuda.to_device(np.arange(N, dtype=np.float32))
    out = cuda.to_device(np.zeros(1, dtype=np.float32))
    kernel[1, 1](x, out)
    return out.copy_to_host()[0]


def _expect_typing_error(kernel, match=None):
    x = cuda.to_device(np.arange(N, dtype=np.float32))
    out = cuda.to_device(np.zeros(1, dtype=np.float32))
    with pytest.raises(TypingError, match=match):
        kernel[1, 1](x, out)


def test_unroll_full_annotates_latch():
    @cuda.jit
    def kernel(x, out):
        acc = np.float32(0.0)
        for k in cuda.unroll(range(N)):
            acc += x[k]
        out[0] = acc

    assert _run_sum(kernel) == np.arange(N, dtype=np.float32).sum()
    testing.filecheck(
        f"""
        CHECK: #llvm.loop_unroll<full = true>
        CHECK: {LATCH}
        """,
        _optimized_mlir(kernel),
    )


@pytest.mark.parametrize("keyword", [False, True], ids=["positional", "keyword"])
def test_unroll_count_annotates_latch(keyword):
    if keyword:

        @cuda.jit
        def kernel(x, out):
            acc = np.float32(0.0)
            for k in cuda.unroll(range(N), count=4):
                acc += x[k]
            out[0] = acc

    else:

        @cuda.jit
        def kernel(x, out):
            acc = np.float32(0.0)
            for k in cuda.unroll(range(N), 4):
                acc += x[k]
            out[0] = acc

    assert _run_sum(kernel) == np.arange(N, dtype=np.float32).sum()
    testing.filecheck(
        f"""
        CHECK: #llvm.loop_unroll<count = 4 : i64>
        CHECK: {LATCH}
        """,
        _optimized_mlir(kernel),
    )


def test_nounroll_annotates_latch():
    @cuda.jit
    def kernel(x, out):
        acc = np.float32(0.0)
        for k in cuda.nounroll(range(N)):
            acc += x[k]
        out[0] = acc

    assert _run_sum(kernel) == np.arange(N, dtype=np.float32).sum()
    testing.filecheck(
        f"""
        CHECK: #llvm.loop_unroll<disable = true>
        CHECK: {LATCH}
        """,
        _optimized_mlir(kernel),
    )


def test_debug_build_annotates_latch():
    @cuda.jit(debug=True, opt=False)
    def kernel(x, out):
        acc = np.float32(0.0)
        for k in cuda.unroll(range(N), 4):
            acc += x[k]
        out[0] = acc

    assert _run_sum(kernel) == np.arange(N, dtype=np.float32).sum()
    mlir_text = _optimized_mlir(kernel)
    assert "loop_unroll:" not in mlir_text
    testing.filecheck(
        f"""
        CHECK: #llvm.loop_unroll<count = 4 : i64>
        CHECK: {LATCH}
        """,
        mlir_text,
    )
    assert '!{!"llvm.loop.unroll.count", i32 4}' in _llvm_ir(kernel)


def test_nounroll_keeps_loop_in_ptx():
    @cuda.jit
    def unrolled(x, out):
        acc = np.float32(0.0)
        for k in range(N):
            acc += x[k]
        out[0] = acc

    @cuda.jit
    def rolled(x, out):
        acc = np.float32(0.0)
        for k in cuda.nounroll(range(N)):
            acc += x[k]
        out[0] = acc

    assert _run_sum(unrolled) == _run_sum(rolled)
    (ptx_unrolled,) = unrolled.inspect_asm().values()
    (ptx_rolled,) = rolled.inspect_asm().values()
    assert ptx_rolled.count("ld.global") == 1
    assert ptx_unrolled.count("ld.global") > 1


def test_unroll_count_shapes_ptx():
    @cuda.jit
    def kernel(x, out):
        acc = np.float32(0.0)
        for k in cuda.unroll(range(N), 4):
            acc += x[k]
        out[0] = acc

    _run_sum(kernel)
    (ptx,) = kernel.inspect_asm().values()
    assert ptx.count("ld.global") == 4


def test_nounroll_survives_lto():
    @cuda.jit(lto=True)
    def kernel(x, out):
        acc = np.float32(0.0)
        for k in cuda.nounroll(range(N)):
            acc += x[k]
        out[0] = acc

    assert _run_sum(kernel) == np.arange(N, dtype=np.float32).sum()
    (ptx,) = kernel.inspect_lto_ptx().values()
    assert ptx.count("ld.global") == 1


def test_break_and_continue_loop_is_annotated():
    @cuda.jit
    def kernel(x, out):
        acc = np.float32(0.0)
        for k in cuda.unroll(range(N), 2):
            if k % 3 == 0:
                continue
            if k > 10:
                break
            acc += x[k]
        out[0] = acc

    assert _run_sum(kernel) == sum(k for k in range(11) if k % 3 != 0)
    testing.filecheck(
        f"""
        CHECK: #llvm.loop_unroll<count = 2 : i64>
        CHECK: {LATCH}
        """,
        _optimized_mlir(kernel),
    )


def test_break_as_last_statement_keeps_hint():
    @cuda.jit
    def kernel(x, out):
        for k in cuda.unroll(range(N), 4):
            v = x[k]
            out[k] = v
            if v > 100.0:
                break

    x = cuda.to_device(np.arange(N, dtype=np.float32))
    out = cuda.to_device(np.zeros(N, dtype=np.float32))
    kernel[1, 1](x, out)
    np.testing.assert_array_equal(out.copy_to_host(), x.copy_to_host())
    assert "#llvm.loop_unroll<count = 4 : i64>" in _optimized_mlir(kernel)
    (ptx,) = kernel.inspect_asm().values()
    assert ptx.count("ld.global") == 4


def test_return_as_last_statement_keeps_hint():
    @cuda.jit
    def kernel(x, out):
        for k in cuda.unroll(range(N)):
            v = x[k]
            out[k] = v
            if v > 100.0:
                return

    x = cuda.to_device(np.arange(N, dtype=np.float32))
    out = cuda.to_device(np.zeros(N, dtype=np.float32))
    kernel[1, 1](x, out)
    np.testing.assert_array_equal(out.copy_to_host(), x.copy_to_host())
    assert "#llvm.loop_unroll<full = true>" in _optimized_mlir(kernel)
    (ptx,) = kernel.inspect_asm().values()
    assert ptx.count("ld.global") == N


def test_nested_loops_carry_their_own_hints():
    @cuda.jit
    def kernel(x, out):
        acc = np.float32(0.0)
        for i in cuda.nounroll(range(4)):
            for k in cuda.unroll(range(4)):
                acc += x[i * 4 + k]
        out[0] = acc

    assert _run_sum(kernel) == np.arange(N, dtype=np.float32).sum()
    testing.filecheck(
        """
        CHECK-DAG: #llvm.loop_unroll<disable = true>
        CHECK-DAG: #llvm.loop_unroll<full = true>
        CHECK: llvm.br ^bb{{[0-9]+}} {loop_annotation = #loop_annotation}
        CHECK: llvm.br ^bb{{[0-9]+}} {loop_annotation = #loop_annotation1}
        """,
        _optimized_mlir(kernel),
    )


def test_latches_of_one_loop_share_its_llvm_loop_id():
    @cuda.jit
    def kernel(x, out):
        acc = np.float32(0.0)
        for k in cuda.unroll(range(N), 2):
            if k % 3 == 0:
                continue
            acc += x[k]
        out[0] = acc

    assert _run_sum(kernel) == sum(k for k in range(N) if k % 3 != 0)
    llvm_ir = _llvm_ir(kernel)
    ids = set(re.findall(r"!llvm\.loop !(\d+)", llvm_ir))
    assert len(ids) == 1
    (loop_id,) = ids
    assert f"!{loop_id} = distinct !{{!{loop_id}, " in llvm_ir


def test_array_loop_is_annotated():
    @cuda.jit
    def kernel(x, out):
        acc = np.float32(0.0)
        for v in cuda.unroll(x, 4):
            acc += v
        out[0] = acc

    assert _run_sum(kernel) == np.arange(N, dtype=np.float32).sum()
    mlir_text = _optimized_mlir(kernel)
    assert "#llvm.loop_unroll<count = 4 : i64>" in mlir_text
    assert "{loop_annotation = #loop_annotation}" in mlir_text


def test_tuple_loop_is_annotated():
    @cuda.jit
    def kernel(x, out):
        acc = np.float32(0.0)
        for k in cuda.nounroll((0, 1, 2, 3)):
            acc += x[k]
        out[0] = acc

    assert _run_sum(kernel) == 6.0
    mlir_text = _optimized_mlir(kernel)
    assert "#llvm.loop_unroll<disable = true>" in mlir_text
    assert "{loop_annotation = #loop_annotation}" in mlir_text


def test_nditer_loop_is_annotated():
    @cuda.jit
    def kernel(x, out):
        acc = np.float32(0.0)
        for v in cuda.unroll(np.nditer(x)):
            acc += v[()]
        out[0] = acc

    assert _run_sum(kernel) == np.arange(N, dtype=np.float32).sum()
    mlir_text = _optimized_mlir(kernel)
    assert "#llvm.loop_unroll<full = true>" in mlir_text
    assert "{loop_annotation = #loop_annotation}" in mlir_text


def test_hinted_iterable_held_in_a_variable():
    @cuda.jit
    def kernel(x, out):
        acc = np.float32(0.0)
        it = cuda.nounroll(x)
        for v in it:
            acc += v
        out[0] = acc

    assert _run_sum(kernel) == np.arange(N, dtype=np.float32).sum()
    assert "#llvm.loop_unroll<disable = true>" in _optimized_mlir(kernel)


def test_runtime_count_is_rejected():
    @cuda.jit
    def kernel(x, out, n):
        acc = np.float32(0.0)
        for k in cuda.unroll(range(N), n):
            acc += x[k]
        out[0] = acc

    x = cuda.to_device(np.arange(N, dtype=np.float32))
    out = cuda.to_device(np.zeros(1, dtype=np.float32))
    with pytest.raises(TypingError):
        kernel[1, 1](x, out, 4)


@pytest.mark.parametrize("count", [0, 2**31], ids=["zero", "above_i32"])
def test_out_of_range_count_is_rejected(count):
    @cuda.jit
    def kernel(x, out):
        acc = np.float32(0.0)
        for k in cuda.unroll(range(N), count):
            acc += x[k]
        out[0] = acc

    _expect_typing_error(kernel, match="unroll count")


def test_non_iterable_is_rejected():
    @cuda.jit
    def kernel(x, out):
        acc = np.float32(0.0)
        for v in cuda.unroll(x[0]):
            acc += v
        out[0] = acc

    _expect_typing_error(kernel)


def test_markers_return_the_iterable_outside_compiled_code():
    r = range(N)
    assert cuda.unroll(r) is r
    assert cuda.unroll(r, 4) is r
    assert cuda.unroll(r, count=4) is r
    assert cuda.unroll(r, None) is r
    assert cuda.nounroll(r) is r
