# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from pathlib import Path

import math
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_utils import (
    BACKEND_BOTH,
    BACKEND_NUMBA_CUDA,
    BACKEND_NUMBA_CUDA_MLIR,
    add_backend_arg,
    add_compile_mode_arg,
    prepare_compile_measurement,
    print_compile_times,
    selected_backend_from_argv,
    should_run_backend,
    skipped_backend,
    time_compile_sequence,
)

SELECTED_BACKEND = selected_backend_from_argv()
if should_run_backend(SELECTED_BACKEND, BACKEND_NUMBA_CUDA):
    from numba import cuda as numba_cuda, types as nty
else:
    numba_cuda = skipped_backend()
    nty = skipped_backend()
if should_run_backend(SELECTED_BACKEND, BACKEND_NUMBA_CUDA_MLIR):
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir.numba_cuda import types as cty
else:
    cuda = skipped_backend()
    cty = skipped_backend()


DEFAULT_SIZE = 8192
SEED = 42
M_PI = np.float32(3.14159265358979323846)
BLOCK_SIZE = 256


@numba_cuda.jit(device=True, inline="always")
def bit_reverse_numba_cuda(x: nty.uint32, bits: nty.int32) -> nty.uint32:
    result = nty.uint32(0)
    for i in range(bits):
        if x & (nty.uint32(1) << i):
            result |= nty.uint32(1) << (bits - 1 - i)
    return result


@numba_cuda.jit
def bitreverse_permute_numba_cuda(in_arr, out_arr, n, logn):
    i = numba_cuda.blockIdx.x * numba_cuda.blockDim.x + numba_cuda.threadIdx.x
    if i >= n:
        return
    rev = bit_reverse_numba_cuda(nty.uint32(i), logn)
    out_arr[rev] = in_arr[i]


@numba_cuda.jit
def fft_stage_inplace_numba_cuda(data, n, stage, inverse):
    tid = numba_cuda.blockIdx.x * numba_cuda.blockDim.x + numba_cuda.threadIdx.x
    m = 1 << (stage + 1)
    half = m >> 1
    butterflies = (n // m) * half

    if tid >= butterflies:
        return

    group = tid // half
    j = tid - group * half
    base = group * m
    i0 = base + j
    i1 = i0 + half

    stride = n // m
    k = nty.float32(j * stride)
    ang = nty.float32(-2.0) * M_PI * k / nty.float32(n)
    if inverse:
        ang = -ang

    c = math.cos(ang)
    s = math.sin(ang)

    a_real, a_imag = data[i0].real, data[i0].imag
    b_real, b_imag = data[i1].real, data[i1].imag

    t_real = c * b_real - s * b_imag
    t_imag = c * b_imag + s * b_real

    data[i0] = complex(a_real + t_real, a_imag + t_imag)
    data[i1] = complex(a_real - t_real, a_imag - t_imag)


@cuda.jit(device=True, inline="always")
def bit_reverse_numba_cuda_mlir(x: cty.uint32, bits: cty.int32) -> cty.uint32:
    result = cty.uint32(0)
    for i in range(bits):
        if x & (cty.uint32(1) << i):
            result |= cty.uint32(1) << (bits - 1 - i)
    return result


@cuda.jit
def bitreverse_permute_numba_cuda_mlir(in_arr, out_arr, n, logn):
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i >= n:
        return
    rev = bit_reverse_numba_cuda_mlir(cty.uint32(i), logn)
    out_arr[rev] = in_arr[i]


@cuda.jit
def fft_stage_inplace_numba_cuda_mlir(data, n, stage, inverse):
    tid = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    m = 1 << (stage + 1)
    half = m >> 1
    butterflies = (n // m) * half

    if tid >= butterflies:
        return

    group = tid // half
    j = tid - group * half
    base = group * m
    i0 = base + j
    i1 = i0 + half

    stride = n // m
    k = cty.float32(j * stride)
    ang = cty.float32(-2.0) * M_PI * k / cty.float32(n)
    if inverse:
        ang = -ang

    c = math.cos(ang)
    s = math.sin(ang)

    a_real, a_imag = data[i0].real, data[i0].imag
    b_real, b_imag = data[i1].real, data[i1].imag

    t_real = c * b_real - s * b_imag
    t_imag = c * b_imag + s * b_real

    data[i0] = complex(a_real + t_real, a_imag + t_imag)
    data[i1] = complex(a_real - t_real, a_imag - t_imag)


def ceil_div(a, b):
    return (a + b - 1) // b


def count_bits(n):
    bits = 0
    t = n
    while t > 1:
        t >>= 1
        bits += 1
    return bits


def get_input_data():
    np.random.seed(SEED)
    tone = 5.0
    t = np.arange(DEFAULT_SIZE, dtype=np.float32)
    phase = 2.0 * np.pi * tone * t / DEFAULT_SIZE
    return (1.0 + np.cos(phase) + 1j * np.sin(phase)).astype(np.complex64)


def run_numba_cuda_mlir_fft(input_array):
    n = len(input_array)
    logn = count_bits(n)
    d_input = cuda.to_device(input_array)
    d_output = cuda.device_array(n, dtype=np.complex64)

    blocks = ceil_div(n, BLOCK_SIZE)
    bitreverse_permute_numba_cuda_mlir[blocks, BLOCK_SIZE](d_input, d_output, n, logn)

    for s in range(logn):
        m = 1 << (s + 1)
        half = m >> 1
        butterflies = (n // m) * half
        blocks = ceil_div(butterflies, BLOCK_SIZE)
        fft_stage_inplace_numba_cuda_mlir[blocks, BLOCK_SIZE](d_output, n, s, 0)

    cuda.synchronize()
    return d_output.copy_to_host()


def test_fft():
    input_array = get_input_data()
    reference = np.fft.fft(input_array)
    numba_cuda_mlir_output = run_numba_cuda_mlir_fft(input_array)

    n = len(input_array)
    tolerance = 1e-2 * math.sqrt(math.log2(n))

    numba_cuda_mlir_err = np.max(np.abs(numba_cuda_mlir_output - reference))
    assert numba_cuda_mlir_err < tolerance, (
        f"numba-cuda-mlir error {numba_cuda_mlir_err} exceeds tolerance {tolerance}"
    )


def test_fft_benchmark(benchmark_runner):
    benchmark_runner(script=__file__)


def run_benchmark_main(compile_mode="cold", backend=BACKEND_BOTH):
    permute_sig = "void(complex64[::1], complex64[::1], int64, int64)"
    stage_sig = "void(complex64[::1], int64, int64, int64)"
    prepare_compile_measurement(compile_mode, backend)

    compile_times = {}
    if should_run_backend(backend, BACKEND_NUMBA_CUDA):
        compile_times[BACKEND_NUMBA_CUDA] = time_compile_sequence(
            (bitreverse_permute_numba_cuda, permute_sig),
            (fft_stage_inplace_numba_cuda, stage_sig),
        )
    if should_run_backend(backend, BACKEND_NUMBA_CUDA_MLIR):
        compile_times[BACKEND_NUMBA_CUDA_MLIR] = time_compile_sequence(
            (bitreverse_permute_numba_cuda_mlir, permute_sig),
            (fft_stage_inplace_numba_cuda_mlir, stage_sig),
        )

    print_compile_times(compile_times)

    n = DEFAULT_SIZE
    logn = count_bits(n)
    input_array = get_input_data()

    if should_run_backend(backend, BACKEND_NUMBA_CUDA):
        d_input = numba_cuda.to_device(input_array)
        d_output = numba_cuda.device_array(n, dtype=np.complex64)
        blocks = ceil_div(n, BLOCK_SIZE)
        bitreverse_permute_numba_cuda[blocks, BLOCK_SIZE](d_input, d_output, n, logn)
        for s in range(logn):
            m = 1 << (s + 1)
            half = m >> 1
            butterflies = (n // m) * half
            blocks = ceil_div(butterflies, BLOCK_SIZE)
            fft_stage_inplace_numba_cuda[blocks, BLOCK_SIZE](d_output, n, s, 0)
        numba_cuda.synchronize()

    if should_run_backend(backend, BACKEND_NUMBA_CUDA_MLIR):
        d_input = cuda.to_device(input_array)
        d_output = cuda.device_array(n, dtype=np.complex64)
        blocks = ceil_div(n, BLOCK_SIZE)
        bitreverse_permute_numba_cuda_mlir[blocks, BLOCK_SIZE](d_input, d_output, n, logn)
        for s in range(logn):
            m = 1 << (s + 1)
            half = m >> 1
            butterflies = (n // m) * half
            blocks = ceil_div(butterflies, BLOCK_SIZE)
            fft_stage_inplace_numba_cuda_mlir[blocks, BLOCK_SIZE](d_output, n, s, 0)
        cuda.synchronize()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FFT benchmark")
    add_compile_mode_arg(parser)
    add_backend_arg(parser)
    args = parser.parse_args()
    run_benchmark_main(args.compile_mode, args.backend)
