# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import math
import numpy as np
from numba import cuda as numba_cuda, float32, int32, uint32, types
import cuda.simt as cusimt_cuda

DEFAULT_SIZE = 8192
SEED = 42
M_PI = np.float32(3.14159265358979323846)
BLOCK_SIZE = 256


@numba_cuda.jit(device=True, inline="always")
def bit_reverse_numba_cuda(x: uint32, bits: int32) -> uint32:
    result = uint32(0)
    for i in range(bits):
        if x & (uint32(1) << i):
            result |= uint32(1) << (bits - 1 - i)
    return result


@numba_cuda.jit
def bitreverse_permute_numba_cuda(in_arr, out_arr, n, logn):
    i = numba_cuda.blockIdx.x * numba_cuda.blockDim.x + numba_cuda.threadIdx.x
    if i >= n:
        return
    rev = bit_reverse_numba_cuda(uint32(i), logn)
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
    k = float32(j * stride)
    ang = float32(-2.0) * M_PI * k / float32(n)
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


@cusimt_cuda.jit(device=True, inline="always")
def bit_reverse_cusimt(x: uint32, bits: int32) -> uint32:
    result = uint32(0)
    for i in range(bits):
        if x & (uint32(1) << i):
            result |= uint32(1) << (bits - 1 - i)
    return result


@cusimt_cuda.jit
def bitreverse_permute_cusimt(in_arr, out_arr, n, logn):
    i = cusimt_cuda.blockIdx.x * cusimt_cuda.blockDim.x + cusimt_cuda.threadIdx.x
    if i >= n:
        return
    rev = bit_reverse_cusimt(uint32(i), logn)
    out_arr[rev] = in_arr[i]


@cusimt_cuda.jit
def fft_stage_inplace_cusimt(data, n, stage, inverse):
    tid = cusimt_cuda.blockIdx.x * cusimt_cuda.blockDim.x + cusimt_cuda.threadIdx.x
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
    k = float32(j * stride)
    ang = float32(-2.0) * M_PI * k / float32(n)
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


def run_cusimt_fft(input_array):
    n = len(input_array)
    logn = count_bits(n)
    d_input = cusimt_cuda.to_device(input_array)
    d_output = cusimt_cuda.device_array(n, dtype=np.complex64)

    blocks = ceil_div(n, BLOCK_SIZE)
    bitreverse_permute_cusimt[blocks, BLOCK_SIZE](d_input, d_output, n, logn)

    for s in range(logn):
        m = 1 << (s + 1)
        half = m >> 1
        butterflies = (n // m) * half
        blocks = ceil_div(butterflies, BLOCK_SIZE)
        fft_stage_inplace_cusimt[blocks, BLOCK_SIZE](d_output, n, s, 0)

    cusimt_cuda.synchronize()
    return d_output.copy_to_host()


def test_fft():
    input_array = get_input_data()
    reference = np.fft.fft(input_array)
    cusimt_output = run_cusimt_fft(input_array)

    n = len(input_array)
    tolerance = 1e-2 * math.sqrt(math.log2(n))

    cusimt_err = np.max(np.abs(cusimt_output - reference))
    assert cusimt_err < tolerance, f"cuSIMT error {cusimt_err} exceeds tolerance {tolerance}"


def test_fft_benchmark(benchmark_runner):
    benchmark_runner(script=__file__)


def run_benchmark_main():
    permute_sig = types.void(types.complex64[::1], types.complex64[::1], types.int64, types.int64)
    stage_sig = types.void(types.complex64[::1], types.int64, types.int64, types.int64)

    start = time.perf_counter()
    bitreverse_permute_numba_cuda.compile(permute_sig)
    fft_stage_inplace_numba_cuda.compile(stage_sig)
    numba_compile_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    bitreverse_permute_cusimt.compile(permute_sig)
    fft_stage_inplace_cusimt.compile(stage_sig)
    cusimt_compile_time = (time.perf_counter() - start) * 1000

    print("\n=== COMPILE TIMES ===")
    print(f"Numba-CUDA: {numba_compile_time:.3f} ms")
    print(f"cuSIMT: {cusimt_compile_time:.3f} ms")

    n = DEFAULT_SIZE
    logn = count_bits(n)
    input_array = get_input_data()

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

    d_input = cusimt_cuda.to_device(input_array)
    d_output = cusimt_cuda.device_array(n, dtype=np.complex64)
    blocks = ceil_div(n, BLOCK_SIZE)
    bitreverse_permute_cusimt[blocks, BLOCK_SIZE](d_input, d_output, n, logn)
    for s in range(logn):
        m = 1 << (s + 1)
        half = m >> 1
        butterflies = (n // m) * half
        blocks = ceil_div(butterflies, BLOCK_SIZE)
        fft_stage_inplace_cusimt[blocks, BLOCK_SIZE](d_output, n, s, 0)
    cusimt_cuda.synchronize()


if __name__ == "__main__":
    run_benchmark_main()
