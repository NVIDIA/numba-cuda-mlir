# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
import numpy as np
import math
from numba import types

import numba.cuda as numba_cuda
import cuda.simt as cusimt_cuda

RISKFREE = 0.02
VOLATILITY = 0.30

A1 = 0.31938153
A2 = -0.356563782
A3 = 1.781477937
A4 = -1.821255978
A5 = 1.330274429
RSQRT2PI = 0.39894228040143267793994605993438

DEFAULT_N = 1024 * 1024
THREADS_PER_BLOCK = 256


def cnd_numpy(d):
    K = 1.0 / (1.0 + 0.2316419 * np.abs(d))
    ret_val = RSQRT2PI * np.exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    return np.where(d > 0, 1.0 - ret_val, ret_val)


def black_scholes_numpy(stockPrice, optionStrike, optionYears, Riskfree, Volatility):
    S, X, T, R, V = stockPrice, optionStrike, optionYears, Riskfree, Volatility
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / X) + (R + 0.5 * V * V) * T) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1, cndd2 = cnd_numpy(d1), cnd_numpy(d2)
    expRT = np.exp(-R * T)
    callResult = S * cndd1 - X * expRT * cndd2
    putResult = X * expRT * (1.0 - cndd2) - S * (1.0 - cndd1)
    return callResult, putResult


@numba_cuda.jit(device=True)
def cnd_numba_cuda(d):
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (
        RSQRT2PI * math.exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    )
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val


@numba_cuda.jit
def black_scholes_numba_cuda(callResult, putResult, S, X, T, R, V):
    i = numba_cuda.threadIdx.x + numba_cuda.blockIdx.x * numba_cuda.blockDim.x
    if i >= S.shape[0]:
        return
    sqrtT = math.sqrt(T[i])
    d1 = (math.log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1, cndd2 = cnd_numba_cuda(d1), cnd_numba_cuda(d2)
    expRT = math.exp(-R * T[i])
    callResult[i] = S[i] * cndd1 - X[i] * expRT * cndd2
    putResult[i] = X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1)


@cusimt_cuda.jit(device=True, inline="always")
def cnd_cusimt(d):
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (
        RSQRT2PI * math.exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    )
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val


@cusimt_cuda.jit
def black_scholes_cusimt(callResult, putResult, S, X, T, R, V):
    i = cusimt_cuda.threadIdx.x + cusimt_cuda.blockIdx.x * cusimt_cuda.blockDim.x
    if i >= S.shape[0]:
        return
    sqrtT = math.sqrt(T[i])
    d1 = (math.log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1, cndd2 = cnd_cusimt(d1), cnd_cusimt(d2)
    expRT = math.exp(-R * T[i])
    callResult[i] = S[i] * cndd1 - X[i] * expRT * cndd2
    putResult[i] = X[i] * expRT * (1.0 - cndd2) - S[i] * (1.0 - cndd1)


def randfloat(rand_var, low, high):
    return (1.0 - rand_var) * low + rand_var * high


def get_input_data(n):
    np.random.seed(42)
    stockPrice = randfloat(np.random.random(n), 5.0, 30.0)
    optionStrike = randfloat(np.random.random(n), 1.0, 100.0)
    optionYears = randfloat(np.random.random(n), 0.25, 10.0)
    return stockPrice, optionStrike, optionYears


def run_cusimt_version(stockPrice, optionStrike, optionYears):
    n = len(stockPrice)
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    d_S = cusimt_cuda.to_device(stockPrice)
    d_X = cusimt_cuda.to_device(optionStrike)
    d_T = cusimt_cuda.to_device(optionYears)
    d_call = cusimt_cuda.device_array(n, dtype=np.float64)
    d_put = cusimt_cuda.device_array(n, dtype=np.float64)
    black_scholes_cusimt[blocks, THREADS_PER_BLOCK](
        d_call, d_put, d_S, d_X, d_T, RISKFREE, VOLATILITY
    )
    cusimt_cuda.synchronize()
    return d_call.copy_to_host(), d_put.copy_to_host()


def test_blackscholes():
    from conftest import verify_against_reference

    n = DEFAULT_N
    stockPrice, optionStrike, optionYears = get_input_data(n)
    call_ref, put_ref = black_scholes_numpy(
        stockPrice, optionStrike, optionYears, RISKFREE, VOLATILITY
    )
    call_cusimt, put_cusimt = run_cusimt_version(stockPrice, optionStrike, optionYears)

    verify_against_reference(
        (call_ref, put_ref), (call_cusimt, put_cusimt), tolerance=1e-10, name="cuSIMT"
    )


def test_blackscholes_benchmark(benchmark_runner):
    benchmark_runner(script=__file__)


def run_benchmark_main():
    sig = types.void(
        types.float64[::1],
        types.float64[::1],
        types.float64[::1],
        types.float64[::1],
        types.float64[::1],
        types.float64,
        types.float64,
    )

    start = time.perf_counter()
    black_scholes_numba_cuda.compile(sig)
    numba_compile_time = (time.perf_counter() - start) * 1000

    start = time.perf_counter()
    black_scholes_cusimt.compile(sig)
    cusimt_compile_time = (time.perf_counter() - start) * 1000

    print("\n=== COMPILE TIMES ===")
    print(f"Numba-CUDA: {numba_compile_time:.3f} ms")
    print(f"cuSIMT: {cusimt_compile_time:.3f} ms")

    n = DEFAULT_N
    stockPrice, optionStrike, optionYears = get_input_data(n)
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    d_S = numba_cuda.to_device(stockPrice)
    d_X = numba_cuda.to_device(optionStrike)
    d_T = numba_cuda.to_device(optionYears)
    d_call = numba_cuda.device_array(n, dtype=np.float64)
    d_put = numba_cuda.device_array(n, dtype=np.float64)
    black_scholes_numba_cuda[blocks, THREADS_PER_BLOCK](
        d_call, d_put, d_S, d_X, d_T, RISKFREE, VOLATILITY
    )
    numba_cuda.synchronize()

    d_S = cusimt_cuda.to_device(stockPrice)
    d_X = cusimt_cuda.to_device(optionStrike)
    d_T = cusimt_cuda.to_device(optionYears)
    d_call = cusimt_cuda.device_array(n, dtype=np.float64)
    d_put = cusimt_cuda.device_array(n, dtype=np.float64)
    black_scholes_cusimt[blocks, THREADS_PER_BLOCK](
        d_call, d_put, d_S, d_X, d_T, RISKFREE, VOLATILITY
    )
    cusimt_cuda.synchronize()


if __name__ == "__main__":
    run_benchmark_main()
