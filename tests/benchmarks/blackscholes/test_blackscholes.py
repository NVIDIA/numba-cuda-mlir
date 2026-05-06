# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from pathlib import Path

import numpy as np
import math

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
    time_compile,
)

SELECTED_BACKEND = selected_backend_from_argv()
if should_run_backend(SELECTED_BACKEND, BACKEND_NUMBA_CUDA):
    import numba.cuda as numba_cuda
else:
    numba_cuda = skipped_backend()
if should_run_backend(SELECTED_BACKEND, BACKEND_NUMBA_CUDA_MLIR):
    from numba_cuda_mlir import cuda
else:
    cuda = skipped_backend()

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


@cuda.jit(device=True, inline="always")
def cnd_numba_cuda_mlir(d):
    K = 1.0 / (1.0 + 0.2316419 * math.fabs(d))
    ret_val = (
        RSQRT2PI * math.exp(-0.5 * d * d) * (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))))
    )
    if d > 0:
        ret_val = 1.0 - ret_val
    return ret_val


@cuda.jit
def black_scholes_numba_cuda_mlir(callResult, putResult, S, X, T, R, V):
    i = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    if i >= S.shape[0]:
        return
    sqrtT = math.sqrt(T[i])
    d1 = (math.log(S[i] / X[i]) + (R + 0.5 * V * V) * T[i]) / (V * sqrtT)
    d2 = d1 - V * sqrtT
    cndd1, cndd2 = cnd_numba_cuda_mlir(d1), cnd_numba_cuda_mlir(d2)
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


def run_numba_cuda_mlir_version(stockPrice, optionStrike, optionYears):
    n = len(stockPrice)
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    d_S = cuda.to_device(stockPrice)
    d_X = cuda.to_device(optionStrike)
    d_T = cuda.to_device(optionYears)
    d_call = cuda.device_array(n, dtype=np.float64)
    d_put = cuda.device_array(n, dtype=np.float64)
    black_scholes_numba_cuda_mlir[blocks, THREADS_PER_BLOCK](
        d_call, d_put, d_S, d_X, d_T, RISKFREE, VOLATILITY
    )
    cuda.synchronize()
    return d_call.copy_to_host(), d_put.copy_to_host()


def test_blackscholes():
    from conftest import verify_against_reference

    n = DEFAULT_N
    stockPrice, optionStrike, optionYears = get_input_data(n)
    call_ref, put_ref = black_scholes_numpy(
        stockPrice, optionStrike, optionYears, RISKFREE, VOLATILITY
    )
    call_numba_cuda_mlir, put_numba_cuda_mlir = run_numba_cuda_mlir_version(
        stockPrice, optionStrike, optionYears
    )

    verify_against_reference(
        (call_ref, put_ref),
        (call_numba_cuda_mlir, put_numba_cuda_mlir),
        tolerance=1e-10,
        name="numba-cuda-mlir",
    )


def test_blackscholes_benchmark(benchmark_runner):
    benchmark_runner(script=__file__)


def run_benchmark_main(compile_mode="cold", backend=BACKEND_BOTH):
    sig = "void(float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64, float64)"
    prepare_compile_measurement(compile_mode, backend)

    compile_times = {}
    if should_run_backend(backend, BACKEND_NUMBA_CUDA):
        compile_times[BACKEND_NUMBA_CUDA] = time_compile(black_scholes_numba_cuda.compile, sig)
    if should_run_backend(backend, BACKEND_NUMBA_CUDA_MLIR):
        compile_times[BACKEND_NUMBA_CUDA_MLIR] = time_compile(
            black_scholes_numba_cuda_mlir.compile, sig
        )

    print_compile_times(compile_times)

    n = DEFAULT_N
    stockPrice, optionStrike, optionYears = get_input_data(n)
    blocks = (n + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    if should_run_backend(backend, BACKEND_NUMBA_CUDA):
        d_S = numba_cuda.to_device(stockPrice)
        d_X = numba_cuda.to_device(optionStrike)
        d_T = numba_cuda.to_device(optionYears)
        d_call = numba_cuda.device_array(n, dtype=np.float64)
        d_put = numba_cuda.device_array(n, dtype=np.float64)
        black_scholes_numba_cuda[blocks, THREADS_PER_BLOCK](
            d_call, d_put, d_S, d_X, d_T, RISKFREE, VOLATILITY
        )
        numba_cuda.synchronize()

    if should_run_backend(backend, BACKEND_NUMBA_CUDA_MLIR):
        d_S = cuda.to_device(stockPrice)
        d_X = cuda.to_device(optionStrike)
        d_T = cuda.to_device(optionYears)
        d_call = cuda.device_array(n, dtype=np.float64)
        d_put = cuda.device_array(n, dtype=np.float64)
        black_scholes_numba_cuda_mlir[blocks, THREADS_PER_BLOCK](
            d_call, d_put, d_S, d_X, d_T, RISKFREE, VOLATILITY
        )
        cuda.synchronize()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Black-Scholes benchmark")
    add_compile_mode_arg(parser)
    add_backend_arg(parser)
    args = parser.parse_args()
    run_benchmark_main(args.compile_mode, args.backend)
