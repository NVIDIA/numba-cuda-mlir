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

S = 1024
D = 64


def attention_numpy(X, seq_len, dim):
    X_2d = X.reshape(seq_len, dim)
    inv_sqrt_d = 1.0 / np.sqrt(dim)
    scores = (X_2d @ X_2d.T) * inv_sqrt_d
    max_scores = np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    output = softmax @ X_2d
    return output.flatten()


@numba_cuda.jit
def numba_cuda_simple_attention(X, Y, S, D):
    i = numba_cuda.blockIdx.x
    scores = numba_cuda.shared.array(0, dtype=numba_cuda.float32)
    inv_sqrt_d = 0.015625

    max_score = -1e38
    for j in range(S):
        dot = 0.0
        for d in range(D):
            dot += X[i * D + d] * X[j * D + d]
        sc = dot * inv_sqrt_d
        scores[j] = sc
        if sc > max_score:
            max_score = sc

    sum_exp = 0.0
    for j in range(S):
        ex = math.exp(scores[j] - max_score)
        scores[j] = ex
        sum_exp += ex

    for j in range(S):
        scores[j] = scores[j] / sum_exp

    for d in range(D):
        acc = 0.0
        for j in range(S):
            acc += scores[j] * X[j * D + d]
        Y[i * D + d] = acc


@cuda.jit
def numba_cuda_mlir_simple_attention(X, Y, S, D):
    i = cuda.blockIdx.x
    scores = cuda.shared.array(0, dtype=cuda.float32)
    inv_sqrt_d = 0.015625

    max_score = -1e38
    for j in range(S):
        dot = 0.0
        for d in range(D):
            dot += X[i * D + d] * X[j * D + d]
        sc = dot * inv_sqrt_d
        scores[j] = sc
        if sc > max_score:
            max_score = sc

    sum_exp = 0.0
    for j in range(S):
        ex = math.exp(scores[j] - max_score)
        scores[j] = ex
        sum_exp += ex

    for j in range(S):
        scores[j] = scores[j] / sum_exp

    for d in range(D):
        acc = 0.0
        for j in range(S):
            acc += scores[j] * X[j * D + d]
        Y[i * D + d] = acc


def get_input_data():
    np.random.seed(42)
    return np.random.rand(S * D).astype(np.float32)


def run_numba_cuda_mlir_version(h_X):
    d_X = cuda.to_device(h_X)
    d_Y = cuda.device_array(S * D, dtype=np.float32)
    numba_cuda_mlir_simple_attention[S, 1, 0, S * 4](d_X, d_Y, S, D)
    cuda.synchronize()
    return d_Y.copy_to_host()


def test_attention():
    from conftest import verify_against_reference

    h_X = get_input_data()
    reference = attention_numpy(h_X, S, D)
    numba_cuda_mlir_output = run_numba_cuda_mlir_version(h_X)
    verify_against_reference(
        reference, numba_cuda_mlir_output, tolerance=2e-2, name="numba-cuda-mlir"
    )


def test_attention_benchmark(benchmark_runner):
    benchmark_runner(script=__file__)


def run_benchmark_main(compile_mode="cold", backend=BACKEND_BOTH):
    sig = "void(float32[::1], float32[::1], int64, int64)"
    prepare_compile_measurement(compile_mode, backend)

    compile_times = {}
    if should_run_backend(backend, BACKEND_NUMBA_CUDA):
        compile_times[BACKEND_NUMBA_CUDA] = time_compile(numba_cuda_simple_attention.compile, sig)
    if should_run_backend(backend, BACKEND_NUMBA_CUDA_MLIR):
        compile_times[BACKEND_NUMBA_CUDA_MLIR] = time_compile(
            numba_cuda_mlir_simple_attention.compile, sig
        )
    print_compile_times(compile_times)

    h_X = get_input_data()
    shared_mem_size = S * 4

    if should_run_backend(backend, BACKEND_NUMBA_CUDA):
        d_X = numba_cuda.to_device(h_X)
        d_Y = numba_cuda.device_array(S * D, dtype=np.float32)
        numba_cuda_simple_attention[S, 1, 0, shared_mem_size](d_X, d_Y, S, D)
        numba_cuda.synchronize()

    if should_run_backend(backend, BACKEND_NUMBA_CUDA_MLIR):
        d_X = cuda.to_device(h_X)
        d_Y = cuda.device_array(S * D, dtype=np.float32)
        numba_cuda_mlir_simple_attention[S, 1, 0, shared_mem_size](d_X, d_Y, S, D)
        cuda.synchronize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attention benchmark")
    add_compile_mode_arg(parser)
    add_backend_arg(parser)
    args = parser.parse_args()
    run_benchmark_main(args.compile_mode, args.backend)
