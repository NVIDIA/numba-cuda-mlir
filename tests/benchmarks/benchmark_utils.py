# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time

import numba.cuda as numba_cuda
from numba_cuda_mlir import cuda as cusimt_cuda


@numba_cuda.jit
def _numba_cuda_warmup_kernel(x):
    if numba_cuda.threadIdx.x == 0:
        x[0] = x[0]


@cusimt_cuda.jit
def _cusimt_warmup_kernel(x):
    if cusimt_cuda.threadIdx.x == 0:
        x[0] = x[0]


def add_compile_mode_arg(parser):
    parser.add_argument(
        "--compile-mode",
        choices=("cold", "warm"),
        default="cold",
        help="Compile measurement mode: cold includes one-time setup, warm excludes it.",
    )


def prepare_compile_measurement(compile_mode):
    if compile_mode == "warm":
        warm_compile_setup()
    elif compile_mode != "cold":
        raise ValueError(f"Unknown compile mode: {compile_mode}")


def warm_compile_setup():
    sig = "void(float32[::1])"
    _numba_cuda_warmup_kernel.compile(sig)
    _cusimt_warmup_kernel.compile(sig)


def time_compile(compile_func, *sigs):
    start = time.perf_counter()
    for sig in sigs:
        compile_func(sig)
    return (time.perf_counter() - start) * 1000


def time_compile_sequence(*dispatcher_sigs):
    start = time.perf_counter()
    for dispatcher, sig in dispatcher_sigs:
        dispatcher.compile(sig)
    return (time.perf_counter() - start) * 1000
