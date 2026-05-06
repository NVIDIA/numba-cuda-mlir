# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time


BACKEND_BOTH = "both"
BACKEND_NUMBA_CUDA = "numba-cuda"
BACKEND_NUMBA_CUDA_MLIR = "numba-cuda-mlir"
BACKENDS = (BACKEND_BOTH, BACKEND_NUMBA_CUDA, BACKEND_NUMBA_CUDA_MLIR)


class _SkippedBackend:
    def jit(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(func):
            return func

        return decorator

    def __getattr__(self, name):
        return self

    def __call__(self, *args, **kwargs):
        return self


_SKIPPED_BACKEND = _SkippedBackend()


def skipped_backend():
    return _SKIPPED_BACKEND


def selected_backend_from_argv(argv=None):
    if argv is None:
        import sys

        argv = sys.argv[1:]

    for i, arg in enumerate(argv):
        if arg == "--backend" and i + 1 < len(argv):
            return _validate_backend(argv[i + 1])
        if arg.startswith("--backend="):
            return _validate_backend(arg.split("=", 1)[1])
    return BACKEND_BOTH


def _validate_backend(backend):
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    return backend


def should_run_backend(selected_backend, backend):
    return selected_backend in (BACKEND_BOTH, backend)


def backend_result_key(backend):
    if backend == BACKEND_NUMBA_CUDA:
        return "numba-cuda"
    if backend == BACKEND_NUMBA_CUDA_MLIR:
        return "numba_cuda_mlir"
    raise ValueError(f"Unsupported result backend: {backend}")


def backend_display_name(backend):
    if backend == BACKEND_NUMBA_CUDA:
        return "Numba-CUDA"
    if backend == BACKEND_NUMBA_CUDA_MLIR:
        return "numba-cuda-mlir"
    raise ValueError(f"Unsupported display backend: {backend}")


def add_compile_mode_arg(parser):
    parser.add_argument(
        "--compile-mode",
        choices=("cold", "warm"),
        default="cold",
        help="Compile measurement mode: cold includes one-time setup, warm excludes it.",
    )


def add_backend_arg(parser):
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default=BACKEND_BOTH,
        help="Backend to run for direct benchmark execution.",
    )


def prepare_compile_measurement(compile_mode, backend=BACKEND_BOTH):
    if compile_mode == "warm":
        warm_compile_setup(backend)
    elif compile_mode != "cold":
        raise ValueError(f"Unknown compile mode: {compile_mode}")


def warm_compile_setup(backend=BACKEND_BOTH):
    sig = "void(float32[::1])"
    if should_run_backend(backend, BACKEND_NUMBA_CUDA):
        import numba.cuda as numba_cuda

        @numba_cuda.jit
        def _numba_cuda_warmup_kernel(x):
            if numba_cuda.threadIdx.x == 0:
                x[0] = x[0]

        _numba_cuda_warmup_kernel.compile(sig)

    if should_run_backend(backend, BACKEND_NUMBA_CUDA_MLIR):
        from numba_cuda_mlir import cuda as cusimt_cuda

        @cusimt_cuda.jit
        def _cusimt_warmup_kernel(x):
            if cusimt_cuda.threadIdx.x == 0:
                x[0] = x[0]

        _cusimt_warmup_kernel.compile(sig)


def print_compile_times(backend_times):
    print("\n=== COMPILE TIMES ===")
    for backend in (BACKEND_NUMBA_CUDA, BACKEND_NUMBA_CUDA_MLIR):
        compile_time = backend_times.get(backend)
        if compile_time is not None:
            print(f"{backend_display_name(backend)}: {compile_time:.3f} ms")


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
