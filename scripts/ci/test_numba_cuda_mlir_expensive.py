#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI test for expensive numba_cuda_mlir tests - benchmarks requiring cupy, torch, etc."""

import argparse
import sys
from pathlib import Path

from env_utils import (
    VEnv,
    install_numba_cuda_mlir_editable,
    pin_nvidia_packages,
    get_cuda_config,
    get_cuda_version,
    NUMBA_CUDA_MLIR_ROOT,
    parse_junit_xml,
    JUnitResults,
    add_standard_parser_args,
    resolve_venv,
)

BENCHMARKS_DIR = NUMBA_CUDA_MLIR_ROOT / "tests" / "benchmarks"


def install_expensive_dependencies(venv: VEnv) -> None:
    cfg = get_cuda_config()
    cuda_ver = get_cuda_version()
    print(f"[expensive] Installing {cfg['cupy']}...")
    venv.install(cfg["cupy"])
    print("[expensive] Installing torch and torchvision...")
    venv.run_pip(
        "install",
        "torch",
        "torchvision",
        "--index-url",
        cfg["torch_index"],
    )
    print("[expensive] Pinning nvidia packages...")
    pin_nvidia_packages(venv, cuda_ver)
    print("[expensive] Installing tqdm...")
    venv.install("tqdm")
    print("[expensive] Installing tiktoken...")
    venv.install("tiktoken")
    print("[expensive] Installing transformers...")
    venv.install("transformers")
    print(f"[expensive] Installing {cfg['nvmath']}...")
    venv.install(cfg["nvmath"])
    print("[expensive] Installing cuda-cooperative...")
    venv.install("cuda-cooperative")
    print("[expensive] All dependencies installed.")


def run_benchmarks(venv: VEnv, pytest_args: list = None) -> tuple[JUnitResults, Path]:
    print(f"\n[expensive] Running benchmark tests from {BENCHMARKS_DIR}...")
    junit_xml = NUMBA_CUDA_MLIR_ROOT / "junit-benchmark-results.xml"
    venv.run_python(
        "-m",
        "pytest",
        str(BENCHMARKS_DIR),
        "-v",
        "--tb=short",
        "-W",
        "ignore::UserWarning",
        f"--junit-xml={junit_xml}",
        "-o",
        "addopts=",
        *(pytest_args or []),
        check=False,
    )
    results = parse_junit_xml(junit_xml)
    if not junit_xml.exists():
        print(f"[expensive] WARNING: {junit_xml} not found!")
    print(
        f"[expensive] Benchmark results: {{'passed': {results.passed}, 'failed': {results.failed}, 'errors': {results.errors}, 'skipped': {results.skipped}}}"
    )
    return results, junit_xml


def main():
    parser = argparse.ArgumentParser(
        description="Run expensive numba_cuda_mlir tests (benchmarks with cupy, torch)"
    )
    add_standard_parser_args(parser)
    args = parser.parse_args()
    print(
        f"[expensive] Starting with args: venv={args.venv}, keep_venv={args.keep_venv}, pytest_args={args.pytest_args}"
    )

    with resolve_venv(
        args, "numba_cuda_mlir_expensive_test_venv", "numba_cuda_mlir_expensive_test_"
    ) as venv:
        install_numba_cuda_mlir_editable(venv)
        install_expensive_dependencies(venv)
        results, _ = run_benchmarks(venv, args.pytest_args)

    results.print_summary()
    if results.has_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
