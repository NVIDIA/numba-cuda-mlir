#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI test for numba_cuda_mlir - creates env, installs numba_cuda_mlir, runs tests."""

import argparse
import os
import sys
from pathlib import Path

from env_utils import (
    VEnv,
    install_numba_cuda_mlir_editable,
    NUMBA_CUDA_MLIR_ROOT,
    run,
    parse_junit_xml,
    JUnitResults,
    add_standard_parser_args,
    resolve_venv,
)

TESTING_DIR = NUMBA_CUDA_MLIR_ROOT / "tests" / "numba_cuda_tests" / "testing"


def run_tests(venv: VEnv, pytest_args: list = None) -> tuple[JUnitResults, Path]:
    junit_xml = NUMBA_CUDA_MLIR_ROOT / "junit-results.xml"
    venv.run_python(
        "-m",
        "pytest",
        str(NUMBA_CUDA_MLIR_ROOT / "tests"),
        "--ignore=tests/benchmarks",
        "-W",
        "ignore::UserWarning",
        "--tb=no",
        f"--junit-xml={junit_xml}",
        *(pytest_args or []),
        check=False,
    )
    return parse_junit_xml(junit_xml), junit_xml


def main():
    parser = argparse.ArgumentParser(description="Run numba_cuda_mlir CI tests")
    add_standard_parser_args(parser)
    args = parser.parse_args()

    with resolve_venv(
        args, "numba_cuda_mlir_test_venv", "numba_cuda_mlir_test_"
    ) as venv:
        install_numba_cuda_mlir_editable(venv)
        run(["make", "-C", str(TESTING_DIR)])
        os.environ["NUMBA_CUDA_MLIR_TEST_BIN_DIR"] = str(TESTING_DIR)
        results, _ = run_tests(venv, args.pytest_args)

    results.print_summary()
    if results.has_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
