#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI test for cuDF compatibility with cusimt."""

import argparse
import sys
import tempfile
from pathlib import Path

from env_utils import (
    VEnv,
    install_cusimt_editable,
    get_cuda_config,
    CUSIMT_ROOT,
    run,
    parse_junit_xml,
    save_junit_errors,
    filter_junit_failures,
    print_failures_by_file,
    JUnitResults,
    add_standard_parser_args,
    resolve_venv,
)

DEFAULT_CUDF_REPO = "https://github.com/rapidsai/cudf.git"
DEFAULT_CUDF_BRANCH = "release/25.12"
RAPIDS_INDEX_URL = "https://pypi.nvidia.com"
CUDF_TEST_DEPS = ["pytest>=8,<9", "pytest-xdist"]

PATCHES_DIR = Path(__file__).parent / "patches"
CUDF_COMPAT_PATCH = PATCHES_DIR / "cudf_numba_cuda_compatibility.patch"

DEFAULT_FAILURE_FILTERS = []


def clone_cudf(clone_dir: Path, branch: str, repo: str) -> Path:
    cudf_dir = clone_dir / "cudf"
    run(
        [
            "git",
            "clone",
            "--single-branch",
            "--branch",
            branch,
            repo,
            str(cudf_dir),
        ]
    )
    return cudf_dir


def apply_cudf_patch(venv: VEnv) -> None:
    if not CUDF_COMPAT_PATCH.exists():
        print(f"Warning: patch not found at {CUDF_COMPAT_PATCH}, skipping", flush=True)
        return
    result = venv.run_python(
        "-c", "import site; print(site.getsitepackages()[0])", capture=True
    )
    site_packages = result.stdout.strip()
    patch_path = CUDF_COMPAT_PATCH.resolve()
    # -N skips hunks already applied; exit code 1 means "already applied", not an error
    result = run(f"patch -N -p3 -d {site_packages} < {patch_path}", check=False)
    if result.returncode > 1:
        raise RuntimeError(f"patch failed with exit code {result.returncode}")


def setup_cudf_env(venv: VEnv) -> None:
    cfg = get_cuda_config()
    cudf_package = f"cudf-{cfg['extra']}==25.12.*"
    # cudf brings in numba-cuda; install cusimt after to overlay numba.cuda
    venv.run_pip(
        "install", "--extra-index-url", RAPIDS_INDEX_URL, cudf_package, *CUDF_TEST_DEPS
    )
    install_cusimt_editable(venv)
    apply_cudf_patch(venv)


def run_tests(
    venv: VEnv, cudf_dir: Path, pytest_args: list = None
) -> tuple[JUnitResults, Path]:
    junit_xml = CUSIMT_ROOT / "cudf-junit-results.xml"
    groupby_junit = CUSIMT_ROOT / "cudf-junit-groupby.xml"
    common = ["-W", "ignore::UserWarning", "-v", "--tb=no", *(pytest_args or [])]

    scalar_udf = cudf_dir / "python/cudf/cudf/tests/dataframe/methods/test_apply.py"
    groupby_udf = cudf_dir / "python/cudf/cudf/tests/groupby/test_apply.py"
    nrt_stats = cudf_dir / "python/cudf/cudf/tests/private_objects/test_nrt_stats.py"

    venv.run_python(
        "-m",
        "pytest",
        str(scalar_udf),
        str(nrt_stats),
        *common,
        f"--junit-xml={junit_xml}",
        check=False,
    )
    venv.run_python(
        "-m",
        "pytest",
        str(groupby_udf),
        "-k",
        "test_groupby_apply",
        *common,
        f"--junit-xml={groupby_junit}",
        check=False,
    )

    r1, r2 = parse_junit_xml(junit_xml), parse_junit_xml(groupby_junit)
    combined = JUnitResults(
        passed=r1.passed + r2.passed,
        failed=r1.failed + r2.failed,
        errors=r1.errors + r2.errors,
        skipped=r1.skipped + r2.skipped,
    )
    return combined, junit_xml


def main():
    parser = argparse.ArgumentParser(description="Run cuDF CI tests with cusimt")
    add_standard_parser_args(parser, CUSIMT_ROOT / "cudf_errors.txt")
    parser.add_argument(
        "--cudf-dir",
        type=Path,
        help="Path to existing cudf clone (skips cloning)",
    )
    parser.add_argument(
        "--cudf-branch",
        default=DEFAULT_CUDF_BRANCH,
        help=f"cudf branch to clone (default: {DEFAULT_CUDF_BRANCH})",
    )
    args = parser.parse_args()

    cudf_dir = args.cudf_dir.resolve() if args.cudf_dir else None

    def do_test(venv: VEnv, cudf_dir: Path) -> tuple[JUnitResults, Path]:
        setup_cudf_env(venv)
        return run_tests(venv, cudf_dir, args.pytest_args)

    results, junit_xml = None, None
    with resolve_venv(args, "cudf_test_venv", "cudf_test_") as venv:
        if cudf_dir:
            results, junit_xml = do_test(venv, cudf_dir)
        elif getattr(args, "keep_venv", False):
            clone_dir = CUSIMT_ROOT / "cudf_test_clone"
            clone_dir.mkdir(exist_ok=True)
            cudf_dir = clone_cudf(clone_dir, args.cudf_branch, DEFAULT_CUDF_REPO)
            results, junit_xml = do_test(venv, cudf_dir)
            print(f"cudf at: {cudf_dir}")
        else:
            with tempfile.TemporaryDirectory(prefix="cudf_") as tmpdir:
                cudf_dir = clone_cudf(Path(tmpdir), args.cudf_branch, DEFAULT_CUDF_REPO)
                results, junit_xml = do_test(venv, cudf_dir)

    if results and DEFAULT_FAILURE_FILTERS:
        results, filtered = filter_junit_failures(junit_xml, DEFAULT_FAILURE_FILTERS)
        if filtered:
            print(f"\nFiltered {filtered} failures matching patterns")
    if results:
        results.print_summary()
    if junit_xml:
        save_junit_errors(junit_xml, args.errors_file)
        print(f"Errors saved to: {args.errors_file}")
        print_failures_by_file(junit_xml)
    if results and results.has_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
