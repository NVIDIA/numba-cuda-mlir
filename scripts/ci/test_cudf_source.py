#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI test for cuDF compatibility with cusimt, building cudf from source."""

import argparse
import os
import sys
from pathlib import Path

from env_utils import (
    get_cuda_config,
    get_cuda_version,
    CUSIMT_ROOT,
    run,
    parse_junit_xml,
    save_junit_errors,
    print_failures_by_file,
    JUnitResults,
    add_standard_parser_args,
    MLIR_PACKAGES,
    MLIR_FIND_LINKS,
)

from test_cudf import clone_cudf

SRC_BUILD_REPO = "https://github.com/rapidsai/cudf.git"
SRC_BUILD_BRANCH = "release/25.12"
MAMBAFORGE_URL = (
    "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
)
CONDA_ENV_NAME = "cudf_build"

CUDA_TO_ENV_YAML = {
    "12.9": "all_cuda-129_arch-x86_64.yaml",
    "13.1": "all_cuda-131_arch-x86_64.yaml",
}


def get_env_yaml(cuda_version: str) -> str:
    if cuda_version not in CUDA_TO_ENV_YAML:
        raise ValueError(
            f"Unsupported CUDA version {cuda_version} for cudf source build. "
            f"Supported: {list(CUDA_TO_ENV_YAML)}"
        )
    return CUDA_TO_ENV_YAML[cuda_version]


def install_mamba(install_dir: Path) -> Path:
    """Install Mambaforge to install_dir; return path to mamba binary."""
    install_dir = install_dir.resolve()
    run(
        [
            "wget",
            "-q",
            MAMBAFORGE_URL,
            "-O",
            "/tmp/Mambaforge-Linux-x86_64.sh",
        ]
    )
    run(
        [
            "bash",
            "/tmp/Mambaforge-Linux-x86_64.sh",
            "-b",
            "-p",
            str(install_dir),
        ]
    )
    return install_dir / "bin" / "mamba"


def setup_cudf_source_env(
    clone_dir: Path,
    mamba_install_dir: Path,
    cudf_branch: str,
    env_yaml_name: str,
    extras: str,
    repo: str,
) -> tuple[Path, Path]:
    """Install mamba, clone cudf, create conda env, build cudf, install MLIR and cusimt.
    Returns (conda_exe, cudf_dir). Skips mamba install / clone / env create if already present.
    """
    clone_dir.mkdir(parents=True, exist_ok=True)

    mamba_install_dir = mamba_install_dir.resolve()
    conda_bin = mamba_install_dir / "bin"
    if (conda_bin / "conda").exists() or (conda_bin / "mamba").exists():
        mamba_bin = conda_bin / "mamba" if (conda_bin / "mamba").exists() else conda_bin / "conda"
    else:
        mamba_bin = install_mamba(mamba_install_dir)
        conda_bin = mamba_bin.parent
    mamba_prefix = conda_bin.parent
    path_prepend = str(conda_bin) + ":" + os.environ.get("PATH", "")
    env = os.environ.copy()
    env["PATH"] = path_prepend
    for key in ("CONDA_EXE", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL"):
        env.pop(key, None)
    env["CONDA_PREFIX"] = str(mamba_prefix)

    cudf_dir = clone_dir / "cudf"
    if not cudf_dir.exists():
        cudf_dir = clone_cudf(clone_dir, cudf_branch, repo)
    print("cudf branch commit (before build):", flush=True)
    run(f"git -C {cudf_dir} log -1 --format='%H %h %s'")
    env_yaml = cudf_dir / "conda" / "environments" / env_yaml_name

    env_exists = (mamba_prefix / "envs" / CONDA_ENV_NAME).exists()
    if not env_exists:
        run(
            [
                str(mamba_bin),
                "env",
                "create",
                "-f",
                str(env_yaml),
                "-n",
                CONDA_ENV_NAME,
                "-y",
            ],
            env=env,
        )

    # Ensure nvidia-smi is reachable inside the conda env (conda run
    # may not include /usr/bin in PATH).
    env_bin = mamba_prefix / "envs" / CONDA_ENV_NAME / "bin"
    nvidia_smi = env_bin / "nvidia-smi"
    if not nvidia_smi.exists():
        import shutil

        host_smi = shutil.which("nvidia-smi")
        if host_smi:
            nvidia_smi.symlink_to(host_smi)

    conda_exe = conda_bin / "conda"  # prefix/bin/conda
    run(
        [
            str(conda_exe),
            "run",
            "-n",
            CONDA_ENV_NAME,
            "bash",
            "-c",
            f"cd {cudf_dir} && bash build.sh",
        ],
        env=env,
    )
    run(
        [
            str(conda_exe),
            "run",
            "-n",
            CONDA_ENV_NAME,
            "pip",
            "install",
            "--upgrade",
            *MLIR_PACKAGES,
            "-f",
            MLIR_FIND_LINKS,
        ],
        env=env,
        check=True,
    )
    run(
        [
            str(conda_exe),
            "run",
            "-n",
            CONDA_ENV_NAME,
            "pip",
            "install",
            "-e",
            f"{CUSIMT_ROOT}[{extras}]",
        ],
        env=env,
        check=True,
    )
    return conda_exe, cudf_dir


def run_tests_conda(
    conda_exe: Path,
    env_name: str,
    cudf_dir: Path,
    junit_xml: Path,
    groupby_junit: Path,
    pytest_args: list | None = None,
) -> tuple[JUnitResults, Path]:
    common = ["-W", "ignore::UserWarning", "-v", "--tb=no", *(pytest_args or [])]

    scalar_udf = cudf_dir / "python/cudf/cudf/tests/dataframe/methods/test_apply.py"
    groupby_udf = cudf_dir / "python/cudf/cudf/tests/groupby/test_apply.py"
    nrt_stats = cudf_dir / "python/cudf/cudf/tests/private_objects/test_nrt_stats.py"

    run(
        [
            str(conda_exe),
            "run",
            "-n",
            env_name,
            "python",
            "-m",
            "pytest",
            str(scalar_udf),
            str(nrt_stats),
            *common,
            f"--junit-xml={junit_xml}",
        ],
        check=False,
    )
    run(
        [
            str(conda_exe),
            "run",
            "-n",
            env_name,
            "python",
            "-m",
            "pytest",
            str(groupby_udf),
            "-k",
            "test_groupby_apply",
            *common,
            f"--junit-xml={groupby_junit}",
        ],
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
    parser = argparse.ArgumentParser(
        description="Run cuDF CI tests with cusimt (cudf built from source)"
    )
    add_standard_parser_args(parser, CUSIMT_ROOT / "cudf_source_errors.txt")
    parser.add_argument(
        "--cudf-dir",
        type=Path,
        help="Path to existing cudf clone (skips cloning and build)",
    )
    parser.add_argument(
        "--cudf-branch",
        default=SRC_BUILD_BRANCH,
        help=f"cudf branch to clone (default: {SRC_BUILD_BRANCH})",
    )
    parser.add_argument(
        "--mamba-install-dir",
        type=Path,
        default=Path(os.environ.get("MAMBA_INSTALL_DIR", "/tmp/mambaforge")),
        help="Directory to install Mambaforge into",
    )
    args = parser.parse_args()

    cuda_version = get_cuda_version()
    env_yaml_name = get_env_yaml(cuda_version)
    cfg = get_cuda_config(cuda_version)
    extras = cfg["extra"]  # cu12 or cu13

    cudf_dir = args.cudf_dir.resolve() if args.cudf_dir else None
    clone_dir = CUSIMT_ROOT / "cudf_source_clone"
    junit_xml = CUSIMT_ROOT / "cudf-source-junit-results.xml"
    groupby_junit = CUSIMT_ROOT / "cudf-source-junit-groupby.xml"

    if cudf_dir is None:
        cudf_repo = os.environ.get("CUDF_REPO", SRC_BUILD_REPO)
        conda_exe, cudf_dir = setup_cudf_source_env(
            clone_dir,
            args.mamba_install_dir,
            args.cudf_branch,
            env_yaml_name,
            extras,
            repo=cudf_repo,
        )
    else:
        # Use existing clone: assume env already exists and cudf is built
        conda_exe = args.mamba_install_dir / "bin" / "conda"
        if not conda_exe.exists():
            conda_exe = Path("/tmp/mambaforge/bin/conda")

    results, _ = run_tests_conda(
        conda_exe,
        CONDA_ENV_NAME,
        cudf_dir,
        junit_xml,
        groupby_junit,
        args.pytest_args,
    )
    results.print_summary()
    save_junit_errors(junit_xml, args.errors_file)
    print(f"Errors saved to: {args.errors_file}")
    print_failures_by_file(junit_xml)
    if results.has_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
