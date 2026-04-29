#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CI test for numbast-internal compatibility with cusimt."""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Optional

from env_utils import (  # type: ignore[import-not-found]
    CUSIMT_ROOT,
    MLIR_FIND_LINKS,
    MLIR_PACKAGES,
    JUnitResults,
    get_cuda_config,
    parse_junit_xml,
    print_failures_by_file,
    run,
    save_junit_errors,
)

NUMBAST_INTERNAL_REPO = os.environ.get(
    "NUMBAST_INTERNAL_REPO",
    "https://gitlab-master.nvidia.com/wangm/numbast-cusimt.git",
)

NUMBAST_TEST_TARGETS = [
    "numbast-cusimt/",
]

CONDA_CREATE_BASE_DEPS = [
    "pip",
    "clangdev>=18,<22.0",
    "cmake",
    "ninja",
]


def sanitize_env_name(raw: str) -> str:
    env_name = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.strip())
    env_name = env_name.strip("-")
    return env_name or "main"


def get_conda_executable() -> str:
    conda_exe = shutil.which("conda")
    if conda_exe:
        return conda_exe
    fallback = Path("/opt/conda/bin/conda")
    if fallback.exists():
        return str(fallback)
    raise RuntimeError(
        "conda executable not found. This test requires conda to create "
        "an environment and build ast_canopy with clangdev."
    )


def conda_env_exists(conda_exe: str, env_name: str) -> bool:
    result = run([conda_exe, "env", "list", "--json"], capture=True)
    envs = json.loads(result.stdout).get("envs", [])
    return any(Path(p).name == env_name for p in envs)


def conda_run(
    conda_exe: str,
    env_name: str,
    cmd: list[str],
    *,
    check: bool = True,
    cwd: Optional[Path] = None,
):
    return run([conda_exe, "run", "-n", env_name, *cmd], check=check, cwd=cwd)


def ensure_conda_env(conda_exe: str, env_name: str) -> None:
    if conda_env_exists(conda_exe, env_name):
        print(f"Using existing conda env: {env_name}")
        return
    print(f"Creating conda env: {env_name}")
    python_version = os.environ.get(
        "NUMBAST_INTERNAL_PYTHON_VERSION",
        f"{sys.version_info.major}.{sys.version_info.minor}",
    )
    python_spec = f"python={python_version}"
    run(
        [
            conda_exe,
            "create",
            "-y",
            "-n",
            env_name,
            "-c",
            "conda-forge",
            "-c",
            "numba",
            "-c",
            "nvidia",
            python_spec,
            *CONDA_CREATE_BASE_DEPS,
        ]
    )


def _inject_ci_token(repo_url: str) -> str:
    token = os.environ.get("CI_JOB_TOKEN")
    if not token or not repo_url.startswith("https://"):
        return repo_url
    return repo_url.replace("https://", f"https://gitlab-ci-token:{token}@", 1)


def clone_numbast_internal(clone_dir: Path, repo: str, branch: Optional[str] = None) -> Path:
    numbast_dir = clone_dir / "numbast-internal"
    auth_repo = _inject_ci_token(repo)
    run(
        [
            "git",
            "clone",
            *(["--single-branch", "--branch", branch] if branch else []),
            auth_repo,
            str(numbast_dir),
        ]
    )
    if branch:
        run(["git", "-C", str(numbast_dir), "fetch", "origin", branch])
        run(
            [
                "git",
                "-C",
                str(numbast_dir),
                "checkout",
                "-B",
                branch,
                f"origin/{branch}",
            ]
        )
    else:
        # Explicitly fast-forward to remote HEAD to guarantee latest tip.
        run(["git", "-C", str(numbast_dir), "pull", "--ff-only"])
    return numbast_dir


def get_current_branch(repo_dir: Path) -> str:
    result = run(
        ["git", "-C", str(repo_dir), "rev-parse", "--abbrev-ref", "HEAD"],
        capture=True,
    )
    branch = result.stdout.strip()
    return branch if branch and branch != "HEAD" else "main"


def resolve_test_targets(numbast_dir: Path) -> list[str]:
    targets = []
    for rel in NUMBAST_TEST_TARGETS:
        path = numbast_dir / rel
        if path.exists():
            targets.append(str(path))
        else:
            print(f"Warning: test target missing, skipping: {path}")
    if not targets:
        raise RuntimeError("No numbast-internal test targets found in clone")
    return targets


def install_cusimt_for_numbast_internal(conda_exe: str, env_name: str, extra: str) -> None:
    wheels = sorted((CUSIMT_ROOT / "dist").glob("cusimt-*.whl"))
    if wheels:
        wheel = wheels[-1]
        spec = f"{wheel}[{extra}]"
        print(f"Installing cusimt from pre-built wheel for {extra}: {wheel.name}")
        conda_run(
            conda_exe,
            env_name,
            ["python", "-m", "pip", "install", spec],
        )
    elif os.environ.get("CI"):
        raise RuntimeError(
            "numbast_internal_tests requires a pre-built cusimt wheel in dist/. "
            "Make sure this job downloads build-wheel artifacts."
        )
    else:
        conda_run(
            conda_exe,
            env_name,
            [
                "python",
                "-m",
                "pip",
                "install",
                "--upgrade",
                *MLIR_PACKAGES,
                "-f",
                MLIR_FIND_LINKS,
            ],
        )
        conda_run(
            conda_exe,
            env_name,
            ["python", "-m", "pip", "install", "-e", f"{CUSIMT_ROOT}[{extra}]"],
        )

    conda_run(
        conda_exe,
        env_name,
        ["python", "-c", "from cusimt._mlir import ir; from cusimt import cuda"],
    )


def setup_numbast_internal_env(conda_exe: str, env_name: str, numbast_dir: Path) -> None:
    cfg = get_cuda_config()
    test_extra = f"test-{cfg['extra']}"

    conda_run(conda_exe, env_name, ["python", "-m", "pip", "install", "--upgrade", "pip"])
    install_cusimt_for_numbast_internal(conda_exe, env_name, cfg["extra"])
    conda_run(
        conda_exe,
        env_name,
        ["bash", str(numbast_dir / "ast_canopy" / "build.sh"), "--develop"],
    )
    conda_run(
        conda_exe,
        env_name,
        [
            "python",
            "-m",
            "pip",
            "install",
            "-e",
            f"{numbast_dir / 'numbast-cusimt'}[{test_extra}]",
        ],
    )
    version_pin = cfg["version_pin"]
    conda_run(
        conda_exe,
        env_name,
        [
            "python",
            "-m",
            "pip",
            "install",
            f"cuda-toolkit[cudart,crt,cccl,nvcc,nvjitlink,nvrtc,nvvm]=={version_pin}",
        ],
    )


def run_tests(
    conda_exe: str, env_name: str, numbast_dir: Path, pytest_args: list[str]
) -> tuple[JUnitResults, Path]:
    junit_xml = CUSIMT_ROOT / "numbast-internal-junit-results.xml"
    targets = resolve_test_targets(numbast_dir)

    conda_run(
        conda_exe,
        env_name,
        [
            "python",
            "-m",
            "pytest",
            *targets,
            "-W",
            "ignore::UserWarning",
            "-v",
            "--tb=no",
            f"--junit-xml={junit_xml}",
            *pytest_args,
        ],
        check=False,
    )

    return parse_junit_xml(junit_xml), junit_xml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run numbast-internal CI tests with cusimt (conda + clangdev)"
    )
    parser.add_argument(
        "--numbast-internal-dir",
        type=Path,
        help="Path to existing numbast-internal clone (skips cloning)",
    )
    parser.add_argument(
        "--numbast-internal-repo",
        default=NUMBAST_INTERNAL_REPO,
        help=f"numbast-internal repository URL (default: {NUMBAST_INTERNAL_REPO})",
    )
    parser.add_argument(
        "--numbast-internal-branch",
        default=None,
        help="Branch to clone (default: repository default branch head)",
    )
    parser.add_argument(
        "--conda-env-name",
        default=None,
        help="Conda env name (default: cloned branch name)",
    )
    parser.add_argument(
        "--keep-clone",
        action="store_true",
        help="Keep clone in numbast_internal_test_clone/ under repo root",
    )
    parser.add_argument(
        "--errors-file",
        type=Path,
        default=CUSIMT_ROOT / "numbast_internal_errors.txt",
        help="File to save error details (default: numbast_internal_errors.txt)",
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Additional arguments to pass to pytest (after --)",
    )
    return parser.parse_args()


def perform_numbast_test_run(
    conda_exe: str,
    numbast_dir: Path,
    pytest_args: list[str],
    errors_file: Path,
    *,
    branch_override: Optional[str] = None,
    env_name_override: Optional[str] = None,
) -> bool:
    """Returns True if all tests passed."""
    branch = branch_override or get_current_branch(numbast_dir)
    env_name = env_name_override or sanitize_env_name(branch)
    ensure_conda_env(conda_exe, env_name)
    setup_numbast_internal_env(conda_exe, env_name, numbast_dir)
    results, junit_xml = run_tests(conda_exe, env_name, numbast_dir, pytest_args)
    results.print_summary()
    save_junit_errors(junit_xml, errors_file)
    print(f"Errors saved to: {errors_file}")
    print_failures_by_file(junit_xml)
    return not results.has_failures


def main():
    args = parse_args()
    conda_exe = get_conda_executable()

    if args.numbast_internal_dir:
        numbast_dir = args.numbast_internal_dir.resolve()
        if not numbast_dir.exists():
            raise FileNotFoundError(f"numbast-internal dir not found: {numbast_dir}")
    elif args.keep_clone:
        clone_root = CUSIMT_ROOT / "numbast_internal_test_clone"
        clone_root.mkdir(exist_ok=True)
        clone_path = clone_root / "numbast-internal"
        if clone_path.exists():
            shutil.rmtree(clone_path)
        numbast_dir = clone_numbast_internal(
            clone_root, args.numbast_internal_repo, args.numbast_internal_branch
        )
        print(f"numbast-internal at: {numbast_dir}")
    else:
        with tempfile.TemporaryDirectory(prefix="numbast_internal_") as tmpdir:
            numbast_dir = clone_numbast_internal(
                Path(tmpdir), args.numbast_internal_repo, args.numbast_internal_branch
            )
            passed = perform_numbast_test_run(
                conda_exe,
                numbast_dir,
                args.pytest_args,
                args.errors_file,
                branch_override=args.numbast_internal_branch,
                env_name_override=args.conda_env_name,
            )
            if not passed:
                sys.exit(1)
            return

    passed = perform_numbast_test_run(
        conda_exe,
        numbast_dir,
        args.pytest_args,
        args.errors_file,
        branch_override=args.numbast_internal_branch,
        env_name_override=args.conda_env_name,
    )
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
