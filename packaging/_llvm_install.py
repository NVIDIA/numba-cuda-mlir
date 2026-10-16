# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for the `cuda-mlir-core` and `cuda-mlir-core-dev` wheels.

Both package files out of the install tree that `ci/build-llvm-modern.sh`
produces, so neither compiles anything of its own.
"""

import os
import re
import shutil
from pathlib import Path

PACKAGING_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGING_ROOT.parent

# LLVM's own major version at the pinned commit. The commit is what actually
# identifies the build; this is here so the version sorts sensibly.
LLVM_MAJOR = "23"


def llvm_commit() -> str:
    """The pinned LLVM commit, read from the same file the build script reads."""
    text = (REPO_ROOT / "ci" / "llvm-version.env").read_text(encoding="utf-8")
    match = re.search(r'^LLVM_MODERN_COMMIT="([0-9a-f]+)"', text, re.MULTILINE)
    if not match:
        raise RuntimeError("LLVM_MODERN_COMMIT not found in ci/llvm-version.env")
    return match.group(1)


def version() -> str:
    """A version whose local segment names the build rather than the release.

    Consumers pin this exactly. Two builds of the same LLVM version that differ
    on RTTI or assertions do not interoperate, so a release number alone is not
    a sufficient compatibility token.
    """
    return f"{LLVM_MAJOR}.0.0.dev0+{llvm_commit()[:8]}"


def install_root() -> Path:
    """The install tree to package, from LLVM_MODERN_INSTALL."""
    root = os.environ.get("LLVM_MODERN_INSTALL")
    if not root:
        raise RuntimeError(
            "LLVM_MODERN_INSTALL is not set. Run ci/build-llvm-modern.sh first "
            "and point it at the resulting install tree."
        )
    path = Path(root).resolve()
    if not path.is_dir():
        raise RuntimeError(f"LLVM_MODERN_INSTALL does not exist: {path}")
    return path


def stage(src: Path, dest: Path) -> None:
    """Replace `dest` with a copy of `src`, which must exist."""
    if not src.exists():
        raise RuntimeError(f"expected to package {src}, which does not exist")
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dest, symlinks=True)
    else:
        shutil.copy2(src, dest)
