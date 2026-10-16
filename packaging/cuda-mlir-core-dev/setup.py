# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The `cuda-mlir-core-dev` wheel: what you need to compile against cudamlir.

Headers, the CMake package, `mlir-tblgen`, and the link archives. This is the
half a dialect build needs, and the reason a dialect build does not have to
compile LLVM itself.

Independent of the Python version -- it holds no extension modules -- but not
of the platform.
"""

import sys
from pathlib import Path

from setuptools import Distribution, setup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _llvm_install import install_root, stage, version  # noqa: E402

HERE = Path(__file__).resolve().parent
PKG = HERE / "mlir_dev"


class BinaryDistribution(Distribution):
    """Platform-specific, Python-version agnostic."""

    def has_ext_modules(self):
        return True

    def get_tag(self):
        return "py3", "none", None


root = install_root()
stage(root / "include", PKG / "include")
stage(root / "lib" / "cmake", PKG / "lib" / "cmake")
stage(root / "bin" / "mlir-tblgen", PKG / "bin" / "mlir-tblgen")

# MLIRConfig.cmake declares a target for every static archive and
# existence-checks all of them, so find_package fails unless they are all here.
# This is what makes the wheel large; see packaging/README.md.
for archive in sorted((root / "lib").glob("*.a")):
    stage(archive, PKG / "lib" / archive.name)

setup(
    name="cuda-mlir-core-dev",
    version=version(),
    description="MLIR headers, CMake package, and mlir-tblgen",
    long_description=(HERE.parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="NVIDIA Corporation",
    license="Apache-2.0",
    python_requires=">=3.10",
    packages=["mlir_dev"],
    include_package_data=True,
    package_data={"mlir_dev": ["**/*"]},
    distclass=BinaryDistribution,
    zip_safe=False,
)
