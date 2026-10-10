# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The `cuda-mlir-core` wheel: MLIR's Python bindings, as built by this repo.

Nothing is compiled here. `ci/build-llvm-modern.sh` has already produced the
bindings; this packages them.
"""

import sys
from pathlib import Path

from setuptools import Distribution, setup

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from _llvm_install import install_root, stage, version  # noqa: E402

HERE = Path(__file__).resolve().parent


class BinaryDistribution(Distribution):
    """Force a platform- and Python-specific wheel tag.

    The tree staged below contains Python extension modules, but setuptools
    cannot see that because it did not build them, and would otherwise tag the
    wheel `py3-none-any`. Installing that on the wrong interpreter produces a
    package whose imports fail.
    """

    def has_ext_modules(self):
        return True


stage(install_root() / "python_packages" / "cuda_mlir_core" / "cudamlir", HERE / "cudamlir")

setup(
    name="cuda-mlir-core",
    version=version(),
    description="MLIR Python bindings for the CUDA ecosystem",
    long_description=(HERE.parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="NVIDIA Corporation",
    license="Apache-2.0",
    python_requires=">=3.10",
    packages=["cudamlir"],
    include_package_data=True,
    package_data={"cudamlir": ["**/*"]},
    distclass=BinaryDistribution,
    zip_safe=False,
)
