#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

set -ex

if [[ "$#" == "0" ]]; then
    LATEST_ONLY="0"
elif [[ "$#" == "1" && "$1" == "latest-only" ]]; then
    LATEST_ONLY="1"
else
    echo "usage: ./build_docs.sh [latest-only]"
    exit 1
fi

# SPHINX_NUMBA_CUDA_MLIR_VER is used to create a subdir under build/html
# and is also read by conf.py to drive the version switcher's version_match.
# For latest-only (dev) builds we pin it to "latest" so the switcher highlights
# the "latest" entry in versions.json. For full builds we use the wheel
# version (stripping any post-release / local suffixes).
if [[ "${LATEST_ONLY}" == "1" ]]; then
    export SPHINX_NUMBA_CUDA_MLIR_VER="latest"
elif [[ -z "${SPHINX_NUMBA_CUDA_MLIR_VER}" ]]; then
    export SPHINX_NUMBA_CUDA_MLIR_VER=$(python -c "from importlib.metadata import version; \
                                                   ver = '.'.join(str(version('numba_cuda_mlir')).split('.')[:3]); \
                                                   print(ver)" \
                                        | awk -F'+' '{print $1}')
fi

sphinx-build -b html source build/html/$SPHINX_NUMBA_CUDA_MLIR_VER

cp versions.json build/html
echo "<meta http-equiv=\"refresh\" content=\"0; url=latest/\" />" > build/html/index.html

# ensure that the latest docs is the one we built
if [[ "${LATEST_ONLY}" == "0" ]]; then
    cp -r build/html/${SPHINX_NUMBA_CUDA_MLIR_VER} build/html/latest
fi
# else: SPHINX_NUMBA_CUDA_MLIR_VER == "latest", so the build already landed in build/html/latest

# ensure that the Sphinx reference uses the latest docs
cp build/html/latest/objects.inv build/html
