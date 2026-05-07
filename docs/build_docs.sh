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

# SPHINX_CUDA_PYTHON_VER is used to create a subdir under build/html
# (the Makefile file for sphinx-build also honors it if defined).
# If there's a post release (ex: .post1) we don't want it to show up in the
# version selector or directory structure.
if [[ -z "${NUMBA_CUDA_MLIR_VER}" ]]; then
    export NUMBA_CUDA_MLIR_VER=$(python -c "from importlib.metadata import version; \
                                            ver = '.'.join(str(version('numba_cuda_mlir')).split('.')[:3]); \
                                            print(ver)" \
                                 | awk -F'+' '{print $1}')
fi

sphinx-build -b html source build/html/$NUMBA_CUDA_MLIR_VER

cp versions.json build/html
echo "<meta http-equiv=\"refresh\" content=\"0; url=latest/\" />" > build/html/index.html

# ensure that the latest docs is the one we built
if [[ $LATEST_ONLY == "0" ]]; then
    cp -r build/html/${NUMBA_CUDA_MLIR_VER} build/html/latest
else
    mv build/html/${NUMBA_CUDA_MLIR_VER} build/html/latest
fi

# ensure that the Sphinx reference uses the latest docs
cp build/html/latest/objects.inv build/html
