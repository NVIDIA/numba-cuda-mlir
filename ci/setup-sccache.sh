#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Configure sccache with RAPIDS S3 backend.
# Assumes sccache is already installed (done in CI before_script).
#
# Based on Paul Taylor's gist:
#   https://gist.github.com/trxcllnt/eaab5d814dd3069ec2103a7cccabf5d1
#
# Requires: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
# env vars (set by aws-actions/configure-aws-credentials in the workflow).
set -euo pipefail

if [ -z "${AWS_ACCESS_KEY_ID:-}" ]; then
    echo ">>> WARNING: AWS_ACCESS_KEY_ID not set, sccache S3 backend will not work"
    echo ">>> sccache will fall back to local mode"
    exit 0
fi

# --- Configure sccache (S3 only, no disk cache in CI) ---
echo ">>> Configuring sccache"
mkdir -p ~/.config/sccache
cat <<EOF > ~/.config/sccache/config
[cache.s3]
bucket = "rapids-sccache-devs"
region = "us-east-2"
key_prefix = "numba_cuda_mlir"

[cache.s3.preprocessor_cache_mode]
use_preprocessor_cache_mode = true
key_prefix = "numba_cuda_mlir/preprocessor"
EOF

# --- Start sccache daemon ---
export SCCACHE_IDLE_TIMEOUT=0
export SCCACHE_ERROR_LOG=/tmp/sccache.log
sccache --start-server
sccache --show-stats

echo ">>> sccache ready"
