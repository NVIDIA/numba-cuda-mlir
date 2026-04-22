#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# S3 binary cache for LLVM build artifacts (fallback layer).
#
# The primary cache layer is GitHub Actions cache (actions/cache in build-llvm.yml).
# This script provides an S3 fallback for when the GHA cache misses (eviction,
# new branch, etc.).
#
# Auth: AWS_ACCESS_KEY_ID env var (set by aws-actions/configure-aws-credentials).
# Missing credentials are silently skipped.
set -euo pipefail

S3_BUCKET="rapids-sccache-devs"
S3_PREFIX="cusimt/llvm-binaries"

# cache_key <label> <version_id> <script_path>
cache_key() {
    local label="$1" version_id="$2" script_path="$3"
    local script_hash arch os_name version_short
    script_hash=$(sha256sum "$script_path" | cut -c1-12)
    version_short=$(echo "$version_id" | cut -c1-12)
    arch=$(uname -m)
    os_name=$(uname -s | tr '[:upper:]' '[:lower:]')
    echo "${label}-${os_name}-${arch}-${version_short}-${script_hash}"
}

# --- S3 helpers ---

_s3_path() {
    echo "s3://${S3_BUCKET}/${S3_PREFIX}/${1}.tar.gz"
}

_s3_download() {
    local key="$1" dest_dir="$2"
    command -v aws &>/dev/null || return 1
    [ -n "${AWS_ACCESS_KEY_ID:-}" ] || return 1
    local s3_path=$(_s3_path "$key")
    if aws s3 ls "$s3_path" &>/dev/null; then
        echo ">>> [S3] Cache HIT: ${key}"
        aws s3 cp "$s3_path" "${key}.tar.gz"
        mkdir -p "$dest_dir"
        tar xzf "${key}.tar.gz" -C "$dest_dir" --strip-components=1
        rm -f "${key}.tar.gz"
        return 0
    fi
    return 1
}

_s3_upload() {
    local key="$1" src_dir="$2"
    command -v aws &>/dev/null || return 0
    [ -n "${AWS_ACCESS_KEY_ID:-}" ] || return 0
    local s3_path=$(_s3_path "$key")
    echo ">>> [S3] Uploading ${key}"
    tar czf "${key}.tar.gz" -C "$(dirname "$src_dir")" "$(basename "$src_dir")"
    aws s3 cp "${key}.tar.gz" "$s3_path"
    rm -f "${key}.tar.gz"
}

# --- Public API ---

# cache_download <key> <dest_dir>
#   Try S3. Returns 0 on hit, 1 on miss.
cache_download() {
    local key="$1" dest_dir="$2"
    echo ">>> Checking cache: ${key}"
    if _s3_download "$key" "$dest_dir"; then
        return 0
    fi
    echo ">>> [S3] Cache MISS"
    return 1
}

# cache_upload <key> <src_dir>
#   Upload to S3.
cache_upload() {
    local key="$1" src_dir="$2"
    _s3_upload "$key" "$src_dir" || true
}
