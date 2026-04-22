#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Two-layer binary cache for LLVM build artifacts:
#   Layer 1: Artifactory (sw-cuda-python-generic-local) — fast internal cache
#   Layer 2: S3 (rapids-sccache-devs) — cross-platform, accessible from GitHub
#
# Auth:
#   Artifactory: ARTIF_GENERIC_TOKEN env var (Bearer auth, optional)
#   S3: ~/.aws/credentials (set up by setup-sccache.sh, optional)
#
# Both layers are optional — missing credentials are silently skipped.
set -euo pipefail

ARTIF_SERVER="https://artifactory.nvidia.com/artifactory"
ARTIF_REPO="sw-cuda-python-generic-local"
ARTIF_PREFIX="cusimt/llvm-cache"

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

# --- Artifactory helpers ---

_artif_url() {
    echo "${ARTIF_SERVER}/${ARTIF_REPO}/${ARTIF_PREFIX}/${1}.tar.gz"
}

_artif_download() {
    local key="$1" dest_dir="$2"
    [ -z "${ARTIF_GENERIC_TOKEN:-}" ] && return 1
    local url=$(_artif_url "$key")
    local http_code
    http_code=$(curl -s -o /dev/null -w "%{http_code}" \
        -H "Authorization: Bearer ${ARTIF_GENERIC_TOKEN}" "$url")
    if [ "$http_code" = "200" ]; then
        echo ">>> [Artifactory] Cache HIT: ${key}"
        curl -s -H "Authorization: Bearer ${ARTIF_GENERIC_TOKEN}" "$url" \
            -o "${key}.tar.gz"
        mkdir -p "$dest_dir"
        tar xzf "${key}.tar.gz" -C "$dest_dir" --strip-components=1
        rm -f "${key}.tar.gz"
        return 0
    fi
    return 1
}

_artif_upload() {
    local key="$1" src_dir="$2"
    [ -z "${ARTIF_GENERIC_TOKEN:-}" ] && return 0
    local url=$(_artif_url "$key")
    echo ">>> [Artifactory] Uploading ${key}"
    tar czf "${key}.tar.gz" -C "$(dirname "$src_dir")" "$(basename "$src_dir")"
    curl -s -H "Authorization: Bearer ${ARTIF_GENERIC_TOKEN}" \
        -T "${key}.tar.gz" "$url"
    rm -f "${key}.tar.gz"
}

# --- S3 helpers ---

_s3_path() {
    echo "s3://${S3_BUCKET}/${S3_PREFIX}/${1}.tar.gz"
}

_s3_exists() {
    local key="$1"
    command -v aws &>/dev/null || return 1
    [ -f ~/.aws/credentials ] || return 1
    aws s3 ls "$(_s3_path "$key")" &>/dev/null
}

_s3_download() {
    local key="$1" dest_dir="$2"
    command -v aws &>/dev/null || return 1
    [ -f ~/.aws/credentials ] || return 1
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
    [ -f ~/.aws/credentials ] || return 0
    local s3_path=$(_s3_path "$key")
    echo ">>> [S3] Uploading ${key}"
    tar czf "${key}.tar.gz" -C "$(dirname "$src_dir")" "$(basename "$src_dir")"
    aws s3 cp "${key}.tar.gz" "$s3_path"
    rm -f "${key}.tar.gz"
}

# --- Public API ---

# cache_download <key> <dest_dir>
#   Try Artifactory first, then S3. Returns 0 on hit, 1 on miss.
cache_download() {
    local key="$1" dest_dir="$2"
    echo ">>> Checking cache: ${key}"

    # Layer 1: Artifactory
    if _artif_download "$key" "$dest_dir"; then
        # Backfill S3 only if it doesn't already have it
        if ! _s3_exists "$key"; then
            _s3_upload "$key" "$dest_dir" || true
        fi
        return 0
    fi
    echo ">>> [Artifactory] Cache MISS"

    # Layer 2: S3
    if _s3_download "$key" "$dest_dir"; then
        # Backfill Artifactory so next internal run is fast
        _artif_upload "$key" "$dest_dir" || true
        return 0
    fi
    echo ">>> [S3] Cache MISS"

    return 1
}

# cache_upload <key> <src_dir>
#   Upload to both Artifactory and S3.
cache_upload() {
    local key="$1" src_dir="$2"
    _artif_upload "$key" "$src_dir" || true
    _s3_upload "$key" "$src_dir" || true
}
