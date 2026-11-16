#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/llvm-version.env"

usage() {
    echo "Usage: $0 modern <host-platform> <py-tag> | llvm7 <host-platform>" >&2
    exit 2
}

hash_inputs() {
    local file label
    for file in "$@"; do
        label="${file#"${SCRIPT_DIR}/"}"
        printf '%s\n' "file:${label}"
        sha256sum "$file" | awk '{print $1}'
    done | sha256sum | cut -c1-12
}

cache_version() {
    local version_id="$1"
    if [[ "${version_id}" =~ ^[0-9a-fA-F]{40}$ ]]; then
        echo "${version_id:0:12}"
    else
        echo "${version_id}"
    fi
}

cache_arch() {
    case "$1" in
        linux-64) echo "x86_64" ;;
        linux-aarch64) echo "aarch64" ;;
        win-64) echo "amd64" ;;
        win-arm64) echo "arm64" ;;
        *) echo "Unsupported host platform: $1" >&2; exit 2 ;;
    esac
}

kind="${1:-}"
host_platform="${2:-}"

# The same scripts build every platform; Windows keys also pin the MSVC toolset.
case "${host_platform}" in
    linux-*) os_tag="linux" ;;
    win-*) os_tag="windows-msvc${LLVM_WINDOWS_MSVC_TOOLSET}" ;;
    *) echo "Unsupported host platform: ${host_platform}" >&2; exit 2 ;;
esac

case "${kind}" in
    modern)
        py_tag="${3:-}"
        [[ -n "${py_tag}" ]] || usage
        build_hash="$(hash_inputs \
            "${SCRIPT_DIR}/build-llvm-modern.sh" \
            "${SCRIPT_DIR}/../cext/mlir-python/CMakeLists.txt")"
        echo "llvm-modern-${os_tag}-$(cache_arch "${host_platform}")-${py_tag}-$(cache_version "${LLVM_MODERN_COMMIT}")-${build_hash}"
        ;;
    llvm7)
        [[ $# -eq 2 && "${host_platform}" != win-arm64 ]] || usage
        build_hash="$(hash_inputs \
            "${SCRIPT_DIR}/build-llvm7.sh" \
            "${SCRIPT_DIR}/tools/gen-llvm-c-exports.py")"
        echo "llvm7-${os_tag}-$(cache_arch "${host_platform}")-$(cache_version "${LLVM7_TAG}")-${build_hash}"
        ;;
    *)
        usage
        ;;
esac
