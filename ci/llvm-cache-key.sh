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

hash_file() {
    sha256sum "$1" | cut -c1-12
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
        *) echo "Unsupported host platform: $1" >&2; exit 2 ;;
    esac
}

kind="${1:-}"
host_platform="${2:-}"

case "${kind}" in
    modern)
        py_tag="${3:-}"
        [[ -n "${host_platform}" && -n "${py_tag}" ]] || usage
        version_short="$(cache_version "${LLVM_MODERN_COMMIT}")"
        case "${host_platform}" in
            linux-*)
                build_hash="$(hash_inputs \
                    "${SCRIPT_DIR}/build-llvm-modern.sh" \
                    "${SCRIPT_DIR}/patch-nanobind-2.13.py")"
                echo "llvm-modern-linux-$(cache_arch "${host_platform}")-${py_tag}-${version_short}-${build_hash}"
                ;;
            win-64)
                build_hash="$(hash_inputs \
                    "${SCRIPT_DIR}/build-windows.sh" \
                    "${SCRIPT_DIR}/../cext/mlir-modern/CMakeLists.txt" \
                    "${SCRIPT_DIR}/../cext/mlir-modern/ModernBridge.cpp" \
                    "${SCRIPT_DIR}/../cext/mlir-modern/ModernBridgeSmoke.cpp" \
                    "${SCRIPT_DIR}/../cext/mlir-modern/include/ModernBridge.h" \
                    "${SCRIPT_DIR}/patch-nanobind-2.13.py" \
                    "${SCRIPT_DIR}/windows-llvm-container-build.ps1" \
                    "${SCRIPT_DIR}/windows-devcontainer.env")"
                echo "llvm-modern-windows-$(cache_arch "${host_platform}")-${py_tag}-${version_short}-${build_hash}"
                ;;
            *)
                echo "Unsupported host platform for modern LLVM: ${host_platform}" >&2
                exit 2
                ;;
        esac
        ;;
    llvm7)
        [[ -n "${host_platform}" && $# -eq 2 ]] || usage
        version_short="$(cache_version "${LLVM7_TAG}")"
        case "${host_platform}" in
            linux-*)
                build_hash="$(hash_file "${SCRIPT_DIR}/build-llvm7.sh")"
                echo "llvm7-linux-$(cache_arch "${host_platform}")-${version_short}-${build_hash}"
                ;;
            win-64)
                build_hash="$(hash_inputs \
                    "${SCRIPT_DIR}/build-windows.sh" \
                    "${SCRIPT_DIR}/tools/gen-llvm-c-exports.py" \
                    "${SCRIPT_DIR}/windows-llvm-container-build.ps1" \
                    "${SCRIPT_DIR}/windows-devcontainer.env")"
                echo "llvm7-windows-$(cache_arch "${host_platform}")-${version_short}-${build_hash}"
                ;;
            *)
                echo "Unsupported host platform for LLVM 7: ${host_platform}" >&2
                exit 2
                ;;
        esac
        ;;
    *)
        usage
        ;;
esac
