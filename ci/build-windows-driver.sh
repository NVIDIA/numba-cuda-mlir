#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODE="${1:-}"

case "${MODE}" in
  modern|llvm7) ;;
  *)
    echo "Usage: $0 [modern|llvm7]" >&2
    exit 2
    ;;
esac

cd "${REPO_ROOT}"
chmod +x ci/*.sh || true
ci/build-windows.sh "${MODE}"
