#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Configure sccache with RAPIDS S3 backend.
# Assumes sccache is already installed (done in CI before_script).
#
# Based on Paul Taylor's gist:
#   https://gist.github.com/trxcllnt/eaab5d814dd3069ec2103a7cccabf5d1
#
# Requires: GH_PAT env var (GitHub PAT with read:org + read:enterprise scopes)
set -euo pipefail

: "${GH_PAT:?Must set GH_PAT (GitHub PAT with read:org + read:enterprise scopes)}"

ARCH=$(uname -m)

# Use PAT for all GitHub API calls to avoid rate limits
_gh_api_header="Authorization: token ${GH_PAT}"

# --- Install GitHub CLI ---
if ! command -v gh &>/dev/null; then
    echo ">>> Installing GitHub CLI"
    GH_ARCH=$(echo "$ARCH" | sed -e 's/x86_64/amd64/' -e 's/aarch64/arm64/')
    GH_VERSION=$(curl -fsSL -H "${_gh_api_header}" https://api.github.com/repos/cli/cli/releases/latest \
        | python3 -c "import sys,json; print(json.load(sys.stdin)['tag_name'].lstrip('v'))")
    curl -fsSL "https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_${GH_ARCH}.tar.gz" \
        | tar -C /usr/local -xz --strip-components=1
fi

# --- Install gh-nv-gha-aws plugin ---
GH_AWS_DIR="${HOME}/.local/share/gh/extensions/gh-nv-gha-aws"
if [ ! -x "${GH_AWS_DIR}/gh-nv-gha-aws" ]; then
    echo ">>> Installing gh-nv-gha-aws plugin"
    mkdir -p "${GH_AWS_DIR}"
    DEB_ARCH=$(echo "$ARCH" | sed -e 's/x86_64/amd64/' -e 's/aarch64/arm64/')
    curl -fsSL -o "${GH_AWS_DIR}/gh-nv-gha-aws" \
        "$(curl -fsSL -H "${_gh_api_header}" https://api.github.com/repos/nv-gha-runners/gh-nv-gha-aws/releases/latest \
           | python3 -c "import sys,json; r=json.load(sys.stdin); print([a['browser_download_url'] for a in r['assets'] if 'linux-${DEB_ARCH}' in a['name']][0])")"
    chmod +x "${GH_AWS_DIR}/gh-nv-gha-aws"
    cat <<EOF > "${GH_AWS_DIR}/manifest.yml"
owner: nv-gha-runners
name: gh-nv-gha-aws
host: github.com
tag: $("${GH_AWS_DIR}/gh-nv-gha-aws" --version | cut -d' ' -f3)
ispinned: false
path: ${GH_AWS_DIR}/gh-nv-gha-aws
EOF
fi

# --- Authenticate with GitHub ---
# PAT must already have read:org + read:enterprise scopes
echo ">>> Authenticating with GitHub"
gh auth login --with-token <<< "${GH_PAT}"

# --- Generate temporary S3 credentials (12h TTL) ---
echo ">>> Generating S3 credentials via gh-nv-gha-aws"
mkdir -p ~/.aws
gh nv-gha-aws org nvidia \
    --duration 43200 \
    --profile default \
    --output creds-file \
    --aud sts.amazonaws.com \
    --idp-url https://token.gha-runners.nvidia.com \
    --role-arn arn:aws:iam::279114543810:role/nv-gha-token-sccache-devs \
    > ~/.aws/credentials

# --- Configure sccache (S3 only, no disk cache in CI) ---
echo ">>> Configuring sccache"
mkdir -p ~/.config/sccache
cat <<EOF > ~/.config/sccache/config
[cache.s3]
bucket = "rapids-sccache-devs"
region = "us-east-2"
key_prefix = "cusimt"

[cache.s3.preprocessor_cache_mode]
use_preprocessor_cache_mode = true
key_prefix = "cusimt/preprocessor"
EOF

# --- Start sccache daemon ---
export SCCACHE_IDLE_TIMEOUT=0
export SCCACHE_ERROR_LOG=/tmp/sccache.log
sccache --start-server
sccache --show-stats

echo ">>> sccache ready"
