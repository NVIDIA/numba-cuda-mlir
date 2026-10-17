#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Stage cuda-bindings and cuda-core wheels for a Python version that
cuda-python does not publish to PyPI yet, by pulling them from its CI artifacts.

TODO: delete this script, its workflow callers, and the
CUDA_PYTHON_ARTIFACTS_TOKEN secret once cuda-python ships cp315 wheels to PyPI,
which is expected alongside Python 3.15.0 itself.
See https://github.com/NVIDIA/cuda-python/issues/2163.

Inputs:  PY_VER, HOST_PLATFORM, GH_TOKEN (needs actions:read on cuda-python).
Output:  a wheelhouse directory, exported as CUDA_PYTHON_WHEELHOUSE for
         run-tests to feed to pip as --find-links.

Uses only the standard library: the test runners are bare containers without
the GitHub CLI, jq, or unzip.
"""

from __future__ import annotations

import datetime
import fnmatch
import io
import json
import os
import shutil
import sys
import time
import typing
import urllib.error
import urllib.request
import zipfile

REPO = "NVIDIA/cuda-python"
WORKFLOW_NAME = "CI"
API = "https://api.github.com"
MAX_RUN_AGE_DAYS = 14
RUN_QUERY_ATTEMPTS = 4
RUN_QUERY_RETRY_SECONDS = 5


def fail(msg: str) -> typing.NoReturn:
    print(f"::error::{msg}")
    sys.exit(1)


class _StripAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """Artifact downloads redirect to blob storage, which rejects our header."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None:
            new.remove_header("Authorization")
        return new


_opener = urllib.request.build_opener(_StripAuthOnRedirect)


def get(url: str, token: str) -> bytes:
    # Guard the scheme before opening: this carries a bearer token, so it must
    # never follow a `file:` or otherwise unexpected URL.
    if not url.startswith("https://"):
        fail(f"refusing to fetch non-https URL: {url}")
    req = urllib.request.Request(  # noqa: S310  (scheme checked above)
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "numba-cuda-mlir-ci",
        },
    )
    try:
        with _opener.open(req) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        fail(f"GET {url} failed: HTTP {exc.code} {exc.reason}. {detail}")


def latest_successful_run(token: str) -> dict:
    # Scope the query to the CI workflow rather than filtering the repo-wide run
    # feed: that feed is mostly other workflows, and a stale read of it has
    # handed back a month-old run before.
    workflows = json.loads(get(f"{API}/repos/{REPO}/actions/workflows?per_page=100", token))
    ids = [w["id"] for w in workflows.get("workflows", []) if w.get("name") == WORKFLOW_NAME]
    if not ids:
        fail(f"no workflow named {WORKFLOW_NAME!r} in {REPO}")

    url = (
        f"{API}/repos/{REPO}/actions/workflows/{ids[0]}/runs?branch=main&status=success&per_page=20"
    )

    # The API occasionally serves a stale replica listing a run that is weeks
    # old. Those wheels are built against an older CTK than our matrix targets,
    # which would otherwise surface as a confusing failure much later, so
    # re-query instead of accepting them.
    run = None
    for attempt in range(1, RUN_QUERY_ATTEMPTS + 1):
        runs = json.loads(get(url, token)).get("workflow_runs", [])
        if not runs:
            fail(f"found no successful {WORKFLOW_NAME} run on {REPO} main")
        run = max(runs, key=lambda r: r["created_at"])
        age = datetime.datetime.now(datetime.timezone.utc) - datetime.datetime.fromisoformat(
            run["created_at"].replace("Z", "+00:00")
        )
        if age <= datetime.timedelta(days=MAX_RUN_AGE_DAYS):
            return run
        print(
            f"::warning::attempt {attempt}/{RUN_QUERY_ATTEMPTS}: newest successful "
            f"{WORKFLOW_NAME} run came back {age.days} days old (run {run['id']}); "
            "looks like a stale read, re-querying"
        )
        if attempt < RUN_QUERY_ATTEMPTS:
            time.sleep(RUN_QUERY_RETRY_SECONDS)

    fail(
        f"newest successful {WORKFLOW_NAME} run on {REPO} main is still older than "
        f"{MAX_RUN_AGE_DAYS} days after {RUN_QUERY_ATTEMPTS} attempts (run {run['id']}). "
        "Refusing to stage wheels that predate our tested CTK."
    )


def list_artifacts(run_id: int, token: str) -> list[dict]:
    out, page = [], 1
    while True:
        url = f"{API}/repos/{REPO}/actions/runs/{run_id}/artifacts?per_page=100&page={page}"
        batch = json.loads(get(url, token)).get("artifacts", [])
        out.extend(batch)
        if len(batch) < 100:
            return out
        page += 1


def main() -> None:
    py_ver = os.environ.get("PY_VER") or fail("PY_VER must be set")
    host_platform = os.environ.get("HOST_PLATFORM") or fail("HOST_PLATFORM must be set")
    token = os.environ.get("GH_TOKEN")
    if not token:
        fail(
            "CUDA_PYTHON_ARTIFACTS_TOKEN is empty, so the cp315 wheels cannot be fetched. "
            f"Python 3.15 jobs need a token with actions:read on {REPO} until cuda-python "
            "publishes cp315 wheels to PyPI. See ci/tools/fetch_cuda_python_wheels.py."
        )

    if host_platform not in ("linux-64", "linux-aarch64", "win-64"):
        fail(f"unsupported host platform for artifact fetch: {host_platform}")

    # cuda-python names artifacts by interpreter tag: 3.15 -> python315,
    # 3.15t -> python315t.
    py_tag = "python" + py_ver.replace(".", "")

    run = latest_successful_run(token)
    print(f"Using cuda-python run https://github.com/{REPO}/actions/runs/{run['id']}")

    artifacts = list_artifacts(run["id"], token)

    # The bindings artifact name embeds the CTK it was built against, which
    # follows cuda-python's main rather than our test matrix, so match it with a
    # glob. Only CUDA 13 artifacts are produced, which is why the 3.15 rows in
    # test-matrix.yml are CUDA 13 only.
    patterns = {
        "cuda_bindings": f"cuda-bindings-{py_tag}-cuda13.*-{host_platform}-*",
        "cuda_core": f"cuda-core-{py_tag}-{host_platform}-*",
    }

    wheelhouse = os.path.join(os.getcwd(), "cuda-python-wheelhouse")
    shutil.rmtree(wheelhouse, ignore_errors=True)
    os.makedirs(wheelhouse)

    for dist, pattern in patterns.items():
        matches = [
            a
            for a in artifacts
            if fnmatch.fnmatch(a["name"], pattern)
            # Sibling artifacts carry the test suite, not wheels.
            and not a["name"].endswith(("-tests", "-test-binaries"))
        ]
        if not matches:
            fail(f"no artifact matching {pattern!r} in cuda-python run {run['id']}")

        staged = []
        for art in matches:
            print(f"Downloading {art['name']}")
            blob = get(f"{API}/repos/{REPO}/actions/artifacts/{art['id']}/zip", token)
            with zipfile.ZipFile(io.BytesIO(blob)) as zf:
                for member in zf.namelist():
                    if member.endswith(".whl"):
                        target = os.path.join(wheelhouse, os.path.basename(member))
                        with zf.open(member) as src, open(target, "wb") as dst:
                            shutil.copyfileobj(src, dst)
                        staged.append(os.path.basename(member))
        if not staged:
            fail(f"artifacts for {dist} on {py_tag}/{host_platform} contained no .whl")
        for name in staged:
            print(f"  staged {name}")

    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as fh:
            fh.write(f"CUDA_PYTHON_WHEELHOUSE={wheelhouse}\n")


if __name__ == "__main__":
    main()
