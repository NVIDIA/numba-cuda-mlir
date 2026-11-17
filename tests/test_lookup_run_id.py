# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "ci/tools/lookup-run-id"
SHA = "a" * 40
OLD_SHA = "b" * 40
BRANCH = "pull-request/469"


@pytest.fixture
def lookup(tmp_path):
    if not shutil.which("bash") or not shutil.which("jq"):
        pytest.skip("lookup-run-id requires bash and jq")
    gh = tmp_path / "gh"
    gh.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "data = json.loads(os.environ['LOOKUP_TEST_DATA'])\n"
        "repo = 'repos/NVIDIA/numba-cuda-mlir'\n"
        "if repo + '/actions/workflows/ci.yaml' in args:\n"
        "    print(123)\n"
        "elif repo + '/branches/' + data['branch'] in args:\n"
        "    print(data['head'])\n"
        "elif repo + '/commits' in args:\n"
        "    print('\\n'.join(data['commits']))\n"
        "elif repo + '/actions/workflows/123/runs' in args:\n"
        "    sha = next(a.split('=', 1)[1] for a in args if a.startswith('head_sha='))\n"
        "    print(json.dumps({'workflow_runs': data['runs'].get(sha, [])}))\n"
        "else:\n"
        "    sys.exit('unexpected gh call: ' + repr(args))\n"
    )
    gh.chmod(0o755)

    def invoke(runs, *, exact=True, allow_failure=False, branch=BRANCH):
        args = ["bash", str(SCRIPT), "--branch", branch, "--event", "push", "--head-sha"]
        if exact:
            args += ["--commit", SHA]
        if allow_failure:
            args += ["--allow-failure"]
        args += ["NVIDIA/numba-cuda-mlir", "ci.yaml"]
        env = dict(
            os.environ, PATH=f"{tmp_path}{os.pathsep}{os.environ['PATH']}", GH_TOKEN=tmp_path.name
        )
        env["LOOKUP_RUN_ID_COMMIT_SCAN_LIMIT"] = "2"
        env["LOOKUP_TEST_DATA"] = json.dumps(
            {"branch": branch, "head": SHA, "commits": [SHA, OLD_SHA], "runs": runs}
        )
        return subprocess.run(args, env=env, capture_output=True, text=True)

    return invoke


def producer(run_id=10, sha=SHA, branch=BRANCH):
    return dict(
        id=run_id,
        head_sha=sha,
        head_branch=branch,
        workflow_id=123,
        event="push",
        status="completed",
        conclusion="success",
        created_at=f"2026-11-17T00:00:{run_id:02d}Z",
    )


def test_manual_failure_requires_opt_in(lookup):
    run = producer()
    run["conclusion"] = "failure"
    runs = {SHA: [run]}
    assert lookup(runs).returncode != 0
    accepted = lookup(runs, allow_failure=True)
    assert accepted.returncode == 0, accepted.stderr
    assert accepted.stdout.splitlines() == ["10", SHA]


@pytest.mark.parametrize(
    "field,value",
    [
        ("conclusion", "cancelled"),
        ("conclusion", "timed_out"),
        ("conclusion", "neutral"),
        ("status", "in_progress"),
        ("workflow_id", 456),
        ("head_branch", "pull-request/461"),
        ("head_sha", OLD_SHA),
        ("event", "workflow_dispatch"),
    ],
)
def test_manual_rejects_ineligible_producer(lookup, field, value):
    run = producer()
    run[field] = value
    result = lookup({SHA: [run]}, allow_failure=True)
    assert result.returncode != 0
    assert "No completed qualifying" in result.stderr
    assert result.stdout == ""


def test_manual_does_not_substitute_ancestor_wheels(lookup):
    result = lookup({OLD_SHA: [producer(sha=OLD_SHA)]}, allow_failure=True)
    assert result.returncode != 0
    assert f"commit {SHA}" in result.stderr
    assert result.stdout == ""


def test_manual_selects_latest_eligible_run_at_dispatch_sha(lookup):
    failed = producer(11)
    failed["conclusion"] = "failure"
    cancelled = producer(12)
    cancelled["conclusion"] = "cancelled"
    result = lookup({SHA: [producer(), failed, cancelled]}, allow_failure=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["11", SHA]


def test_scheduled_lookup_retains_successful_ancestor_fallback(lookup):
    failed = producer(branch="main")
    failed["conclusion"] = "failure"
    result = lookup(
        {SHA: [failed], OLD_SHA: [producer(sha=OLD_SHA, branch="main")]},
        exact=False,
        branch="main",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["10", OLD_SHA]
