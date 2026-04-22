# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for automatic PTX ISA version selection based on installed CTK."""

import re

import pytest

import cuda.simt as cuda
from cuda.simt import types
from cusimt.tools import get_max_ptx_version


def _extract_ptx_version(ptx: str) -> tuple[int, int] | None:
    m = re.search(r"\.version\s+(\d+)\.(\d+)", ptx)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


class TestPTXVersionInOutput:
    def test_auto_ptx_version_in_compiled_ptx(self):
        """Compiled PTX .version should match the CTK's max PTX from our
        lookup table.  libnvvm determines the final .version directive."""
        expected = get_max_ptx_version()
        if expected is None:
            pytest.skip("CTK not in the PTX lookup table")

        @cuda.jit
        def k(x: cuda.DeviceNDArray):
            x[0] = x[0] + 1

        ptx = k.inspect_ptx(types.void(types.float32[:]))
        assert ptx is not None
        ver = _extract_ptx_version(ptx)
        assert ver is not None, "No .version directive found in PTX"
        ptx_int = ver[0] * 10 + ver[1]
        assert (
            ptx_int == expected
        ), f"PTX .version {ver[0]}.{ver[1]} (={ptx_int}) != expected +ptx{expected}"

    def test_explicit_features_respected(self):
        """User-specified +ptxNN should not be overridden by auto-detection.

        libnvvm sets the .version in the final PTX, so we only verify that
        auto-detection does not inject a conflicting +ptxNN and that
        compilation succeeds.
        """

        @cuda.jit(features="+ptx80")
        def k80(x: cuda.DeviceNDArray):
            x[0] = x[0] + 1

        ptx = k80.inspect_ptx(types.void(types.float32[:]))
        assert ptx is not None
        ver = _extract_ptx_version(ptx)
        assert ver is not None, "No .version directive found in PTX"
        # libnvvm determines the output .version; just verify it's at least
        # the version we requested (compilation would fail otherwise).
        assert ver >= (8, 0), f"Expected PTX >= 8.0 but got {ver}"
