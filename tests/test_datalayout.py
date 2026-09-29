# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import re
from unittest import mock

from numba_cuda_mlir.lowering_utilities import llvm_utils
from numba_cuda_mlir.numba_cuda.cudadrv.nvvm import NVVM


def _datalayout_for_host(machine):
    """Return NVPTX64_DATALAYOUT as a fresh import of llvm_utils would define it.

    Executed into a throwaway namespace so the real module, which caches loaded
    CAPI handles at import time, is left untouched.
    """
    spec = importlib.util.spec_from_file_location("_llvm_utils_host_probe", llvm_utils.__file__)
    module = importlib.util.module_from_spec(spec)
    with mock.patch("platform.machine", return_value=machine):
        spec.loader.exec_module(module)
    return module.NVPTX64_DATALAYOUT


def test_datalayout_does_not_vary_by_host_platform():
    # The data layout describes the nvptx64 device ABI, so what the host reports
    # through platform.machine() must not influence it.
    layouts = {m: _datalayout_for_host(m) for m in ("x86_64", "aarch64", "AMD64", "ARM64")}
    assert set(layouts.values()) == {llvm_utils.NVPTX64_DATALAYOUT}, (
        f"data layout varies by host platform: {layouts}"
    )


def test_datalayout_declares_no_stack_alignment():
    # "S<n>" is a stack natural alignment, a host-ABI property. Device code has
    # no such requirement and libNVVM's target does not declare one, so
    # declaring it here is a mismatch libNVVM rejects.
    assert re.search(r"-S\d", llvm_utils.NVPTX64_DATALAYOUT) is None


def test_nvvm_reports_the_same_datalayout():
    # Read the getter off the class so this does not require libNVVM to be
    # installed; it ignores self.
    assert NVVM.data_layout.fget(None) == llvm_utils.NVPTX64_DATALAYOUT
