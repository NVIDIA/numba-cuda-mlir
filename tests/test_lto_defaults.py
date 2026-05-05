# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba_cuda_mlir.decorators import mlir_jit, verify_target_options
from numba_cuda_mlir.numba_cuda.cudadrv.linkable_code import CUSource
from numba_cuda_mlir.numba_cuda.core import config
from numba_cuda_mlir.numba_cuda.cudadrv import driver


def test_target_options_do_not_resolve_default_chip(monkeypatch):
    import numba_cuda_mlir.tools as tools

    def fail_compute_capability(*args):
        raise AssertionError("GPU compute capability should be resolved at compile time")

    monkeypatch.setattr(tools, "get_gpu_compute_capability", fail_compute_capability)
    monkeypatch.setattr(driver, "_have_nvjitlink", lambda: True)

    targetoptions = verify_target_options({})

    assert targetoptions["chip"] is None


def test_explicit_cc_maps_to_chip():
    targetoptions = verify_target_options({"cc": (8, 0)})

    assert targetoptions["chip"] == "sm_80"


def test_mlir_jit_allows_explicit_opt_level_with_default_opt_config(monkeypatch):
    monkeypatch.setattr(config, "OPT", 1)

    def kernel(a):
        return None

    dispatcher = mlir_jit(kernel, opt_level=3)

    assert dispatcher.targetoptions["opt"] is None
    assert dispatcher.targetoptions["opt_level"] == 3


def test_lto_defaults_to_ltoir_when_nvjitlink_is_available(monkeypatch):
    monkeypatch.setattr(driver, "_have_nvjitlink", lambda: True)

    targetoptions = verify_target_options({})

    assert targetoptions["lto"] is True
    assert targetoptions["output"] == "ltoir"


def test_lto_defaults_to_ptx_when_nvjitlink_is_unavailable(monkeypatch):
    monkeypatch.setattr(driver, "_have_nvjitlink", lambda: False)

    targetoptions = verify_target_options({})

    assert targetoptions["lto"] is False
    assert targetoptions["output"] == "ptx"


def test_lto_false_keeps_ptx_output_when_nvjitlink_is_available(monkeypatch):
    monkeypatch.setattr(driver, "_have_nvjitlink", lambda: True)

    targetoptions = verify_target_options({"lto": False})

    assert targetoptions["lto"] is False
    assert targetoptions["output"] == "ptx"


def test_explicit_ptx_output_keeps_lto_disabled_when_nvjitlink_is_available(monkeypatch):
    monkeypatch.setattr(driver, "_have_nvjitlink", lambda: True)

    targetoptions = verify_target_options({"output": "ptx"})

    assert targetoptions["lto"] is False
    assert targetoptions["output"] == "ptx"


def test_lineinfo_default_keeps_ptx_output_when_nvjitlink_is_available(monkeypatch):
    monkeypatch.setattr(driver, "_have_nvjitlink", lambda: True)

    targetoptions = verify_target_options({"lineinfo": True})

    assert targetoptions["lto"] is False
    assert targetoptions["output"] == "ptx"


def test_callback_link_default_keeps_ptx_output_when_nvjitlink_is_available(monkeypatch):
    monkeypatch.setattr(driver, "_have_nvjitlink", lambda: True)

    targetoptions = verify_target_options({"link": [CUSource("", setup_callback=lambda obj: None)]})

    assert targetoptions["lto"] is False
    assert targetoptions["output"] == "ptx"


def test_lto_default_is_disabled_for_debug_builds(monkeypatch):
    monkeypatch.setattr(driver, "_have_nvjitlink", lambda: True)

    targetoptions = verify_target_options({"debug": True, "opt": False})

    assert targetoptions["lto"] is False
    assert targetoptions["output"] == "ptx"


def test_lto_default_uses_resolved_debug_config(monkeypatch):
    monkeypatch.setattr(driver, "_have_nvjitlink", lambda: True)
    monkeypatch.setattr(config, "CUDA_DEBUGINFO_DEFAULT", 1)
    monkeypatch.setattr(config, "OPT", 0)

    def kernel(a):
        return None

    dispatcher = mlir_jit(kernel)

    assert dispatcher.targetoptions["debug"] is True
    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["output"] == "ptx"
