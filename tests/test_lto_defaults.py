# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIR JIT LTO option defaults."""

from numba_cuda_mlir import cuda


def _kernel():
    pass


def _set_nvjitlink_available(monkeypatch, available):
    from numba_cuda_mlir.numba_cuda.cudadrv import driver

    monkeypatch.setattr(driver, "_have_nvjitlink", lambda: available)


def test_mlir_jit_defaults_to_ptx_linking():
    dispatcher = cuda.jit(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_mlir_jit_enables_implicit_lto_for_external_link_items(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)

    dispatcher = cuda.jit(link=["external.ltoir"])(_kernel)

    assert dispatcher.targetoptions["lto"] is True
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_mlir_jit_disables_implicit_lto_without_nvjitlink(monkeypatch):
    _set_nvjitlink_available(monkeypatch, False)

    dispatcher = cuda.jit(link=["external.ltoir"])(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_mlir_jit_preserves_explicit_lto_false_with_external_link_items():
    dispatcher = cuda.jit(link=["external.ltoir"], lto=False)(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is True


def test_mlir_jit_disables_implicit_lto_for_debug_external_link_items(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)

    # debug=True requires opt=False so verify_target_options resolves opt_level=0.
    dispatcher = cuda.jit(link=["external.ltoir"], debug=True, opt=False)(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_mlir_jit_disables_implicit_lto_for_lineinfo_external_link_items(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)

    dispatcher = cuda.jit(link=["external.ltoir"], lineinfo=True)(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_mlir_jit_explicit_output_ptx_takes_precedence_over_link_items(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)

    dispatcher = cuda.jit(link=["external.ltoir"], output="ptx")(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_mlir_jit_preserves_explicit_lto_true():
    dispatcher = cuda.jit(lto=True)(_kernel)

    assert dispatcher.targetoptions["lto"] is True
    assert dispatcher.targetoptions["_lto_explicit"] is True


def test_mlir_jit_disables_implicit_lto_for_callback_link_items():
    class LinkItem:
        def __init__(self):
            self.setup_callback = lambda: None
            self.teardown_callback = None

    dispatcher = cuda.jit(link=[LinkItem()])(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_mlir_jit_output_ltoir_enables_lto():
    dispatcher = cuda.jit(output="ltoir")(_kernel)

    assert dispatcher.targetoptions["lto"] is True
    assert dispatcher.targetoptions["_lto_explicit"] is False
