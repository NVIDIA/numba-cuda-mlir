# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIR JIT LTO option defaults."""

import types

from numba_cuda_mlir import cuda
from numba_cuda_mlir.caching import _targetoptions_cache_key
from numba_cuda_mlir.mlir_lowering import MLIRLower


def _kernel():
    pass


def _set_nvjitlink_available(monkeypatch, available):
    from numba_cuda_mlir.numba_cuda.cudadrv import driver

    monkeypatch.setattr(driver, "_have_nvjitlink", lambda: available)


def test_mlir_jit_defaults_to_ptx_linking():
    dispatcher = cuda.jit(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False
    assert dispatcher.targetoptions["_output_explicit"] is False


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
    assert dispatcher.targetoptions["_output_explicit"] is True


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
    assert dispatcher.targetoptions["_output_explicit"] is True


def test_mlir_jit_explicit_output_ltoir_takes_precedence_over_callback_link_items():
    dispatcher = cuda.jit(output="ltoir", link=[_CallbackLinkItem()])(_kernel)

    assert dispatcher.targetoptions["lto"] is True
    assert dispatcher.targetoptions["output"] == "ltoir"
    assert dispatcher.targetoptions["_lto_explicit"] is False
    assert dispatcher.targetoptions["_output_explicit"] is True


class _RecordingLinker:
    def __init__(self):
        self.link_items = []

    def add_file_guess_ext(self, link_item):
        self.link_items.append(link_item)


class _CallbackLinkItem:
    def setup_callback(self, *args):
        return None


def _fake_lower_for_linking(targetoptions):
    lower = MLIRLower.__new__(MLIRLower)
    lower.targetoptions = targetoptions
    lower._linker_config = {"lto": targetoptions.get("lto", False)}
    lower._linked_external_items = set()
    lower._linked_external_link_items = []
    lower._setup_callbacks = []
    lower._teardown_callbacks = []
    linkers = [_RecordingLinker()]
    lower.linker = linkers[0]

    def create_linker(self):
        linkers.append(_RecordingLinker())
        return linkers[-1]

    lower._create_linker = types.MethodType(create_linker, lower)
    return lower, linkers


def test_lowering_enables_implicit_lto_for_discovered_external_link_item(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    lower, linkers = _fake_lower_for_linking(
        {
            "lto": False,
            "debug": False,
            "lineinfo": False,
            "_lto_explicit": False,
            "_output_explicit": False,
        }
    )

    lower.link_external_item("external.cu")

    assert lower.targetoptions["lto"] is True
    assert lower._linker_config["lto"] is True
    assert linkers[-1].link_items == ["external.cu"]


def test_lowering_preserves_explicit_lto_false_for_discovered_external_link_item(
    monkeypatch,
):
    _set_nvjitlink_available(monkeypatch, True)
    lower, linkers = _fake_lower_for_linking(
        {
            "lto": False,
            "debug": False,
            "lineinfo": False,
            "_lto_explicit": True,
            "_output_explicit": False,
        }
    )

    lower.link_external_item("external.cu")

    assert lower.targetoptions["lto"] is False
    assert lower._linker_config["lto"] is False
    assert linkers[-1].link_items == ["external.cu"]


def test_lowering_preserves_explicit_output_ptx_for_discovered_external_link_item(
    monkeypatch,
):
    _set_nvjitlink_available(monkeypatch, True)
    lower, linkers = _fake_lower_for_linking(
        {
            "lto": False,
            "debug": False,
            "lineinfo": False,
            "_lto_explicit": False,
            "_output_explicit": True,
        }
    )

    lower.link_external_item("external.cu")

    assert lower.targetoptions["lto"] is False
    assert lower._linker_config["lto"] is False
    assert linkers[-1].link_items == ["external.cu"]


def test_lowering_preserves_explicit_output_ltoir_for_callback_link_item(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    callback_item = _CallbackLinkItem()
    lower, linkers = _fake_lower_for_linking(
        {
            "lto": True,
            "output": "ltoir",
            "debug": False,
            "lineinfo": False,
            "_lto_explicit": False,
            "_output_explicit": True,
        }
    )

    lower.link_external_item(callback_item)

    assert lower.targetoptions["lto"] is True
    assert lower.targetoptions["output"] == "ltoir"
    assert lower._linker_config["lto"] is True
    assert linkers[-1].link_items == [callback_item]


def test_lowering_preserves_callback_ptx_for_later_external_link_item(monkeypatch):
    _set_nvjitlink_available(monkeypatch, True)
    callback_item = _CallbackLinkItem()
    lower, linkers = _fake_lower_for_linking(
        {
            "lto": False,
            "debug": False,
            "lineinfo": False,
            "_lto_explicit": False,
            "_output_explicit": False,
        }
    )

    lower.link_external_item(callback_item)
    lower.link_external_item("external.cu")

    assert lower.targetoptions["lto"] is False
    assert lower.targetoptions["output"] == "ptx"
    assert lower._linker_config["lto"] is False
    assert linkers[-1].link_items == [callback_item, "external.cu"]


def test_cache_key_uses_lto_intent_for_implicit_lto():
    targetoptions = {
        "lto": False,
        "output": "ptx",
        "_lto_explicit": False,
        "_output_explicit": False,
    }
    key_before_lowering = _targetoptions_cache_key(targetoptions)

    targetoptions["lto"] = True

    assert _targetoptions_cache_key(targetoptions) == key_before_lowering


def test_cache_key_keeps_explicit_lto_false_distinct():
    explicit_false = {
        "lto": False,
        "output": "ptx",
        "_lto_explicit": True,
        "_output_explicit": False,
    }
    implicit = {
        "lto": False,
        "output": "ptx",
        "_lto_explicit": False,
        "_output_explicit": False,
    }

    assert _targetoptions_cache_key(explicit_false) != _targetoptions_cache_key(implicit)
