# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for MLIR JIT LTO option defaults."""

import pytest

from numba_cuda_mlir import cuda
from numba_cuda_mlir.linker import Linker, resolve_link_plan


def _kernel():
    pass


def test_mlir_jit_rejects_output_option():
    with pytest.raises(ValueError, match="Got invalid options: output"):
        cuda.jit(output="ltoir")(_kernel)


def test_mlir_jit_defaults_to_ptx_linking():
    dispatcher = cuda.jit(_kernel)

    assert dispatcher.targetoptions["lto"] is False
    assert dispatcher.targetoptions["_lto_explicit"] is False


def test_link_plan_keeps_implicit_ptx_for_external_ptx_link_items():
    dispatcher = cuda.jit(link=["external.ptx"])(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is False
    assert plan.lto_explicit is False


def test_link_plan_enables_implicit_lto_for_cu_source_link_items():
    dispatcher = cuda.jit(link=[cuda.CUSource("")])(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is True
    assert plan.lto_explicit is False


def test_link_plan_ptx_disables_implicit_lto_for_cu_source_link_items():
    dispatcher = cuda.jit(link=[cuda.CUSource(""), "external.ptx"])(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is False


def test_link_plan_ptx_disables_implicit_lto_for_ltoir_link_items():
    dispatcher = cuda.jit(link=["external.ltoir", "external.ptx"])(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is False


def test_link_plan_enables_implicit_lto_for_ltoir_link_items():
    dispatcher = cuda.jit(link=["external.ltoir"])(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is True
    assert plan.lto_explicit is False
    assert plan.has_ltoir_link_items is True


def test_link_plan_preserves_explicit_lto_false_without_ltoir():
    dispatcher = cuda.jit(link=["external.ptx"], lto=False)(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is False
    assert plan.lto_explicit is True
    assert plan.requested_lto is False


def test_link_plan_preserves_explicit_lto_true_with_ptx():
    dispatcher = cuda.jit(link=["external.ptx"], lto=True)(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is True
    assert plan.lto_explicit is True


def test_link_plan_rejects_explicit_lto_false_with_ltoir():
    dispatcher = cuda.jit(link=["external.ltoir"], lto=False)(_kernel)

    with pytest.raises(ValueError, match="Cannot link LTOIR inputs with lto=False"):
        resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])


def test_link_plan_allows_debug_ptx_with_implicit_ltoir_input():
    dispatcher = cuda.jit(link=["external.ltoir"], debug=True, opt=False)(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is False
    assert plan.has_ltoir_link_items is True


def test_cu_source_uses_resolved_lto_mode_after_lto_enabled():
    cu_source = cuda.CUSource("")
    linker = Linker(cc=(8, 0), arch="sm_80", lto=True)

    linker.add_file_guess_ext(cu_source, compile_cu_as_ltoir=False)

    assert linker.lto is True
    assert linker._pending_cu == [("linkable", cu_source, cu_source.name, False)]


def test_link_plan_lineinfo_preserves_implicit_lto_for_ltoir_inputs():
    dispatcher = cuda.jit(link=["external.ltoir"], lineinfo=True)(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is True


def test_link_plan_callbacks_do_not_disable_implicit_lto():
    def setup(_):
        pass

    callback_item = cuda.CUSource("", setup_callback=setup)
    dispatcher = cuda.jit(link=[callback_item])(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is True
    assert plan.has_callback_link_items is True


def test_link_plan_explicit_lto_preserved_with_callbacks():
    def setup(_):
        pass

    callback_item = cuda.CUSource("", setup_callback=setup)
    dispatcher = cuda.jit(link=[callback_item], lto=True)(_kernel)

    plan = resolve_link_plan(dispatcher.targetoptions, dispatcher.targetoptions["link"])

    assert plan.compile_new_inputs_as_ltoir is True
    assert plan.lto_explicit is True
