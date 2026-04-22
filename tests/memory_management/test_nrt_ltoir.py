# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NRT LTOIR compilation and linking."""

import os
import pytest

from cusimt.memory_management import compile_nrt_ltoir, get_include, needs_nrt_linking
from cusimt.linker import Linker
from cusimt.tools import get_gpu_compute_capability


def test_compile_nrt_ltoir():
    """Test that NRT sources compile to LTOIR successfully."""
    cc = get_gpu_compute_capability(tuple)
    ltoir = compile_nrt_ltoir(cc)

    assert isinstance(ltoir, bytes)
    assert len(ltoir) > 0


def test_get_include():
    """Test that get_include returns a valid path."""
    include_path = get_include()
    assert os.path.isdir(include_path)
    assert os.path.exists(os.path.join(include_path, "nrt.cu"))
    assert os.path.exists(os.path.join(include_path, "nrt.cuh"))
    assert os.path.exists(os.path.join(include_path, "memsys.cuh"))


def test_needs_nrt_linking():
    """Test NRT function detection in PTX."""
    # PTX with NRT function call
    ptx_with_nrt = """
    .extern .func NRT_Allocate;
    call.uni NRT_incref, (%r1);
    """
    assert needs_nrt_linking(ptx_with_nrt)

    # PTX without NRT
    ptx_without_nrt = """
    .func foo() {
        ret;
    }
    """
    assert not needs_nrt_linking(ptx_without_nrt)


def test_nrt_ltoir_caching():
    """Test that NRT LTOIR compilation is cached."""
    cc = get_gpu_compute_capability(tuple)

    ltoir1 = compile_nrt_ltoir(cc)
    ltoir2 = compile_nrt_ltoir(cc)

    # Should be the exact same object due to caching
    assert ltoir1 is ltoir2


def test_nrt_linking_with_linker():
    """Test that NRT LTOIR can be added to linker."""
    linker = Linker(lto=True)

    # Simulate PTX with NRT function calls
    ptx_with_nrt = b"""
    .extern .func NRT_Allocate;
    .visible .entry test_kernel() {
        ret;
    }
    """

    # Check that NRT linking is needed
    assert needs_nrt_linking(ptx_with_nrt.decode())

    # Get NRT LTOIR for linker's CC and add it
    nrt_ltoir = compile_nrt_ltoir(linker.cc)
    linker.add_ltoir(nrt_ltoir)

    # Verify the linker has LTOIR
    assert linker.lto
    assert nrt_ltoir in linker._ltoirs.values()


def test_nrt_disk_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("CUSIMT_CACHE_DIR", str(tmp_path))
    compile_nrt_ltoir.cache_clear()

    cc = get_gpu_compute_capability(tuple)
    ltoir = compile_nrt_ltoir(cc)

    cached_files = list((tmp_path / "nrt").glob("*.ltoir"))
    assert len(cached_files) == 1
    assert cached_files[0].read_bytes() == ltoir


def test_nrt_disk_cache_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("CUSIMT_CACHE_DIR", str(tmp_path))

    cache_dir = tmp_path / "nrt"
    cache_dir.mkdir(parents=True)
    (cache_dir / "fake.ltoir").write_bytes(b"fake")

    monkeypatch.setenv("CUSIMT_DISABLE_CACHE", "1")
    compile_nrt_ltoir.cache_clear()

    cc = get_gpu_compute_capability(tuple)
    compile_nrt_ltoir(cc)

    assert not cache_dir.exists()
