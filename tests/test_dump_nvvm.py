# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from numba_cuda_mlir import mlir_optimization
from numba_cuda_mlir.numba_cuda import config

BITCODE = b"BC\xc0\xde\x00\x01\x02\x03"
TEXT_IR = b"; ModuleID = 'kernel'\ndefine void @foo() {\n  ret void\n}\n"


@pytest.fixture
def dump_nvvm(monkeypatch):
    """Set config.CUDA_DUMP_NVVM for the duration of a test."""

    def _set(value):
        monkeypatch.setattr(config, "CUDA_DUMP_NVVM", value)

    return _set


def test_dump_nvvm_disabled_is_noop(dump_nvvm, tmp_path, capsys):
    dump_nvvm("")
    mlir_optimization._maybe_dump_nvvm(BITCODE)
    assert capsys.readouterr().err == ""
    assert list(tmp_path.iterdir()) == []


def test_dump_nvvm_text_to_stderr(dump_nvvm, capsys):
    dump_nvvm("stderr")
    mlir_optimization._maybe_dump_nvvm(TEXT_IR)
    err = capsys.readouterr().err
    assert "NVVM IR" in err
    assert "define void @foo()" in err


def test_dump_nvvm_bitcode_to_stderr_is_summarized(dump_nvvm, capsys):
    dump_nvvm("1")
    mlir_optimization._maybe_dump_nvvm(BITCODE)
    err = capsys.readouterr().err
    assert "bitcode" in err
    assert f"{len(BITCODE)} bytes" in err


def test_dump_nvvm_bitcode_to_directory(dump_nvvm, tmp_path):
    dump_nvvm(str(tmp_path))
    mlir_optimization._maybe_dump_nvvm(BITCODE)
    dumps = list(tmp_path.iterdir())
    assert len(dumps) == 1
    assert dumps[0].suffix == ".bc"
    assert dumps[0].read_bytes() == BITCODE


def test_dump_nvvm_text_to_directory(dump_nvvm, tmp_path):
    dump_nvvm(str(tmp_path))
    mlir_optimization._maybe_dump_nvvm(TEXT_IR)
    dumps = list(tmp_path.iterdir())
    assert len(dumps) == 1
    assert dumps[0].suffix == ".ll"
    assert dumps[0].read_bytes() == TEXT_IR


def test_dump_nvvm_to_explicit_file(dump_nvvm, tmp_path):
    target = tmp_path / "nested" / "out.bc"
    dump_nvvm(str(target))
    mlir_optimization._maybe_dump_nvvm(BITCODE)
    assert target.read_bytes() == BITCODE
