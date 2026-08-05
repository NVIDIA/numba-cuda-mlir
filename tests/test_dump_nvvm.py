# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from numba_cuda_mlir import cuda, mlir_optimization, types
from numba_cuda_mlir.numba_cuda import config


BITCODE = b"BC\xc0\xde\x00\x01\x02\x03"
TEXT_IR = b"; ModuleID = 'kernel'\ndefine void @foo() {\n  ret void\n}\n"


@pytest.fixture
def dump_nvvm(monkeypatch):
    def set_dump_target(value):
        monkeypatch.setattr(config, "CUDA_DUMP_NVVM", value)

    return set_dump_target


def test_dump_nvvm_disabled_is_noop(dump_nvvm, tmp_path, capsys):
    dump_nvvm("")
    mlir_optimization._maybe_dump_nvvm(BITCODE)
    assert capsys.readouterr().err == ""
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize(
    "value,contents",
    [
        pytest.param("stderr", TEXT_IR.decode(), id="text"),
        pytest.param("1", f"bitcode, {len(BITCODE)} bytes", id="bitcode"),
    ],
)
def test_dump_nvvm_to_stderr(dump_nvvm, capsys, value, contents):
    dump_nvvm(value)
    mlir_optimization._maybe_dump_nvvm(TEXT_IR if value == "stderr" else BITCODE)
    assert contents in capsys.readouterr().err


@pytest.mark.parametrize(
    "contents,suffix",
    [
        pytest.param(BITCODE, ".bc", id="bitcode"),
        pytest.param(TEXT_IR, ".ll", id="text"),
    ],
)
def test_dump_nvvm_to_directory(dump_nvvm, tmp_path, contents, suffix):
    dump_nvvm(str(tmp_path))
    mlir_optimization._maybe_dump_nvvm(contents)
    dumps = list(tmp_path.iterdir())
    assert len(dumps) == 1
    assert dumps[0].suffix == suffix
    assert dumps[0].read_bytes() == contents


def test_dump_nvvm_to_explicit_file(dump_nvvm, tmp_path):
    target = tmp_path / "nested" / "out.bc"
    dump_nvvm(str(target))
    mlir_optimization._maybe_dump_nvvm(BITCODE)
    assert target.read_bytes() == BITCODE


def test_failed_llvm70_translation_dumps_nvvm_input(dump_nvvm, tmp_path, monkeypatch):
    dump_nvvm(str(tmp_path))
    monkeypatch.setattr(mlir_optimization, "_get_libnvvm_path", lambda: b"/missing/libnvvm.so")

    @cuda.jit(device=True, chip="sm_90")
    def foo(value):
        return value + 1

    with pytest.raises(RuntimeError, match="llvm70 translation failed"):
        foo.compile((types.int32,))

    dumps = list(tmp_path.iterdir())
    assert len(dumps) == 1
    assert dumps[0].suffix == ".bc"
    assert dumps[0].read_bytes().startswith(mlir_optimization._BITCODE_MAGIC)


@pytest.mark.parametrize("chip", ["sm_90", "sm_100"])
@pytest.mark.parametrize("lto", [False, True], ids=["ptx", "lto"])
def test_compilation_dumps_exact_nvvm_input(dump_nvvm, tmp_path, chip, lto):
    dump_nvvm(str(tmp_path))

    @cuda.jit(device=True, chip=chip, lto=lto)
    def foo(value):
        return value + 1

    foo.compile((types.int32,))

    dumps = list(tmp_path.iterdir())
    assert len(dumps) == 1
    contents = dumps[0].read_bytes()
    is_bitcode = contents.startswith(mlir_optimization._BITCODE_MAGIC)
    assert dumps[0].suffix == (".bc" if is_bitcode else ".ll")
    if not is_bitcode:
        assert b"target triple" in contents
