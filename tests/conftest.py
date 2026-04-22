# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
from typing import NamedTuple

from cuda.core import Device
from cuda.core._utils.cuda_utils import CUDAError
from cuda.pathfinder import find_nvidia_binary_utility
import pytest

from gpu_utils import check_cc_min, check_cc_exact
from numba_cuda_mlir.numba_cuda.core import config as cuda_config


def _run(cmd: list[str], cwd: Path) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace").strip()
        pytest.skip(f"nvcc compilation failed (exit {result.returncode}): {stderr}")


def _get_gpu_cc() -> str:
    try:
        arch = Device().arch
    except CUDAError as e:
        pytest.skip(f"Could not query CUDA device architecture: {e}")
    return _gpu_cc_from_arch(arch)


def _gpu_cc_from_arch(arch) -> str:
    if hasattr(arch, "major") and hasattr(arch, "minor"):
        return f"{arch.major}{arch.minor}"

    if isinstance(arch, tuple) and len(arch) >= 2:
        return f"{arch[0]}{arch[1]}"

    match = re.fullmatch(r"(?:sm_|compute_)?(\d+[a-z]?)", str(arch))
    if match:
        return match.group(1)

    match = re.fullmatch(r"(\d+)\.(\d+)", str(arch))
    if match:
        return "".join(match.groups())

    pytest.skip(f"Could not interpret CUDA device architecture: {arch!r}")


def _determine_cuda_include_flags(nvcc: str) -> list[str]:
    cp = subprocess.run([nvcc, "-v", "__dummy"], capture_output=True)
    stream = cp.stdout if platform.system() == "Windows" else cp.stderr
    lines = stream.decode(errors="replace").splitlines()
    includes_lines = [line for line in lines if line.startswith("#$ INCLUDES=")]
    if len(includes_lines) != 1:
        pytest.skip("Could not determine CUDA include paths from nvcc")

    quoted_flags = shlex.split(
        includes_lines[0].split("INCLUDES=", 1)[1].strip(), posix=False
    )
    include_flags = [flag.strip('"') for flag in quoted_flags]
    include_flags += [flag + os.path.sep + "cccl" for flag in include_flags]
    return include_flags


def _check_nvrtc(args):
    result, *rest = args
    value = result.value
    if value:
        from cuda.bindings import nvrtc

        error_string = _check_nvrtc(nvrtc.nvrtcGetErrorString(result)).decode()
        pytest.fail(f"NVRTC error, code {value}: {error_string}")
    if not rest:
        return None
    if len(rest) == 1:
        return rest[0]
    return tuple(rest)


def _write_raw_ltoir(
    source_path: Path, output_path: Path, arch: str, nvcc: str
) -> None:
    from cuda.bindings import nvrtc

    source = source_path.read_text()
    program = _check_nvrtc(
        nvrtc.nvrtcCreateProgram(source.encode(), source_path.name.encode(), 0, [], [])
    )
    options = [
        f"--gpu-architecture={arch}",
        "-dlto",
        "-rdc",
        "true",
        "-std=c++17",
        *_determine_cuda_include_flags(nvcc),
    ]
    encoded_options = [option.encode() for option in options]
    result = nvrtc.nvrtcCompileProgram(program, len(encoded_options), encoded_options)

    if result[0] == nvrtc.nvrtcResult.NVRTC_ERROR_COMPILATION:
        log_size = _check_nvrtc(nvrtc.nvrtcGetProgramLogSize(program))
        log = b" " * log_size
        _check_nvrtc(nvrtc.nvrtcGetProgramLog(program, log))
        pytest.fail(f"NVRTC compilation error:\n{log.decode()}")

    _check_nvrtc(result)
    ltoir_size = _check_nvrtc(nvrtc.nvrtcGetLTOIRSize(program))
    ltoir = b" " * ltoir_size
    _check_nvrtc(nvrtc.nvrtcGetLTOIR(program, ltoir))

    ltoir_magic = 0x7F4E43ED
    header = int.from_bytes(ltoir[:4], byteorder="little")
    if header != ltoir_magic:
        pytest.fail(
            f"Unexpected LTO-IR header 0x{header:X}; expected 0x{ltoir_magic:X}"
        )

    output_path.write_bytes(ltoir)


class NumbaCudaBinaryArtifact(NamedTuple):
    name: str
    filename: str
    path_key: str
    gencode_kind: str | None = None
    nvcc_output_flags: tuple[str, ...] = ()
    source_file: bool = False
    raw_ltoir: bool = False


_NVCC_FLAGS = ("-O3", "-rdc", "true", "-std=c++17")

_GENCODE_DEFAULT = "default"
_GENCODE_FATBIN = "fatbin"
_GENCODE_MULTI_FATBIN = "multi_fatbin"

_NUMBA_CUDA_BINARY_ARTIFACT_CONFIGS = (
    NumbaCudaBinaryArtifact(
        name="a",
        filename="test_device_functions.a",
        path_key="test_device_functions_a",
        gencode_kind=_GENCODE_DEFAULT,
        nvcc_output_flags=("-lib",),
    ),
    NumbaCudaBinaryArtifact(
        name="cubin",
        filename="test_device_functions.cubin",
        path_key="test_device_functions_cubin",
        gencode_kind=_GENCODE_DEFAULT,
        nvcc_output_flags=("--cubin",),
    ),
    NumbaCudaBinaryArtifact(
        name="cu",
        filename="test_device_functions.cu",
        path_key="test_device_functions_cu",
        source_file=True,
    ),
    NumbaCudaBinaryArtifact(
        name="fatbin",
        filename="test_device_functions.fatbin",
        path_key="test_device_functions_fatbin",
        gencode_kind=_GENCODE_FATBIN,
        nvcc_output_flags=("--fatbin",),
    ),
    NumbaCudaBinaryArtifact(
        name="fatbin_multi",
        filename="test_device_functions_multi.fatbin",
        path_key="test_device_functions_fatbin_multi",
        gencode_kind=_GENCODE_MULTI_FATBIN,
        nvcc_output_flags=("--fatbin",),
    ),
    NumbaCudaBinaryArtifact(
        name="o",
        filename="test_device_functions.o",
        path_key="test_device_functions_o",
        gencode_kind=_GENCODE_DEFAULT,
        nvcc_output_flags=("-dc",),
    ),
    NumbaCudaBinaryArtifact(
        name="ptx",
        filename="test_device_functions.ptx",
        path_key="test_device_functions_ptx",
        gencode_kind=_GENCODE_DEFAULT,
        nvcc_output_flags=("-ptx",),
    ),
    NumbaCudaBinaryArtifact(
        name="ltoir",
        filename="test_device_functions.ltoir",
        path_key="test_device_functions_ltoir",
        raw_ltoir=True,
    ),
)

_NUMBA_CUDA_BINARY_ARTIFACTS = {
    artifact.name: artifact for artifact in _NUMBA_CUDA_BINARY_ARTIFACT_CONFIGS
}


def _artifact_gencode_flags(
    artifact_config: NumbaCudaBinaryArtifact, gpu_cc: str
) -> list[str]:
    alt_cc = "80" if gpu_cc.startswith("7") else "75"

    default_gencode = ["-gencode", f"arch=compute_{gpu_cc},code=sm_{gpu_cc}"]
    multi_gencode = [
        "-gencode",
        f"arch=compute_{gpu_cc},code=[sm_{gpu_cc},lto_{gpu_cc}]",
    ]

    if artifact_config.gencode_kind is None:
        return []
    if artifact_config.gencode_kind == _GENCODE_DEFAULT:
        return default_gencode
    if artifact_config.gencode_kind == _GENCODE_FATBIN:
        return [
            *default_gencode,
            "-gencode",
            f"arch=compute_{alt_cc},code=sm_{alt_cc}",
        ]
    if artifact_config.gencode_kind == _GENCODE_MULTI_FATBIN:
        return [
            *multi_gencode,
            "-gencode",
            f"arch=compute_{alt_cc},code=[sm_{alt_cc},lto_{alt_cc}]",
        ]
    raise ValueError(
        f"Unknown Numba-CUDA artifact gencode: {artifact_config.gencode_kind}"
    )


class NumbaCudaTestBinaries:
    def __init__(self, paths: dict[str, str]) -> None:
        for artifact in _NUMBA_CUDA_BINARY_ARTIFACTS.values():
            setattr(self, artifact.path_key, None)
        for path_key, path in paths.items():
            setattr(self, path_key, path)

    @property
    def require_cuobjdump(self) -> tuple[str, ...]:
        return tuple(
            path
            for path in (
                self.test_device_functions_fatbin_multi,
                self.test_device_functions_fatbin,
                self.test_device_functions_o,
            )
            if path is not None
        )


class NumbaCudaTestBinaryBuilder:
    def __init__(self, root: Path) -> None:
        self.root = root
        source_dir = Path(__file__).parent / "numba_cuda_tests" / "data"
        self.source = root / "test_device_functions.cu"
        shutil.copy2(source_dir / "test_device_functions.cu", self.source)
        self._nvcc = None
        self._gpu_cc = None

    @property
    def nvcc(self) -> str:
        if self._nvcc is None:
            nvcc = find_nvidia_binary_utility("nvcc")
            if not nvcc:
                pytest.skip("nvcc is required to build Numba-CUDA test binaries")
            self._nvcc = nvcc
        return self._nvcc

    @property
    def gpu_cc(self) -> str:
        if self._gpu_cc is None:
            self._gpu_cc = _get_gpu_cc()
        return self._gpu_cc

    def build(self, artifact: str) -> str:
        if artifact not in _NUMBA_CUDA_BINARY_ARTIFACTS:
            known = ", ".join(sorted(_NUMBA_CUDA_BINARY_ARTIFACTS))
            raise ValueError(
                f"Unknown Numba-CUDA test binary artifact: {artifact}; "
                f"known artifacts: {known}"
            )

        artifact_config = _NUMBA_CUDA_BINARY_ARTIFACTS[artifact]
        output = self.root / artifact_config.filename
        if output.exists():
            return str(output)

        if artifact_config.source_file:
            return str(self.source)

        gpu_cc = self.gpu_cc

        if artifact_config.raw_ltoir:
            _write_raw_ltoir(self.source, output, f"sm_{gpu_cc}", self.nvcc)
            return str(output)

        output_flags = [
            *_artifact_gencode_flags(artifact_config, gpu_cc),
            *artifact_config.nvcc_output_flags,
        ]
        _run(
            [
                self.nvcc,
                *_NVCC_FLAGS,
                *output_flags,
                "-o",
                artifact_config.filename,
                self.source.name,
            ],
            cwd=self.root,
        )
        return str(output)

    def binaries(self, artifacts: tuple[str, ...]) -> NumbaCudaTestBinaries:
        paths = {}
        for artifact in artifacts:
            artifact_config = _NUMBA_CUDA_BINARY_ARTIFACTS[artifact]
            paths[artifact_config.path_key] = self.build(artifact)
        return NumbaCudaTestBinaries(paths)


@pytest.fixture(scope="session")
def numba_cuda_test_binary_builder(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("numba_cuda_test_binaries")
    return NumbaCudaTestBinaryBuilder(output_dir)


@pytest.fixture(autouse=True)
def numba_cuda_test_binaries(request):
    marker = request.node.get_closest_marker("numba_cuda_test_binaries")
    if marker is None:
        return None
    binary_builder = request.getfixturevalue("numba_cuda_test_binary_builder")
    binaries = binary_builder.binaries(tuple(marker.args))
    target = request.instance if request.instance is not None else request.cls
    if target is not None:
        target.numba_cuda_test_binaries = binaries
    return binaries


@pytest.fixture(scope="session", autouse=True)
def disable_low_occupancy_warnings():
    """Disable low occupancy warnings during tests (similar to numba-cuda CUDATestCase)."""
    original_value = cuda_config.CUDA_LOW_OCCUPANCY_WARNINGS
    cuda_config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
    yield
    cuda_config.CUDA_LOW_OCCUPANCY_WARNINGS = original_value


@pytest.fixture(scope="session", autouse=True)
def show_full_ice_tracebacks():
    """Show full tracebacks for internal compiler errors during tests."""
    import os

    original = os.environ.get("NUMBA_CUDA_MLIR_ICE_FULL_TB")
    os.environ["NUMBA_CUDA_MLIR_ICE_FULL_TB"] = "1"
    yield
    if original is None:
        os.environ.pop("NUMBA_CUDA_MLIR_ICE_FULL_TB", None)
    else:
        os.environ["NUMBA_CUDA_MLIR_ICE_FULL_TB"] = original


def pytest_addoption(parser):
    parser.addoption(
        "--dump-failed-filechecks",
        action="store_true",
        help="Dump reproducers for FileCheck tests that fail.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "numba_cuda_test_binaries(*artifacts): build only the named "
        "Numba-CUDA test binary artifacts for this test",
    )
    config.addinivalue_line(
        "markers",
        "requires_cc_min(cc, feature): skip test if GPU compute capability < cc",
    )
    config.addinivalue_line(
        "markers",
        "requires_cc_exact(cc, feature): skip test if GPU compute capability != cc",
    )
    if config.getoption("--pdb"):
        import logging

        logging.basicConfig(level=logging.DEBUG)
        config.option.reruns = 0
        config.option.capture = "yes"
        config.option.maxfail = 1
        config.option.verbose = 1
        config.option.showcapture = 1
        config.option.numprocesses = 0


def pytest_runtest_setup(item):
    for marker in item.iter_markers("requires_cc_min"):
        min_cc = marker.args[0]
        feature = marker.args[1] if len(marker.args) > 1 else "This feature"
        should_skip, msg = check_cc_min(min_cc, feature)
        if should_skip:
            pytest.skip(msg)

    for marker in item.iter_markers("requires_cc_exact"):
        exact_cc = marker.args[0]
        feature = marker.args[1] if len(marker.args) > 1 else "This feature"
        should_skip, msg = check_cc_exact(exact_cc, feature)
        if should_skip:
            pytest.skip(msg)


@pytest.fixture(scope="class")
def initialize_from_pytest_config(request):
    """
    Fixture to initialize the test case with pytest configuration options.
    """
    request.cls._dump_failed_filechecks = request.config.getoption(
        "dump_failed_filechecks"
    )
