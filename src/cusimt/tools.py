from pathlib import Path
import os
import subprocess
import re
from functools import lru_cache
from typing import overload
from numba.core import itanium_mangler
from numba.core import types


def parse_compute_capability(compute_cap: str) -> tuple[int, int]:
    match = re.match(r"sm_([1-9]\d?)(\d)(a)?", compute_cap)
    if match:
        return int(match.group(1)), int(match.group(2))
    else:
        raise ValueError(f"Invalid compute capability: {compute_cap}")


def format_arch(cc: tuple[int, int]) -> str:
    """
    Format a compute capability tuple as an sm_XX[a] string.

    Args:
        cc: Compute capability as (major, minor) tuple
    """
    return f"sm_{cc[0]}{cc[1]}"


@lru_cache(maxsize=1)
def get_cuda_toolkit_path() -> str | None:
    """Get CUDA toolkit root path from env vars, numba-cuda discovery, or system."""
    for var in ("CUDA_HOME", "CUDA_PATH"):
        val = os.environ.get(var)
        if val and os.path.isdir(os.path.join(val, "bin")):
            return val

    from numba.cuda.cuda_paths import get_cuda_paths

    libdevice = get_cuda_paths().get("libdevice")
    if libdevice and libdevice.info:
        # Toolkit root is 3 levels up: toolkit/nvvm/libdevice/libdevice.10.bc
        root = os.path.dirname(os.path.dirname(os.path.dirname(libdevice.info)))
        if os.path.isdir(os.path.join(root, "bin")):
            return root

    if os.path.isdir("/usr/local/cuda/bin"):
        return "/usr/local/cuda"

    return None


@lru_cache(maxsize=1)
def get_cuda_runtime_version() -> tuple[int, int]:
    """
    Get the CUDA toolkit version as a (major, minor) tuple.

    Use ``nvrtcVersion()`` from libnvrtc, which reflects the version of
    installed CUDA toolkit.
    """
    from cuda.bindings import nvrtc

    err, major, minor = nvrtc.nvrtcVersion()
    if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
        raise RuntimeError(f"nvrtcVersion() failed: {err}")
    return (major, minor)


# CTK (major, minor) -> max PTX ISA version supported by that toolkit's libnvvm.
# Expressed as the integer used in the `+ptxNN` target feature string.
_CTK_TO_MAX_PTX: dict[tuple[int, int], int] = {
    (12, 8): 87,  # PTX 8.7
    (12, 9): 88,  # PTX 8.8
    (13, 0): 90,  # PTX 9.0
    (13, 1): 91,  # PTX 9.1
    (13, 2): 92,  # PTX 9.2
}


@lru_cache(maxsize=1)
def get_max_ptx_version() -> int | None:
    """Return the highest PTX ISA version (as ``+ptxNN`` integer) the installed
    CUDA toolkit can assemble, or ``None`` if the toolkit version is not in the
    lookup table.

    When ``None`` is returned the caller should fall back to the NVPTX backend
    default (minimum PTX for the target SM).
    """
    ctk = get_cuda_runtime_version()
    return _CTK_TO_MAX_PTX.get(ctk)


@overload
def get_gpu_compute_capability(as_type: type = str) -> str: ...


@overload
def get_gpu_compute_capability(as_type: type = tuple) -> tuple[int, int]: ...


def get_gpu_compute_capability(as_type: type = str) -> str | tuple[int, int]:
    """
    Query the GPU compute capability and return the appropriate sm_arch.

    """
    assert as_type in (str, tuple), "as_type must be str or tuple"
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0 and result.stdout.strip():
            compute_cap = result.stdout.strip().split("\n")[0]  # Get first GPU
            major, minor = compute_cap.split(".")
            cc = (int(major), int(minor))
            if as_type is tuple:
                return cc
            return format_arch(cc)
        else:
            raise RuntimeError(
                f"Error running nvidia-smi to get GPU compute capability: {result.returncode}"
            )

    except Exception as e:
        raise RuntimeError(f"Error running nvidia-smi to get GPU compute capability: {e}")


def _check_cuda_result(result):
    """Unwrap CUDA driver API result, raising on error."""
    from cuda.bindings import driver

    if result[0].value != 0:
        _, name = driver.cuGetErrorName(result[0])
        raise RuntimeError(f"CUDA error: {name}")
    return result[1] if len(result) == 2 else result[1:] if len(result) > 2 else None


@lru_cache(maxsize=32)
def get_max_active_clusters(cluster_size: int, device_id: int = 0) -> int:
    """
    Query the maximum number of active clusters for a given cluster size.

    This compiles a minimal dummy kernel and uses cuOccupancyMaxActiveClusters
    to determine the hardware limit for concurrent cluster execution.

    Args:
        cluster_size: Number of CTAs per cluster (1-32)
        device_id: CUDA device ID (default: 0)

    Returns:
        Maximum number of clusters that can be active concurrently
    """
    if cluster_size <= 0 or cluster_size > 32:
        raise ValueError(f"Cluster size must be between 1 and 32, got {cluster_size}")

    from cuda.bindings import driver

    device = _check_cuda_result(driver.cuDeviceGet(device_id))
    max_smem = _check_cuda_result(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            device,
        )
    )

    kernel = _get_dummy_kernel_function()
    _check_cuda_result(
        driver.cuFuncSetAttribute(
            kernel,
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
            max_smem,
        )
    )
    max_dyn_smem = _check_cuda_result(driver.cuOccupancyAvailableDynamicSMemPerBlock(kernel, 1, 1))
    max_active_blocks = _check_cuda_result(
        driver.cuOccupancyMaxActiveBlocksPerMultiprocessor(kernel, 1, max_dyn_smem)
    )
    _check_cuda_result(
        driver.cuFuncSetAttribute(
            kernel,
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED,
            1,
        )
    )

    cluster_dims_attr = driver.CUlaunchAttribute()
    cluster_dims_attr.id = driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
    (
        cluster_dims_attr.value.clusterDim.x,
        cluster_dims_attr.value.clusterDim.y,
        cluster_dims_attr.value.clusterDim.z,
    ) = (cluster_size, 1, 1)

    launch_config = driver.CUlaunchConfig()
    launch_config.blockDimX, launch_config.blockDimY, launch_config.blockDimZ = (
        128,
        1,
        1,
    )
    launch_config.gridDimX, launch_config.gridDimY, launch_config.gridDimZ = (
        cluster_size,
        max_active_blocks,
        1,
    )
    launch_config.sharedMemBytes, launch_config.numAttrs, launch_config.attrs = (
        max_dyn_smem,
        1,
        [cluster_dims_attr],
    )

    return _check_cuda_result(driver.cuOccupancyMaxActiveClusters(kernel, launch_config))


@lru_cache(maxsize=1)
def _get_dummy_kernel_function():
    """Compile and cache a minimal dummy kernel for occupancy queries."""
    from cuda.bindings import driver
    from cusimt.compiler import compile_cubin
    from cusimt import types

    def _dummy_kernel() -> types.void:
        pass

    cubin = compile_cubin(_dummy_kernel, ())
    cuda_library = _check_cuda_result(driver.cuLibraryLoadData(cubin, None, None, 0, None, None, 0))
    kernels = _check_cuda_result(driver.cuLibraryEnumerateKernels(1, cuda_library))
    return _check_cuda_result(driver.cuKernelGetFunction(kernels[0]))


@lru_cache(maxsize=1)
def is_using_llvm70() -> bool:
    """Return True if the current environment will use the LLVM70 compilation path."""
    from cusimt.mlir_optimization import _needs_llvm70_path

    cc = get_gpu_compute_capability().replace("sm_", "")
    return _needs_llvm70_path(cc)


@lru_cache(maxsize=1)
def get_llvm70_capi_path() -> str:
    """Resolve path to the libMLIRToLLVM70.so shared library."""
    import cusimt._mlir._mlir_libs as _mlir_libs

    candidates = [
        Path(__file__).parent / "libMLIRToLLVM70.so",
        Path(_mlir_libs.__path__[0]) / "libMLIRToLLVM70.so",
    ]
    for c in candidates:
        if c.exists():
            return str(c.resolve())
    raise FileNotFoundError(
        "libMLIRToLLVM70.so not found. Rebuild cusimt with MLIR_DIR env var set."
    )


def generate_mangled_name(func_name, argtypes):
    """
    Return a mangled name given a function name and argtypes using Numba internals.
    """
    normalized_argtypes = [argtype_normalization(argtype) for argtype in argtypes]
    return itanium_mangler.mangle(func_name, normalized_argtypes)


def argtype_normalization(argtype):
    """
    Normalize the Numba data type
    """
    if isinstance(argtype, types.BooleanLiteral):
        return types.boolean
    elif isinstance(argtype, types.IntegerLiteral):
        return types.int64
    elif isinstance(argtype, types.UniTuple):
        if isinstance(argtype.dtype, types.IntegerLiteral):
            return types.UniTuple(dtype=argtype_normalization(argtype.dtype), count=argtype.count)
        else:
            return argtype
    elif isinstance(argtype, types.NumberClass):
        return argtype.dtype
    else:
        return argtype


def replace_numba_cuda_in_numba_install():
    """
    Replace the numba-cuda import in the numba install with the cusimt import.
    """
    import importlib.util
    import os

    numba_spec = importlib.util.find_spec("numba")
    numba_cuda_spec = importlib.util.find_spec("numba_cuda")
    numba = Path(numba_cuda_spec.origin).parent
    numba_cuda = Path(numba_cuda_spec.origin).parent
    their_cuda = numba / "cuda"
    our_cuda = numba_cuda / "numba" / "cuda"
    print(f"replacing {their_cuda} with {our_cuda}")
    if their_cuda.exists():
        if their_cuda.is_symlink():
            their_cuda.unlink()
        else:
            their_cuda.rmdir()
    their_cuda.symlink_to(our_cuda, target_is_directory=True)


def create_numba_cuda_stubs():
    """
    Create stubs for the Numba-CUDA repository.
    The redirector confuses IDE features, so generating stubs allows the IDE to
    find the correct symbols overridden by numba-cuda.
    """
    import importlib.util
    import os

    pyright_spec = importlib.util.find_spec("pyright")
    basedpyright_spec = importlib.util.find_spec("basedpyright")
    if pyright_spec is None and basedpyright_spec is None:
        raise RuntimeError("Could not find pyright or basedpyright")
    if pyright_spec is not None:
        import pyright
    else:
        import basedpyright as pyright
    import cusimt
    import shutil

    project_root = Path(cusimt.__path__[0]).parent
    os.chdir(project_root)

    # Start with numba and overwrite with numba-cuda stubs
    for spec in [
        "numba",
        "numba.experimental",
        "numba.types",
        "numba_cuda.numba.cuda",
        "numba_cuda.numba.cuda.core",
        "numba_cuda.numba.cuda.types",
        "numba_cuda.numba.cuda.types.ext_types",
        "cusimt",
    ]:
        print(f"Creating stubs for {spec=}")
        print(pyright.run("--createstub", spec))

    for root, _, files in os.walk(project_root / "typings" / "numba"):
        numba_cuda = root.replace("numba/cuda", "numba_cuda/numba/cuda")
        if "cuda" not in root:
            continue
        for file in files:
            ours = Path(numba_cuda) / file
            theirs = Path(root) / file
            if file.endswith(".pyi"):
                if ours.exists() and theirs.exists():
                    print(ours, theirs)
                    stheirs = open(theirs, "r").read()
                    sours = open(ours, "r").read()
                    with open(theirs, "w") as f:
                        f.write(sours)
                        f.write(stheirs)
                elif ours.exists():
                    shutil.copy(ours, theirs)
    print(f"Created stubs for numba-cuda in {project_root / 'typings'}")


def generate_libdevice_stubs():
    print(
        """# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

from cusimt.numba_cuda.types import (
    int16,
    int32,
    int64,
    float32,
    float64,
    UniTuple,
    Tuple,
)
"""
    )

    T = '''
def {name}({params}) -> {return_type}:
    """
    See https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_{name}.html

    CAPI:

    {capi};
    """
'''
    from cusimt.cuda.libdevicefuncs import libdevice_descriptors

    for descriptor in libdevice_descriptors():
        print(descriptor)
