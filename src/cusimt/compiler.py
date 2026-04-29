# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from cusimt.descriptor import MLIRDispatcher
from cusimt.lowering_utilities.type_conversions import to_numba_type
import inspect
from typing import Callable, Any
from cusimt import types, typing
from functools import lru_cache
from cusimt.numba_cuda.typing.templates import (
    ConcreteTemplate,
    AttributeTemplate,
    Registry,
)
from pathlib import Path
from cusimt.mlir_optimization import optimize
from cusimt.numba_cuda.codegen import ExternalCodeLibrary
from cusimt.numba_cuda.compiler import sigutils
from cusimt.descriptor import mlir_target
from cusimt.numba_cuda.core import funcdesc
from cusimt.logging import trace
from cusimt.typing.externals import ExternMLIRLibrary, ExternMLIRLibraryFunction


class CUFunc:
    """
    Wrapper around a CUDA function handle providing configuration methods.

    This provides the same interface as numba-cuda's cufunc objects for
    configuring cache preferences and shared memory carveout.
    """

    def __init__(self, handle):
        """
        Initialize with a CUfunction handle.

        Args:
            handle: A CUfunction handle from cuModuleGetFunction
        """
        self._handle = handle

    @property
    def handle(self):
        return self._handle

    def cache_config(self, prefer_shared=False, prefer_cache=False, prefer_equal=False):
        """
        Set the cache configuration preference for this function.

        Args:
            prefer_shared: Prefer more shared memory over L1 cache
            prefer_cache: Prefer more L1 cache over shared memory
            prefer_equal: Prefer equal shared memory and L1 cache

        If no preference is specified, uses the default (no preference).
        """
        from cuda.bindings import driver

        # Determine cache config based on arguments
        if prefer_shared:
            config = driver.CUfunc_cache.CU_FUNC_CACHE_PREFER_SHARED
        elif prefer_cache:
            config = driver.CUfunc_cache.CU_FUNC_CACHE_PREFER_L1
        elif prefer_equal:
            config = driver.CUfunc_cache.CU_FUNC_CACHE_PREFER_EQUAL
        else:
            config = driver.CUfunc_cache.CU_FUNC_CACHE_PREFER_NONE

        result = driver.cuFuncSetCacheConfig(self._handle, config)
        if result[0].value != 0:
            raise RuntimeError(f"cuFuncSetCacheConfig failed with error {result[0]}")

    def set_shared_memory_carveout(self, value):
        """
        Set the shared memory carveout percentage for this function.

        Args:
            value: Percentage of shared memory to carve out (0-100), or -1 for default

        Raises:
            ValueError: If value is not in valid range [-1, 100]
        """
        from cuda.bindings import driver

        if not isinstance(value, int) or value < -1 or value > 100:
            raise ValueError(
                f"Carveout value must be an integer in range [-1, 100], got {value}"
            )

        attr = (
            driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT
        )
        result = driver.cuFuncSetAttribute(self._handle, attr, value)
        if result[0].value != 0:
            raise RuntimeError(f"cuFuncSetAttribute failed with error {result[0]}")


class CodeLibrary:
    """
    Code library that holds compiled CUDA code and provides access to CUDA functions.

    This provides the same interface as numba-cuda's CUDACodeLibrary for accessing
    the compiled CUDA function with its configuration methods.
    """

    def __init__(self, cubin, func_name):
        """
        Initialize with compiled cubin and function name.

        Args:
            cubin: The compiled cubin bytes
            func_name: The mangled function name in the cubin
        """
        self._cubin = cubin
        self._func_name = func_name
        self._module = None
        self._cufunc_cache = {}

    def get_cufunc(self):
        """
        Get the CUDA function handle, loading the module if necessary.

        Returns:
            CUFunc: A wrapper around the CUfunction handle
        """
        from cuda.bindings import driver
        from cusimt.numba_cuda.cudadrv import devices

        # Get current device
        ctx = devices.get_context()
        device = ctx.device
        device_id = device.id

        # Check cache first
        if device_id in self._cufunc_cache:
            return self._cufunc_cache[device_id]

        # Load module from cubin
        result = driver.cuModuleLoadData(self._cubin)
        if result[0].value != 0:
            raise RuntimeError(f"cuModuleLoadData failed with error {result[0]}")
        self._module = result[1]

        # Get function from module
        result = driver.cuModuleGetFunction(self._module, self._func_name.encode())
        if result[0].value != 0:
            raise RuntimeError(
                f"cuModuleGetFunction failed for '{self._func_name}' with error {result[0]}"
            )
        handle = result[1]

        # Wrap and cache
        cufunc = CUFunc(handle)
        self._cufunc_cache[device_id] = cufunc

        return cufunc

    def get_kernel_attributes(self):
        """Query kernel resource usage attributes from the CUDA driver."""
        from cuda.bindings import driver

        cufunc = self.get_cufunc()
        handle = cufunc._handle

        def _get_attr(attr):
            err, val = driver.cuFuncGetAttribute(attr, handle)
            if err.value != 0:
                raise RuntimeError(f"cuFuncGetAttribute failed for {attr}: {err}")
            return int(val)

        return {
            "regs_per_thread": _get_attr(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS
            ),
            "shared_mem_per_block": _get_attr(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES
            ),
            "const_mem_size": _get_attr(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES
            ),
            "local_mem_per_thread": _get_attr(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES
            ),
            "max_threads_per_block": _get_attr(
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK
            ),
        }


class CompileResult:
    def __init__(self, cres):
        self.cres = cres
        self._code_library = None

    def __getattr__(self, attr):
        if hasattr(self.cres, attr):
            return getattr(self.cres, attr)
        if attr in self.cres.metadata.keys():
            return self.cres.metadata[attr]
        # Provide compatibility attributes
        if attr == "_codelibrary":
            # Lazily create the code library with cubin and func_name from metadata
            if self._code_library is None:
                cubin = self.cres.metadata.get("cubin")
                func_name = self.cres.metadata.get("func_name")
                if cubin is not None and func_name is not None:
                    self._code_library = CodeLibrary(cubin, func_name)
            return self._code_library
        if attr == "_type_annotation":
            return getattr(self.cres, "type_annotation", None)
        if attr == "cooperative":
            return getattr(
                self.cres,
                "use_cooperative",
                self.cres.metadata.get("use_cooperative", False),
            )
        if attr == "max_cooperative_grid_blocks":
            return self._max_cooperative_grid_blocks
        if attr == "inspect_asm":
            return self._inspect_asm
        if attr == "inspect_llvm":
            raise NotImplementedError(
                "inspect_llvm is not supported. "
                "Use inspect_asm() to inspect the PTX."
            )
        return getattr(self.cres, attr)

    def _ensure_kernel_attrs(self):
        """Lazily populate kernel resource attributes from the CUDA driver."""
        if "regs_per_thread" in self.cres.metadata:
            return
        lib = self._codelibrary
        if lib is not None:
            attrs = lib.get_kernel_attributes()
            self.cres.metadata.update(attrs)

    def _max_cooperative_grid_blocks(self, blockdim, dynsmemsize=0):
        import functools
        from cusimt.numba_cuda.cudadrv import devices

        ctx = devices.get_context()
        cufunc = self._codelibrary.get_cufunc()
        if isinstance(blockdim, tuple):
            blockdim = functools.reduce(lambda x, y: x * y, blockdim)
        active_per_sm = ctx.get_active_blocks_per_multiprocessor(
            cufunc, blockdim, dynsmemsize
        )
        sm_count = ctx.device.MULTIPROCESSOR_COUNT
        return active_per_sm * sm_count

    def _inspect_asm(self, cc=None):
        """Return PTX assembly from the compiled cubin metadata."""
        ptx = self.cres.metadata.get("ptx", "")
        if ptx:
            return ptx
        return self.cres.metadata.get("asm", "")


def _compile_and_optimize(pyfunc, sig=None, targetoptions=None):
    return _compile(pyfunc, sig, targetoptions, optimized=True)


def _compile_only(pyfunc, sig=None, targetoptions=None):
    """Compile to MLIR without running the optimization pipeline."""
    from cusimt.cuda import jit
    from cusimt import mlir_compiler
    from cusimt.numba_cuda.core import sigutils

    dispatcher = pyfunc
    if not isinstance(dispatcher, MLIRDispatcher):
        kws = targetoptions or {}
        dispatcher = jit(dispatcher, **kws)
    else:
        pyfunc = dispatcher.py_func

    if sig is None:
        sig = to_numba_type(inspect.signature(pyfunc))

    argtypes, return_type = sigutils.normalize_signature(sig)
    cres = mlir_compiler.compile_mlir(
        dispatcher.py_func,
        return_type,
        argtypes,
        targetoptions=dispatcher.targetoptions,
    )
    return CompileResult(cres)


def _compile(pyfunc, sig=None, targetoptions=None, optimized=True):
    from cusimt.cuda import jit

    dispatcher = pyfunc
    if not isinstance(dispatcher, MLIRDispatcher):
        kws = targetoptions or {}
        # Don't call verify_target_options here - jit() will do it
        dispatcher = jit(dispatcher, **kws)
    else:
        pyfunc = dispatcher.py_func

    if sig is None:
        sig = to_numba_type(inspect.signature(pyfunc))

    abi_info = (
        targetoptions.get("abi_info", None) if targetoptions is not None else None
    )
    output = targetoptions.get("output", None) if targetoptions is not None else None
    cres = dispatcher.compile(sig, abi_info=abi_info, output=output)
    if optimized:
        optimize(cres)
    return CompileResult(cres)


def compile_for(func, *args):
    from cusimt.numba_cuda.typing.typeof import typeof

    sig = typing.signature(types.none, *[typeof(arg) for arg in args])
    cres = _compile_and_optimize(func, sig)
    return cres


def compile_ptx(
    pyfunc,
    sig,
    debug=None,
    lineinfo=False,
    device=False,
    fastmath=False,
    cc=None,
    opt=None,
    abi="numba",
    abi_info=None,
    forceinline=False,
    launch_bounds=None,
):
    """Compile a Python function to PTX for a given signature."""
    return compile(
        pyfunc,
        sig,
        debug=debug,
        lineinfo=lineinfo,
        device=device,
        fastmath=fastmath,
        cc=cc,
        opt=opt,
        abi=abi,
        abi_info=abi_info,
        output="ptx",
        forceinline=forceinline,
        launch_bounds=launch_bounds,
    )


def compile(
    pyfunc,
    sig,
    debug=None,
    lineinfo=False,
    device=True,
    fastmath=False,
    cc=None,
    opt=None,
    abi="c",
    abi_info=None,
    output="ptx",
    forceinline=False,
    launch_bounds=None,
):
    """Compile a Python function to PTX or LTO-IR (numba-cuda compatible API)."""
    # Validate output type
    if output not in ("ptx", "ltoir"):
        raise NotImplementedError(f"Unsupported output type: {output}")

    # C ABI is not supported for kernels
    if abi == "c" and not device:
        raise NotImplementedError("The C ABI is not supported for kernels")

    # forceinline requires LTO-IR output
    if forceinline and output != "ltoir":
        raise ValueError("Can only designate forced inlining in LTO-IR")

    targetoptions = {}
    if debug is not None:
        targetoptions["debug"] = debug
    if lineinfo:
        targetoptions["lineinfo"] = lineinfo
    if device:
        targetoptions["device"] = device
    if fastmath:
        targetoptions["fastmath"] = fastmath
    if cc is not None:
        targetoptions["cc"] = cc
    if opt is not None:
        targetoptions["opt"] = opt
    if abi != "c":
        targetoptions["abi"] = abi
    if abi_info is not None:
        targetoptions["abi_info"] = abi_info
    if forceinline:
        targetoptions["inline"] = "always"
    if launch_bounds is not None:
        targetoptions["launch_bounds"] = launch_bounds
    if output != "ptx":
        targetoptions["output"] = output

    optimized = _compile_and_optimize(pyfunc, sig, targetoptions)

    # Kernels must have void return type
    if not device and optimized.signature.return_type != types.none:
        raise TypeError("CUDA kernel must have void return type")

    if output == "ptx":
        return optimized.ptx, optimized.signature.return_type
    elif output == "ltoir":
        return optimized.metadata.get("ltoir"), optimized.signature.return_type
    else:
        return optimized.ptx, optimized.signature.return_type


def compile_result(pyfunc, sig=None):
    """Compile and return full CompileResult for internal use."""
    return _compile_and_optimize(pyfunc, sig)


def compile_mlir(pyfunc, sig, optimized=False, **targetoptions):
    if optimized:
        cres = _compile_and_optimize(pyfunc, sig, targetoptions)
        return cres.metadata["mlir_module_optimized"]
    else:
        cres = _compile_only(pyfunc, sig, targetoptions)
        return cres.metadata["mlir_module_str"]


def compile_cubin(pyfunc, sig, **targetoptions):
    cres = _compile_and_optimize(pyfunc, sig, targetoptions)
    return cres.metadata["cubin"]


@lru_cache(maxsize=None)
def _get_typing_key_for_callable[T](callable: T, sig: typing.Signature) -> T:
    from cusimt.descriptor import mlir_target

    typingctx = mlir_target.typing_context
    targetctx = mlir_target.target_context

    class mlir_function_template(ConcreteTemplate):
        key = callable
        cases = [sig]

    typingctx.insert_user_function(callable, mlir_function_template)
    return mlir_function_template.key


def declare_mlir_library(
    source: str | Path,
) -> Callable[[...], Any]:
    """
    Declare a library defined in MLIR to be callable from a cuSIMT Python kernel.
    """
    from cusimt.typing.externals import ExternMLIRLibrary
    from cusimt.lowering_utilities.discover_functions import discover_functions

    if isinstance(source, Path):
        with open(source, "r") as f:
            source = f.read()

    functions = discover_functions(source)

    lib = ExternMLIRLibrary(functions, source)
    return lib


class ExternFunction:
    """A descriptor that can be used to call the external function from within
    a Python kernel.

    Note: This class intentionally does not define __hash__ or __eq__ so that
    each instance is unique by identity. This matches numba-cuda's behavior
    and prevents "key already in dictionary" errors when the same external
    function is declared multiple times.
    """

    def __init__(self, name, sig, use_cooperative=False, link=tuple(), abi="numba"):
        self.name = name
        self.sig = sig
        self.use_cooperative = use_cooperative
        self.link = link
        self.abi = abi


def declare_device_function(
    name, restype, argtypes, link, use_cooperative, abi="numba"
):
    from cusimt.descriptor import mlir_target

    typingctx = mlir_target.typing_context
    targetctx = mlir_target.target_context
    sig = typing.signature(restype, *argtypes)

    extfn = ExternFunction(name, sig, use_cooperative, link, abi=abi)

    # TODO(ajm): move cusimt to use numba.cuda's typing infrastructure, then use that instead
    class device_function_template(ConcreteTemplate):
        key = extfn
        cases = [sig]

    # Typing
    typingctx.insert_user_function(extfn, device_function_template)

    return device_function_template


def declare_device(name, sig, link=None, use_cooperative=False, abi="numba"):
    """
    Declare the signature of a foreign function. Returns a descriptor that can
    be used to call the function from a Python kernel.

    :param name: The name of the foreign function.
    :type name: str
    :param sig: The Numba signature of the function.
    :param link: External code to link when calling the function.
    :param use_cooperative: External code requires cooperative launch.
    :param abi: The ABI to use for the function. "numba" for Numba ABI, "c" for C ABI.
    """
    if abi not in ("numba", "c"):
        raise NotImplementedError(f"Unsupported ABI: {abi}")

    if link is None:
        link = tuple()
    else:
        if not isinstance(link, (list, tuple, set)):
            link = (link,)

    argtypes, restype = sigutils.normalize_signature(sig)
    if restype is None:
        msg = "Return type must be provided for device declarations"
        raise TypeError(msg)

    template = declare_device_function(
        name, restype, argtypes, link, use_cooperative, abi
    )

    return template.key
