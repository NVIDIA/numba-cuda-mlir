# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import ctypes
import os

from cusimt._mlir.passmanager import PassManager
from cusimt.tools import generate_mangled_name
from cusimt._mlir import ir
from cusimt.lowering_utilities import context
from cusimt.optimization import run_pre_codegen_patterns
from cusimt.numba_cuda.cudadrv.nvvm import CompilationUnit
from cusimt.lowering.numba_compat.llvm_utils import (
    MLIR_CAPI_LIB_PATH,
    NVPTX64_DATALAYOUT,
    NVPTX64_TRIPLE,
    translate_to_llvmir,
    dump_llvmir,
)
from numba.core.errors import UnsupportedError


def _maybe_link_nrt(linker) -> None:
    """Link NRT object code if NRT is enabled."""
    # Read at call time so we respect cudf/numba's config.CUDA_ENABLE_NRT (they may set it
    # after our config module was imported).
    from numba.cuda.core import config

    if not getattr(config, "CUDA_ENABLE_NRT", False):
        return

    cc = linker.cc
    if linker.lto:
        from cusimt.memory_management import compile_nrt_ltoir

        if linker._ltoirs:
            raise UnsupportedError(
                "Using LTOIR linking is not supported with NRT enabled."
            )
        linker.add_ltoir(compile_nrt_ltoir(cc))
    else:
        from cusimt.memory_management.nrt import compile_nrt_object

        linker.add_ptx(compile_nrt_object(cc))


def get_inline_pipeline():
    return "inline{default-pipeline=canonicalize max-iterations=10}"


def get_base_pipeline():
    inline = get_inline_pipeline()
    return (
        """
    builtin.module(
      reconcile-unrealized-casts,
      convert-shape-to-std,
      one-shot-bufferize,
      """
        + inline
        + """,
      convert-linalg-to-parallel-loops,
      convert-complex-to-standard,
      convert-complex-to-llvm,
      convert-nvgpu-to-nvvm,
      """
        + inline
        + """,
      convert-math-to-nvvm,
      gpu-kernel-outlining{data-layout-str=},
      convert-vector-to-scf{full-unroll=false lower-scalable=false lower-tensors=false target-rank=1},
      convert-scf-to-cf,
      convert-nvvm-to-llvm,
      convert-func-to-llvm{index-bitwidth=0 use-bare-ptr-memref-call-conv=false},
      expand-strided-metadata,
      lower-affine,
      math-uplift-to-fma,
      gpu.module(
        convert-gpu-to-nvvm{ has-redux=false index-bitwidth=64 use-bare-ptr-memref-call-conv=false}
      ),
      convert-arith-to-llvm{index-bitwidth=0},
      convert-index-to-llvm{index-bitwidth=64},
      canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
      cse,
      gpu.module(
        canonicalize{max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true}
      ),
      gpu.module(
        cse
      ),
      gpu.module(
        reconcile-unrealized-casts
      ),
      gpu-to-llvm{intersperse-sizes-for-kernels=false use-bare-pointers-for-host=false use-bare-pointers-for-kernels=false},
      """
        + inline
        + """,
      canonicalize{  max-iterations=10 max-num-rewrites=-1 region-simplify=normal test-convergence=false top-down=true},
      cse
    )
    """
    )


def _needs_nvvm70_path(cc: str) -> bool:
    """Return True when libnvvm requires the LLVM 7 dialect of NVVM IR.

    The modern dialect (based on LLVM 20+) is used on Blackwell and later
    (sm_100+).  Everything below sm_100 requires the NVVM70 path which
    translates MLIR to the LLVM 7 dialect for the NVVM70 reader.
    """
    sm = int("".join(c for c in cc if c.isdigit()))
    return sm < 100


_nvvm70_capi = None


def _get_nvvm70_capi():
    global _nvvm70_capi
    if _nvvm70_capi is not None:
        return _nvvm70_capi

    from cusimt.tools import get_nvvm70_capi_path

    lib = ctypes.CDLL(get_nvvm70_capi_path())
    lib.nvvm70_translate_gpu_module_from_op.restype = ctypes.c_int
    lib.nvvm70_translate_gpu_module_from_op.argtypes = [
        ctypes.c_void_p,  # raw_op (Operation*)
        ctypes.c_char_p,  # chip
        ctypes.c_char_p,  # data_layout
        ctypes.c_char_p,  # libllvm
        ctypes.c_char_p,  # libnvvm
        ctypes.c_char_p,  # libdevice
        ctypes.c_int,  # gen_lto
        ctypes.c_int,  # opt_level
        ctypes.c_int,  # gen_lineinfo
        ctypes.POINTER(ctypes.c_char_p),  # out
        ctypes.POINTER(ctypes.c_size_t),  # out_len
        ctypes.POINTER(ctypes.c_char_p),  # err_out
    ]
    lib.nvvm70_free.restype = None
    lib.nvvm70_free.argtypes = [ctypes.c_void_p]
    _nvvm70_capi = lib
    return lib


def _get_libnvvm_path() -> bytes:
    """Resolve libnvvm.so from the user's CTK (CUDA_HOME, conda, or pip)."""
    from numba.cuda.cudadrv.libs import get_cudalib

    return get_cudalib("nvvm").encode()


def _get_op_ptr(op) -> ctypes.c_void_p:
    """Extract raw mlir::Operation* from a Python MLIR Operation via its capsule."""
    capsule = op._CAPIPtr
    ptr = ctypes.pythonapi.PyCapsule_GetPointer
    ptr.restype = ctypes.c_void_p
    ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return ptr(capsule, b"cusimt._mlir.ir.Operation._CAPIPtr")


def _call_nvvm70_capi(module, target_options, gen_lto=False) -> bytes:
    """Compile MLIR gpu.module via in-process NVVM70 C API (raw Operation*)."""
    from cusimt._mlir.dialects import gpu
    from cusimt.tools import get_gpu_compute_capability
    from numba.cuda.cudadrv.libs import get_libdevice

    lib = _get_nvvm70_capi()
    chip = target_options.get("chip", get_gpu_compute_capability())

    gpu_modules = [op for op in module.body if isinstance(op, gpu.GPUModuleOp)]
    if len(gpu_modules) != 1:
        raise ValueError(f"Expected exactly one gpu.module, found {len(gpu_modules)}")
    gpu_mod = gpu_modules[0]

    if target_options.get("dump_mlir") or target_options.get("dump"):
        print(f"=============== NVVM70 MLIR Module ===============\n\n{gpu_mod}\n")

    raw_op = _get_op_ptr(gpu_mod.operation)

    libllvm = os.environ.get("LIBLLVM7", "")
    if not libllvm:
        bundled = os.path.join(os.path.dirname(__file__), "lib", "libLLVM-7.so")
        if os.path.isfile(bundled):
            libllvm = os.path.realpath(bundled)

    if not libllvm:
        raise RuntimeError(
            "NVVM70 path requires libLLVM-7.so. Set LIBLLVM7=/path/to/libLLVM-7.so"
        )

    libnvvm = _get_libnvvm_path().decode()
    libdevice = get_libdevice()
    opt_level = int(target_options.get("opt_level", 2))
    if target_options.get("debug", False):
        debug_level = 2
    elif target_options.get("lineinfo", False):
        debug_level = 1
    else:
        debug_level = 0

    out = ctypes.c_char_p()
    out_len = ctypes.c_size_t()
    err_out = ctypes.c_char_p()

    rc = lib.nvvm70_translate_gpu_module_from_op(
        raw_op,
        chip.encode(),
        None,
        libllvm.encode(),
        libnvvm.encode(),
        libdevice.encode(),
        1 if gen_lto else 0,
        opt_level,
        debug_level,
        ctypes.byref(out),
        ctypes.byref(out_len),
        ctypes.byref(err_out),
    )

    if rc != 0:
        msg = err_out.value.decode() if err_out.value else "unknown error"
        if err_out.value:
            lib.nvvm70_free(err_out)
        raise RuntimeError(f"nvvm70 translation failed: {msg}")

    result = ctypes.string_at(out, out_len.value)
    lib.nvvm70_free(out)
    return result


def _prepare_llvm_ir(module, dump=False) -> bytes:
    """Translate gpu.module to LLVM IR and apply libnvvm compatibility downgrades."""
    from cusimt._mlir.dialects import gpu
    from cusimt._cext import downgrade_for_libnvvm
    from cusimt.tools import get_cuda_runtime_version

    gpu_modules = [op for op in module.body if isinstance(op, gpu.GPUModuleOp)]
    if len(gpu_modules) != 1:
        raise ValueError(f"Expected exactly one gpu.module, found {len(gpu_modules)}")

    gpu_mod = gpu_modules[0]
    gpu_mod.operation.attributes["llvm.data_layout"] = ir.StringAttr.get(
        NVPTX64_DATALAYOUT
    )
    gpu_mod.operation.attributes["llvm.target_triple"] = ir.StringAttr.get(
        NVPTX64_TRIPLE
    )

    llvm_mod, llvm_ctx = translate_to_llvmir(gpu_mod.operation)

    if dump:
        print(f"=============== LLVM IR ===============\n\n{dump_llvmir(llvm_mod)}\n\n")

    ctk_major, ctk_minor = get_cuda_runtime_version()
    return downgrade_for_libnvvm(
        llvm_mod, llvm_ctx, ctk_major, ctk_minor, MLIR_CAPI_LIB_PATH
    )


def _nvvm_options(cc: str, target_options=None, **extra) -> dict:
    """Build libnvvm CompilationUnit options from arch + target options."""
    opts = {"arch": f"compute_{cc}", **extra}
    if target_options is None:
        return opts
    if target_options.get("fastmath"):
        opts.update({"ftz": True, "fma": True, "prec_div": False, "prec_sqrt": False})
    # Note: we intentionally omit -g and -generate-line-info here.
    # Our MLIR pipeline embeds DWARF metadata (DICompileUnit, DISubprogram, DILocation)
    # into the LLVM IR when debug=True or lineinfo=True. libnvvm honors that metadata
    # to produce debug sections and .loc/.file PTX directives automatically. Passing
    # either flag to libnvvm conflicts with pre-existing debug metadata, causing
    # NVVM_ERROR_IR_VERSION_MISMATCH.
    opt = target_options.get("opt")
    if opt is False or opt == 0:
        opts["opt"] = 0
    return opts


def _compile_to_ptx(llvm_ir: bytes, cc: str, libdevice, nvvm_opts=None) -> bytes:
    """Compile LLVM IR to PTX via libnvvm."""
    if nvvm_opts is None:
        nvvm_opts = {"arch": f"compute_{cc}"}
    cu = CompilationUnit(nvvm_opts)
    cu.add_module(llvm_ir)
    cu.verify()
    cu.lazy_add_module(libdevice.get())
    return cu.compile()


def _dump_module(mod, header):
    print(header, end="\n\n")
    # Include loc and #llvm.di_* when present (e.g. debug/lineinfo).
    mod.operation.print(enable_debug_info=True)
    print("\n\n")


def optimize(cres):
    with context.get_context():
        target_options = cres.metadata["targetoptions"]
        dump_mlir = target_options.get("dump_mlir", False) or target_options.get(
            "dump", False
        )
        # Parse pre-optimization MLIR (debug metadata present when debug/lineinfo enabled).
        module = ir.Module.parse(cres.metadata["mlir_module_str"])

        if dump_mlir:
            _dump_module(module, "=============== MLIR Module ===============")

        pm = PassManager.parse(get_base_pipeline())
        pm.enable_ir_printing(
            print_before_all=target_options.get("print_before_all", False),
            print_after_all=target_options.get("print_after_all", False),
        )
        pm.run(module.operation)
        cres.metadata["mlir_module_optimized"] = str(module)
        if dump_mlir:
            _dump_module(
                module, "=============== Optimized MLIR Module ==============="
            )

        run_pre_codegen_patterns(module)
        if dump_mlir:
            _dump_module(
                module,
                "=============== Optimized MLIR Module (after pre-codegen patterns) ===============",
            )

        chip = target_options.get("chip")
        if not chip:
            from cusimt.tools import get_gpu_compute_capability

            chip = get_gpu_compute_capability()
        cc = chip.replace("sm_", "")
        is_lto = target_options.get("output", "ptx") == "ltoir"

        from numba.cuda.cudadrv.nvvm import LibDevice

        use_nvvm70 = _needs_nvvm70_path(cc)

        if use_nvvm70:
            ptx = _call_nvvm70_capi(module, target_options)
            llvm_ir = None
        else:
            llvm_ir = _prepare_llvm_ir(
                module, dump=target_options.get("dump_llvmir", False)
            )

            libdevice = LibDevice()
            nvvm_opts = _nvvm_options(cc, target_options)

            ptx = _compile_to_ptx(llvm_ir, cc, libdevice, nvvm_opts)

        cres.metadata["ptx"] = ptx.decode()

        if target_options.get("dump_ptx", False):
            print(f"=============== PTX ===============\n\n{cres.metadata['ptx']}\n\n")

        linker = copy.deepcopy(cres.metadata["linker"])

        if is_lto:
            if use_nvvm70:
                ltoir = _call_nvvm70_capi(module, target_options, gen_lto=True)
            else:
                nvvm_opts = _nvvm_options(cc, target_options)
                cu_lto = CompilationUnit({**nvvm_opts, "gen-lto": None})
                cu_lto.add_module(llvm_ir)
                cu_lto.verify()
                cu_lto.lazy_add_module(LibDevice().get())
                ltoir = cu_lto.compile()
            cres.metadata["ltoir"] = ltoir
            linker.add_ltoir(ltoir)
        else:
            linker.add_ptx(ptx)

        _maybe_link_nrt(linker)
        code = linker.complete()
        cres.metadata["cubin"] = code.code

        if target_options.get("dump_cubin", False):
            print(f"=============== Cubin ===============\n\n{code.code}\n\n")

        # TODO: parse CC from the object and ensure it's not greater than the
        # greatest supported CC via _get_gpu_compute_capability()
        cres.metadata["func_name"] = generate_mangled_name(
            cres.fndesc.qualname, cres.fndesc.argtypes
        )

        cres.library._ptx = cres.metadata["ptx"]
        cres.library._mlir_str = cres.metadata["mlir_module_optimized"]
