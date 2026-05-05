# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import os
from io import StringIO

from numba_cuda_mlir._mlir.passmanager import PassManager
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.tools import generate_mangled_name
from numba_cuda_mlir._mlir import ir
from numba_cuda_mlir.lowering_utilities import context
from numba_cuda_mlir.optimization import run_pre_codegen_patterns
from numba_cuda_mlir.numba_cuda.cudadrv.nvvm import CompilationUnit
from numba_cuda_mlir.logging import trace
from numba_cuda_mlir.mlir.util import find_ops
from numba_cuda_mlir.lowering.numba_compat.llvm_utils import (
    MLIR_CAPI_LIB_PATH,
    NVPTX64_DATALAYOUT,
    NVPTX64_TRIPLE,
    translate_to_llvmir,
    dump_llvmir,
)
from numba_cuda_mlir.numba_cuda.core.errors import UnsupportedError

from numba_cuda_mlir.memory_management.rtsys import rtsys


def _maybe_link_nrt(linker) -> None:
    """Link NRT object code unconditionally.

    The caller gates this on cres.metadata["needs_nrt"], so once we get here
    NRT linking is required.
    """
    cc = linker.cc
    if linker.lto:
        from numba_cuda_mlir.memory_management import compile_nrt_ltoir

        if linker._ltoirs:
            raise UnsupportedError("Using LTOIR linking is not supported with NRT enabled.")
        linker.add_ltoir(compile_nrt_ltoir(cc))
    else:
        from numba_cuda_mlir.memory_management.nrt import compile_nrt_object

        linker.add_ptx(compile_nrt_object(cc))


def _nrt_memsys_setup_callback(loaded_module):
    """Post-load callback that sets the shared NRT_MemSys pointer for a kernel module.

    Invoked by ``_make_post_load_hook`` after the CUlibrary is loaded.
    This ensures the kernel's ``TheMSys`` device global points to the
    shared allocation so NRT statistics are collected.
    """
    rtsys.ensure_allocated()
    rtsys.ensure_initialized()
    rtsys.set_memsys_to_library(loaded_module.handle)


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


def _needs_llvm70_path(cc: str) -> bool:
    """Return True when libnvvm requires the LLVM 7 dialect of NVVM IR.

    The modern dialect (based on LLVM 20+) is used on Blackwell and later
    (sm_100+).  Everything below sm_100 requires the LLVM70 path which
    translates MLIR to the LLVM 7 dialect for the LLVM70 reader.
    """
    sm = int("".join(c for c in cc if c.isdigit()))
    return sm < 100


_llvm70_capi = None


def _get_llvm70_capi():
    global _llvm70_capi
    if _llvm70_capi is not None:
        return _llvm70_capi

    from numba_cuda_mlir.tools import get_llvm70_capi_path

    lib = ctypes.CDLL(get_llvm70_capi_path())
    lib.llvm70_translate_gpu_module_from_op.restype = ctypes.c_int
    lib.llvm70_translate_gpu_module_from_op.argtypes = [
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
    lib.llvm70_free.restype = None
    lib.llvm70_free.argtypes = [ctypes.c_void_p]
    _llvm70_capi = lib
    return lib


def _get_libnvvm_path() -> bytes:
    """Resolve libnvvm.so from the user's CTK (CUDA_HOME, conda, or pip)."""
    from numba_cuda_mlir.numba_cuda.cudadrv.libs import get_cudalib

    return get_cudalib("nvvm").encode()


def _get_op_ptr(op) -> ctypes.c_void_p:
    """Extract raw mlir::Operation* from a Python MLIR Operation via its capsule."""
    capsule = op._CAPIPtr
    ptr = ctypes.pythonapi.PyCapsule_GetPointer
    ptr.restype = ctypes.c_void_p
    ptr.argtypes = [ctypes.py_object, ctypes.c_char_p]
    return ptr(capsule, b"numba_cuda_mlir._mlir.ir.Operation._CAPIPtr")


def _call_llvm70_capi(module, target_options, gen_lto=False) -> bytes:
    """Compile MLIR gpu.module via in-process LLVM70 C API (raw Operation*)."""
    from numba_cuda_mlir._mlir.dialects import gpu
    from numba_cuda_mlir.tools import get_gpu_compute_capability
    from numba_cuda_mlir.numba_cuda.cudadrv.libs import get_libdevice

    lib = _get_llvm70_capi()
    chip = target_options.get("chip", get_gpu_compute_capability())

    gpu_modules = [op for op in module.body if isinstance(op, gpu.GPUModuleOp)]
    if len(gpu_modules) != 1:
        raise ValueError(f"Expected exactly one gpu.module, found {len(gpu_modules)}")
    gpu_mod = gpu_modules[0]

    if target_options.get("dump_mlir") or target_options.get("dump"):
        print(f"=============== LLVM70 MLIR Module ===============\n\n{gpu_mod}\n")

    raw_op = _get_op_ptr(gpu_mod.operation)

    libllvm = os.environ.get("LIBLLVM7", "")
    if not libllvm:
        bundled = os.path.join(os.path.dirname(__file__), "lib", "libLLVM-7.so")
        if os.path.isfile(bundled):
            libllvm = os.path.realpath(bundled)

    if not libllvm:
        raise RuntimeError("LLVM70 path requires libLLVM-7.so. Set LIBLLVM7=/path/to/libLLVM-7.so")

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

    rc = lib.llvm70_translate_gpu_module_from_op(
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
            lib.llvm70_free(err_out)
        raise RuntimeError(f"llvm70 translation failed: {msg}")

    result = ctypes.string_at(out, out_len.value)
    lib.llvm70_free(out)
    return result


def _prepare_llvm_ir(module, dump=False) -> bytes:
    """Translate gpu.module to LLVM IR and apply libnvvm compatibility downgrades."""
    from numba_cuda_mlir._mlir.dialects import gpu
    from numba_cuda_mlir._cext import downgrade_for_libnvvm
    from numba_cuda_mlir.tools import get_cuda_runtime_version

    gpu_modules = [op for op in module.body if isinstance(op, gpu.GPUModuleOp)]
    if len(gpu_modules) != 1:
        raise ValueError(f"Expected exactly one gpu.module, found {len(gpu_modules)}")

    gpu_mod = gpu_modules[0]
    gpu_mod.operation.attributes["llvm.data_layout"] = ir.StringAttr.get(NVPTX64_DATALAYOUT)
    gpu_mod.operation.attributes["llvm.target_triple"] = ir.StringAttr.get(NVPTX64_TRIPLE)

    llvm_mod, llvm_ctx = translate_to_llvmir(gpu_mod.operation)

    if dump:
        print(f"=============== LLVM IR ===============\n\n{dump_llvmir(llvm_mod)}\n\n")

    ctk_major, ctk_minor = get_cuda_runtime_version()
    return downgrade_for_libnvvm(llvm_mod, llvm_ctx, ctk_major, ctk_minor, MLIR_CAPI_LIB_PATH)


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


def _find_dbg_var_name(loc):
    """Extract dbg variable name from nested locations."""
    if isinstance(loc, ir.NameLoc) and loc.name_str.startswith("dbg_var:"):
        return loc.name_str[len("dbg_var:") :]
    if isinstance(loc, ir.FusedLoc):
        for nested_loc in loc.locations:
            name = _find_dbg_var_name(nested_loc)
            if name is not None:
                return name
    return None


def _strip_dbg_var_nameloc(loc):
    """Strip dbg_var: NameLoc entries from a location tree."""
    if isinstance(loc, ir.NameLoc) and loc.name_str.startswith("dbg_var:"):
        return None
    if not isinstance(loc, ir.FusedLoc):
        return loc
    stripped = []
    for nested_loc in loc.locations:
        new_loc = _strip_dbg_var_nameloc(nested_loc)
        if new_loc is not None:
            stripped.append(new_loc)
    if not stripped:
        return loc
    if len(stripped) == 1:
        return stripped[0]
    return ir.Location.fused(stripped)


def _cleanup_deferred_dbg_attrs(module_attrs):
    """Remove internal deferred-debug module attributes."""
    expr_key = "metadata.dbg_declare_expr"
    var_key_prefix = "metadata.dbg_declare_var_"
    for key in [k for k in module_attrs.keys() if k.startswith(var_key_prefix)]:
        del module_attrs[key]
    if expr_key in module_attrs:
        del module_attrs[expr_key]


def _emit_deferred_dbg_declares(module):
    """A deferred emission of dbg.declare for variables tagged during lowering.

    Lowering tags memref.alloca ops with ``dbg_var:<name>`` NameLoc for variables
    that need to be emitted as dbg.declare after the base pipeline. This helper
    emits llvm.intr.dbg.declare for tagged allocas in module-local DI scope.
    """
    module_attrs = module.operation.attributes
    var_key_prefix = "metadata.dbg_declare_var_"
    expr_key = "metadata.dbg_declare_expr"
    expr_attr = module_attrs[expr_key] if expr_key in module_attrs else None
    if expr_attr is None:
        _cleanup_deferred_dbg_attrs(module_attrs)
        return False

    tagged_vars = []
    emitted_vars = set()
    for op in find_ops(module, lambda o: o.name == "llvm.alloca"):
        loc = op.location
        var_name = _find_dbg_var_name(loc)
        if var_name is None:
            continue
        if var_name in emitted_vars:
            continue
        key = f"{var_key_prefix}{var_name}"
        var_attr = module_attrs[key] if key in module_attrs else None
        if var_attr is None:
            continue
        tagged_vars.append((op, var_name, var_attr))
        emitted_vars.add(var_name)

    for op, var_name, var_attr in tagged_vars:
        # Recover the clean/original location (without dbg_var tags) and use it
        # consistently for both the alloca and dbg.declare ops.
        op.location = _strip_dbg_var_nameloc(op.location)
        with ir.InsertionPoint.after(op), op.location:
            llvm.intr_dbg_declare(op.result, var_attr, location_expr=expr_attr)
        trace("Deferred dbg.declare emitted for %s", var_name)
    _cleanup_deferred_dbg_attrs(module_attrs)
    return bool(tagged_vars)


def _dump_lto_assembly(cres, linker, target_options):
    """Diagnostic LTO-to-PTX link to dump post-LTO assembly and warn about
    non-LTO linkables, mirroring CUDACodeLibrary.get_cubin() / get_lto_ptx()."""
    from numba_cuda_mlir.numba_cuda.cudadrv.driver import _Linker

    diag_linker = _Linker(
        max_registers=linker.max_registers,
        cc=linker.cc,
        additional_flags=["-ptx"],
        lto=True,
    )
    ltoir = cres.metadata.get("ltoir")
    if ltoir:
        diag_linker.add_ltoir(ltoir)
    for link_file in target_options.get("link", []):
        diag_linker.add_file_guess_ext(link_file, ignore_nonlto=True)
    ptx_after_lto = diag_linker.get_linked_ptx().decode("utf-8")
    name = cres.fndesc.qualname
    print(("ASSEMBLY (AFTER LTO) %s" % name).center(80, "-"))
    print(ptx_after_lto)
    print("=" * 80)


def optimize(cres):
    with context.get_context():
        target_options = cres.metadata["targetoptions"]
        dump_mlir = target_options.get("dump_mlir", False) or target_options.get("dump", False)
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

        if target_options.get("debug"):
            _emit_deferred_dbg_declares(module)
        if target_options.get("debug") or target_options.get("lineinfo"):
            with StringIO() as sb:
                module.operation.print(enable_debug_info=True, file=sb)
                cres.metadata["mlir_module_optimized"] = sb.getvalue()
        else:
            cres.metadata["mlir_module_optimized"] = str(module)
        if dump_mlir:
            _dump_module(module, "=============== Optimized MLIR Module ===============")

        run_pre_codegen_patterns(module)
        if dump_mlir:
            _dump_module(
                module,
                "=============== Optimized MLIR Module (after pre-codegen patterns) ===============",
            )

        chip = target_options.get("chip")
        if not chip:
            from numba_cuda_mlir.tools import get_gpu_compute_capability

            chip = get_gpu_compute_capability()
        cc = chip.replace("sm_", "")
        is_lto = target_options.get("output", "ptx") == "ltoir"

        from numba_cuda_mlir.numba_cuda.cudadrv.nvvm import LibDevice

        use_llvm70 = _needs_llvm70_path(cc)

        if use_llvm70:
            ptx = _call_llvm70_capi(module, target_options)
            llvm_ir = None
        else:
            llvm_ir = _prepare_llvm_ir(module, dump=target_options.get("dump_llvmir", False))

            libdevice = LibDevice()
            nvvm_opts = _nvvm_options(cc, target_options)

            ptx = _compile_to_ptx(llvm_ir, cc, libdevice, nvvm_opts)

        cres.metadata["ptx"] = ptx.decode()

        if target_options.get("dump_ptx", False):
            print(f"=============== PTX ===============\n\n{cres.metadata['ptx']}\n\n")

        linker = cres.metadata["linker"].recreate_with_lto()

        if is_lto:
            if use_llvm70:
                ltoir = _call_llvm70_capi(module, target_options, gen_lto=True)
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

        if cres.metadata.get("needs_nrt") and not cres.metadata.get("nrt_inline"):
            _maybe_link_nrt(linker)
        code = linker.complete()
        cres.metadata["cubin"] = code.code

        if is_lto:
            from numba_cuda_mlir.numba_cuda import config

            if config.DUMP_ASSEMBLY:
                _dump_lto_assembly(cres, linker, target_options)

        if target_options.get("dump_cubin", False):
            print(f"=============== Cubin ===============\n\n{code.code}\n\n")

        if cres.metadata.get("needs_nrt"):
            cres.metadata.setdefault("setup_callbacks", [])
            if _nrt_memsys_setup_callback not in cres.metadata["setup_callbacks"]:
                cres.metadata["setup_callbacks"].append(_nrt_memsys_setup_callback)

        # TODO: parse CC from the object and ensure it's not greater than the
        # greatest supported CC via _get_gpu_compute_capability()
        cres.metadata["func_name"] = generate_mangled_name(
            cres.fndesc.qualname, cres.fndesc.argtypes
        )

        cres.library._ptx = cres.metadata["ptx"]
        cres.library._mlir_str = cres.metadata["mlir_module_optimized"]
