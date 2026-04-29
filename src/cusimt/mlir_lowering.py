# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cusimt.descriptor import MLIRDispatcherType
from functools import lru_cache
from typeguard import typechecked
from io import StringIO
import logging
import operator
from typing import Any, Callable, Sequence
import numpy as np
from cusimt.numba_cuda import typing, utils
from cusimt.numba_cuda.core import targetconfig, errors
from cusimt.numba_cuda.core import ir as numba_ir
from cusimt.numba_cuda import dispatcher
from cusimt.numba_cuda.core import analysis
from cusimt.numba_cuda.extending import _Intrinsic

from cusimt.numba_cuda.core.environment import Environment
from cusimt.numba_cuda import types
from cusimt.annotations import Builder, AnyCallable, PS
from cusimt.errors import InternalCompilerError, ensure_verifies
from cusimt.lowering_utilities.type_conversions import to_mlir_type, to_numba_type
from cusimt.lowering_utilities import context as cusimt_context, constant
from cusimt.lowering_utilities import (
    DeferredLowering,
    convert,
    equal,
    index_of,
    int_of,
    i64_of,
    RangeObject,
    ArrayIterObject,
    UniTupleIterObject,
    IterResult,
    get_or_insert_function,
    user_signature_to_external_abi_signature,
    lookup_callee_in_module,
    get_func_type,
)
from cusimt.compiler import ExternFunction, ExternMLIRLibrary, ExternMLIRLibraryFunction

from cusimt._mlir import ir
from cusimt._mlir.ir import MemRefType, StringAttr, UnitAttr, TypeAttr, Block
from cusimt._mlir.extras import types as T
from cusimt._mlir.dialects import (
    llvm,
    arith,
    builtin,
    bufferization,
    cf,
    func,
    linalg,
    math,
    memref,
    scf,
    nvvm,
    gpu,
    tensor,
    shape,
)
from cusimt._mlir.dialects import complex as complex_dialect

from cusimt.logging import trace
from cusimt import mlir_debuginfo, mlir_lowering_registry
from cusimt.tools import (
    get_gpu_compute_capability,
    generate_mangled_name,
    parse_compute_capability,
    get_max_ptx_version,
)
from cusimt.linker import Linker
from cusimt.type_defs.aggregate_types import AggregateType, UnionType
from cusimt.types import Record

DEFAULT_SM_ARCH = get_gpu_compute_capability()

ERROR_CODE_GLOBAL_NAME = "__cusimt_error_code"
KERNEL_ERROR_CODES = {
    AssertionError: 1,
    IndexError: 2,
    ValueError: 3,
    RuntimeError: 4,
    ZeroDivisionError: 5,
}


def _is_omitted_arg(argty):
    return isinstance(argty, (types.Omitted, types.NoneType))


def _is_valid_memref_element_type(mlir_type: ir.Type) -> bool:
    """
    MemRef element types must be builtin/tensor scalars etc.; LLVM dialect
    types (structs used by MaskedType, pointers, llvm.array, ...) are invalid
    and must use llvm.alloca + llvm.load/store instead.
    """
    return not str(mlir_type).startswith("!llvm.")


def get_gpu_module_name():
    return ir.StringAttr.get("cusimt_gpu_module")


class MLIRLower(object):
    """
    Lower Numba IR to MLIR.
    """

    def __init__(self, context, fndesc, func_ir, metadata):
        trace("%s(%s) -> %s", fndesc.qualname, fndesc.argtypes, fndesc.restype)
        self.fndesc = fndesc
        self.blocks = utils.SortedMap(func_ir.blocks.items())
        self.func_ir = func_ir
        self.metadata = metadata
        self.flags = targetconfig.ConfigStack.top_or_none()
        self.targetoptions = self.metadata["targetoptions"]
        if "chip" in self.targetoptions:
            arch = self.targetoptions["chip"]
            cc = parse_compute_capability(arch)
            arch_suffix = arch.removeprefix(f"sm_{cc[0]}{cc[1]}")
            arch_specific_cc = (*cc, arch_suffix) if arch_suffix in ("a", "f") else cc
        else:
            cc = get_gpu_compute_capability(tuple)
            arch = get_gpu_compute_capability(str)
            arch_specific_cc = cc

        assert isinstance(cc, tuple)

        # The linker cubin arch must be >= the PTX target. For forward-compat
        # (e.g. sm_75 PTX on sm_100 host), link for the host. Otherwise
        # preserve the user's arch string (e.g. sm_100a) for the linker.
        host_cc = get_gpu_compute_capability(tuple)
        host_arch = get_gpu_compute_capability(str)
        if cc < host_cc:
            linker_cc = host_cc
            linker_arch = host_arch
        else:
            linker_cc = arch_specific_cc
            linker_arch = arch

        link_files = list(self.targetoptions.get("link", []))
        has_ltoir_files = any(
            (
                f.endswith(".ltoir")
                if isinstance(f, str)
                else type(f).__name__ == "LTOIR"
            )
            for f in link_files
        )
        needs_lto = (
            self.targetoptions.get("lto", False)
            or has_ltoir_files
            or self.targetoptions.get("output", "ptx") == "ltoir"
        )

        self._debug_full = self.targetoptions.get("debug", False)
        self._line_only = self.targetoptions.get("lineinfo", False)

        self._linker_config = dict(
            cc=linker_cc,
            arch=linker_arch,
            verbose=self.targetoptions.get("dump", False),
            optimization_level=int(self.targetoptions.get("opt_level", 3)),
            debug=self._debug_full,
            lineinfo=self._line_only,
            ptxas_options=self.targetoptions.get("ptxas_options", None),
            lto=needs_lto,
            max_registers=self.targetoptions.get("max_registers", None),
        )
        self._seen_mlir_libraries = set()
        self._cloned_device_funcs: set[str] = set()
        self._linked_external_items = set()
        self.linker = self._create_linker()

        # Collect module callbacks from LinkableCode objects (e.g. CUSource)
        # for invocation after the CUlibrary is loaded by C++.
        self._setup_callbacks = []
        self._teardown_callbacks = []
        for link_file in link_files:
            self._link_external_item(link_file)

        self._capi_sym_name = None
        if capi := self.targetoptions.get("capi", False):
            if isinstance(capi, str):
                self._capi_sym_name = capi
            else:
                self._capi_sym_name = func_ir.func_id.func_name
            trace("capi sym name: %s", self._capi_sym_name)

        # Initialize MLIR
        loc = self.get_loc(func_ir.loc)
        self.mlir_loc = ir.Location.file(
            filename=loc.filename,
            line=loc.line,
            col=loc.col,
            context=cusimt_context.get_context(),
        )
        self._di_subprogram = None
        self._di_builder = None
        if self._debug_full or self._line_only:
            ctx = cusimt_context.get_context()
            opt_level = int(self.targetoptions.get("opt_level", 3))
            self._di_builder = mlir_debuginfo.DIBuilder(
                loc,
                func_ir.func_id.func_name,
                line_only=self._line_only,
                opt=(opt_level > 0),
                context=ctx,
            )
            if self._di_builder.valid:
                if self._debug_full:
                    # Variable-level debug info only for full debug, not lineinfo-only
                    self._collect_debug_variables()
                self._di_subprogram = self._di_builder.build()
                self.mlir_loc = ir.Location.fused(
                    [self.mlir_loc], metadata=self._di_subprogram, context=ctx
                )

        # Python execution environment (will be available to the compiled
        # function).
        self.env = Environment.from_fndesc(self.fndesc)

        # Internal states
        self.blkmap = {}
        self.varmap = {}
        self.var_assign_count = {}  # {'var': count}
        self.user_defined_functions = {}
        self.firstblk = min(self.blocks.keys())
        self.loc = -1
        self.mlir_funcOp: gpu.GPUFuncOp | func.FuncOp | None = None
        self._mlir_module: ir.Module | None = None
        self._mlir_gpu_module: gpu.GPUModuleOp | None = None
        self._shared_memory_base: ir.Value | None = None
        self._total_shared_memory_bytes: ir.Value | None = None
        self._deferred_dbg_declare_vars: set[str] = set()
        self._debug_forced_alloca: set[str] = set()

        # Specializes the target context as seen inside the Lowerer
        # This adds:
        #  - environment: the python execution environment
        self.context = context.subtarget(environment=self.env, fndesc=self.fndesc)

    def _create_linker(self):
        return Linker(
            **self._linker_config,
            optimize_unused_variables=True,
        )

    def _collect_debug_variables(self):
        """Scan fndesc.args and fndesc.typemap to add debug variables to the DIBuilder."""
        typemap = self.fndesc.typemap
        func_line = getattr(self.func_ir.loc, "line", 0) or 0

        for arg_idx, arg_name in enumerate(self.fndesc.args):
            if arg_name in typemap:
                numba_type = typemap[arg_name]
                self._di_builder.add_local_variable(
                    arg_name,
                    func_line,
                    numba_type,
                    arg_index=arg_idx + 1,  # DWARF arg indices are 1-based
                )

        seen = set(self.fndesc.args)
        for var_name, numba_type in typemap.items():
            if var_name in seen or var_name.startswith("$") or "." in var_name:
                continue
            seen.add(var_name)
            var_loc = self._find_var_def_line(var_name)
            self._di_builder.add_local_variable(var_name, var_loc, numba_type)

    def _find_var_def_line(self, var_name):
        """Find the first assignment line for a variable."""
        for block in self.blocks.values():
            for inst in block.body:
                if isinstance(inst, numba_ir.Assign) and inst.target.name == var_name:
                    line = getattr(inst.loc, "line", 0)
                    if line:
                        return line
        return getattr(self.func_ir.loc, "line", 0) or 0

    def lower_to_mlir(self):
        token = cusimt_context.set_compilation_options(self.targetoptions)
        try:
            with cusimt_context.get_context(), self.mlir_loc:
                self._mlir_module = ir.Module.create()
                self.setup_func_op()
                self.lower_function_body()
                self.lower_capi_thunks()
                self._materialize_deferred_dbg_declare_attrs()
                assert self.metadata
                self.metadata["mlir_module"] = self.mlir_module
                # In debug/lineinfo mode, serialize with enable_debug_info so LLVM dialect
                # debug metadata (#llvm.di_*, loc) is not stripped from mlir_module_str.
                if self._di_subprogram is not None:
                    with StringIO() as sb:
                        self.mlir_module.operation.print(
                            enable_debug_info=True, file=sb
                        )
                        self.metadata["mlir_module_str"] = sb.getvalue()
                else:
                    self.metadata["mlir_module_str"] = str(self.mlir_module)
                self.metadata["linker"] = self.linker
                if self._setup_callbacks or self._teardown_callbacks:
                    self.metadata["setup_callbacks"] = self._setup_callbacks
                    self.metadata["teardown_callbacks"] = self._teardown_callbacks
        finally:
            cusimt_context._compilation_options.reset(token)

    @property
    def mlir_module(self) -> ir.Module:
        assert self._mlir_module is not None
        return self._mlir_module

    @property
    def mlir_gpu_module(self) -> gpu.GPUModuleOp:
        assert self._mlir_gpu_module is not None
        return self._mlir_gpu_module

    @property
    def _is_kernel(self) -> bool:
        return isinstance(self.mlir_funcOp, gpu.GPUFuncOp)

    def _get_or_create_error_global(self) -> ir.Value:
        """Get/create the error code global. Returns LLVM pointer to error code.

        Every compilation unit (kernel or device function) gets a real
        definition so that standalone PTX linking succeeds. When device
        function MLIR is later linked into a kernel module, duplicate
        globals are skipped by the linker.
        """
        if not hasattr(self, "_error_global_created"):
            gpu_block = self.mlir_gpu_module.bodyRegion.blocks[0]
            # Check if one already exists (e.g. linked from a prior
            # device function compilation).
            already_exists = any(
                isinstance(op, llvm.GlobalOp)
                and op.sym_name.value == ERROR_CODE_GLOBAL_NAME
                for op in gpu_block
            )
            if not already_exists:
                with ir.InsertionPoint.at_block_begin(gpu_block):
                    linkage = ir.Attribute.parse("#llvm.linkage<external>")
                    llvm.GlobalOp(
                        T.i32(),
                        ERROR_CODE_GLOBAL_NAME,
                        linkage,
                        value=ir.IntegerAttr.get(T.i32(), 0),
                        addr_space=0,
                    )
            self._error_global_created = True

        return llvm.mlir_addressof(llvm.PointerType.get(), ERROR_CODE_GLOBAL_NAME)

    def set_error_code(self, error_code: int | ir.Value):
        """Atomically set the error code global (first error wins).

        Emits a compare-and-swap on __cusimt_error_code. Works in both
        kernels and device functions.
        """
        error_ptr = self._get_or_create_error_global()
        from cusimt.lowering_utilities import convert

        if isinstance(error_code, ir.Value):
            error_code = convert(error_code, T.i32())
        else:
            error_code = llvm.ConstantOp(
                T.i32(), ir.IntegerAttr.get(T.i32(), error_code)
            ).result
        zero = llvm.ConstantOp(T.i32(), ir.IntegerAttr.get(T.i32(), 0)).result
        llvm.cmpxchg(
            error_ptr,
            zero,
            error_code,
            llvm.AtomicOrdering.monotonic,
            llvm.AtomicOrdering.monotonic,
        )

    def lower_capi_thunks(self):
        trace()
        if not self._capi_sym_name:
            return

        with ir.InsertionPoint(self.mlir_gpu_module.regions[0].blocks[-1]) as ip:
            clone = self.mlir_funcOp.clone()
            clone.attributes["sym_name"] = clone.attributes["cusimt.capi_name"]
            from cusimt.lowering_utilities.type_conversions import to_numba_type

            capi_type = to_numba_type(clone.function_type)
            from textwrap import dedent

            names = self.func_ir.arg_names
            self.metadata["capi"] = dedent(
                f"""
extern "C" __global__ void
{self._capi_sym_name}(
    {",\n    ".join([f"{argtype} {name}" for argtype, name in zip(capi_type.args, names)])}
)
""".replace(
                    "none*", "void*"
                )
            )

    def setup_func_op(self):
        # Setup MLIR funcOp
        logging.info(f"Setting up funcOp for {self.fndesc.qualname}")
        assert self.mlir_module is not None
        with ir.InsertionPoint(self.mlir_module.body) as ip:
            # Derive target attributes from user-provided targetoptions, with sensible defaults
            chip = 'chip = "' + self.targetoptions.get("chip", DEFAULT_SM_ARCH) + '"'
            opt_level = int(self.targetoptions.get("opt_level", 2))
            flags = []
            if self.targetoptions.get("fastmath", False):
                flags.extend(["fast"])
            features = self.targetoptions.get("features", "")
            if not features or "+ptx" not in features:
                ptx_ver = get_max_ptx_version()
                if ptx_ver is not None:
                    ptx_feat = f"+ptx{ptx_ver}"
                    features = f"{features},{ptx_feat}" if features else ptx_feat
            features = ', features = "' + features + '"' if features else ""
            flags_clause = f", flags = {{{', '.join(flags)}}}" if flags else ""
            target_attr = (
                "#nvvm.target<"
                + chip
                + flags_clause
                + ", O = "
                + str(opt_level)
                + features
                + ">"
            )
            targets = ir.ArrayAttr.get([ir.Attribute.parse(target_attr)])
            self._mlir_gpu_module = gpu.GPUModuleOp(
                targets=targets, sym_name=get_gpu_module_name()
            )
            self._mlir_gpu_module.attributes["cusimt.link_target"] = ir.UnitAttr.get()

        with ir.InsertionPoint(self.mlir_gpu_module.bodyRegion.blocks.append()):
            # arguments with default values are represented as Omitted type in Numba,
            # these arguments are ignored when construct the MLIR function signatures
            # ommited arguments will be initialized at the beginning of the MLIR function.
            #
            # Note: this approach is different from clang and other compilers, which
            # embedded default values at function callsite instead of initializing them
            # at the beginning of the callee function. The reason is simply because this
            # lowering class does not have a global view of the invoked user-defined function
            # due to the nature of jit compilation.
            non_omitted_argtypes = [
                argty for argty in self.fndesc.argtypes if not _is_omitted_arg(argty)
            ]
            # Flatten tuple types for MLIR function signature
            flat_argtypes = []
            for argty in non_omitted_argtypes:
                flat_argtypes.extend(self._flatten_type(argty))
            argtypes = [self.get_mlir_type(argtype) for argtype in flat_argtypes]
            flat_restypes = self._flatten_type(self.fndesc.restype)
            restypes = [self.get_mlir_type(rt) for rt in flat_restypes]
            if not restypes:
                mlir_funcOp_type = ir.FunctionType.get(argtypes, [])
            else:
                mlir_funcOp_type = ir.FunctionType.get(argtypes, restypes)

            # A function is a kernel when device=True is not set AND it returns
            # void.  Non-void functions are always device functions (kernels
            # cannot return values).
            kernel = not self.targetoptions.get("device", False) and not restypes

            abi_info = self.targetoptions.get("abi_info") or {}
            if abi_name := abi_info.get("abi_name"):
                sym_name = abi_name
            else:
                sym_name = generate_mangled_name(
                    self.fndesc.qualname,
                    [
                        argtype
                        for argtype in self.fndesc.argtypes
                        if not _is_omitted_arg(argtype)
                    ],
                )

            if kernel:
                self.mlir_funcOp = gpu.GPUFuncOp(
                    function_type=mlir_funcOp_type,
                    sym_name=sym_name,
                    loc=self.mlir_loc,
                    kernel=True,
                )
                # self.mlir_funcOp.attributes['nvvm.kernel'] = ir.UnitAttr.get()

                # Set launch bounds if specified
                launch_bounds = self.targetoptions.get("launch_bounds")
                if launch_bounds is not None:
                    if isinstance(launch_bounds, int):
                        max_threads = launch_bounds
                        min_blocks = None
                        max_cluster_rank = None
                    else:
                        max_threads = launch_bounds[0]
                        min_blocks = (
                            launch_bounds[1] if len(launch_bounds) > 1 else None
                        )
                        max_cluster_rank = (
                            launch_bounds[2] if len(launch_bounds) > 2 else None
                        )

                    # nvvm.maxntid specifies maximum threads per block (x, y, z)
                    self.mlir_funcOp.attributes["nvvm.maxntid"] = (
                        ir.DenseI32ArrayAttr.get([max_threads, 1, 1])
                    )

                    # nvvm.minctasm specifies minimum CTAs per SM (optional)
                    if min_blocks is not None:
                        self.mlir_funcOp.attributes["nvvm.minctasm"] = (
                            ir.IntegerAttr.get(T.i32(), min_blocks)
                        )

                    if max_cluster_rank is not None:
                        self.mlir_funcOp.attributes["nvvm.cluster_max_blocks"] = (
                            ir.IntegerAttr.get(T.i32(), max_cluster_rank)
                        )

                with ir.InsertionPoint(self.mlir_funcOp.add_entry_block()):
                    self._total_shared_memory_bytes = arith.constant(
                        result=T.index(), value=0
                    )
            else:
                self.mlir_funcOp = func.FuncOp(
                    name=sym_name,
                    type=mlir_funcOp_type,
                    loc=self.mlir_loc,
                )
                self.mlir_funcOp.add_entry_block()

            if kernel:
                self.mlir_funcOp.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()

            inline_strategy = self.targetoptions.get("inline", "always")
            if not kernel:
                if inline_strategy == "always":
                    self.mlir_funcOp.attributes["always_inline"] = ir.UnitAttr.get()
                elif inline_strategy == "never":
                    self.mlir_funcOp.attributes["no_inline"] = ir.UnitAttr.get()

            # TODO: this does not work how I expect. can't set byval or align on non-llvm-ptrs.
            arg_attrs = [dict() for i in range(len(mlir_funcOp_type.inputs))]
            for i, arg in enumerate(self.fndesc.argtypes):
                if attrs := getattr(arg, "__cusimt_attributes__", None):
                    for key, value in attrs.items():
                        match value:
                            case bool():
                                if value:
                                    arg_attrs[i][key] = ir.UnitAttr.get()
                            case int():
                                arg_attrs[i][key] = ir.IntegerAttr.get(T.i64(), value)
                            case types.Type():
                                arg_attrs[i][key] = ir.TypeAttr.get(to_mlir_type(value))
                            case _:
                                raise NotImplementedError(
                                    f"Not implemented for type {type(value)}"
                                )
            self.mlir_funcOp.attributes["cusimt.orig_arg_types"] = ir.ArrayAttr.get(
                [
                    ir.TypeAttr.get(i)
                    for i in self.mlir_funcOp.function_type.value.inputs
                ]
            )
            self.mlir_funcOp.attributes["cusimt.arg_attrs"] = ir.ArrayAttr.get(
                [ir.DictAttr.get(arg_attr) for arg_attr in arg_attrs]
            )
            if self._capi_sym_name:
                self.mlir_funcOp.attributes["cusimt.capi_name"] = ir.StringAttr.get(
                    self._capi_sym_name
                )

    @typechecked
    def alloca(self, ty: ir.Type, count: int = 1) -> ir.Value:
        with self.alloca_insertion_point():
            count_value = i64_of(count)
            return ir.Value(llvm.alloca(llvm.PointerType.get(), count_value, ty))

    def alloca_insertion_point(self):
        assert self.mlir_funcOp is not None
        try:
            ip = ir.InsertionPoint.at_block_terminator(self.mlir_funcOp.entry_block)
        except ValueError as e:
            # block does not have a termiator yet... I don't see a better way to put an operation
            # at the end of the block if it has no terminator, or before the terminator if it has one.
            # :(
            ip = ir.InsertionPoint(self.mlir_funcOp.entry_block)
        return ip

    def func_op_insertion_point(self):
        return ir.InsertionPoint(self.mlir_module.body)

    def lower_function_body(self):
        """
        Lower the function body into MLIR.
        """
        trace()
        assert self.mlir_funcOp is not None
        # Create a one-on-one mapping from numba block to mlir block
        for offset in self.blocks:
            if offset == self.firstblk:
                self.blkmap[offset] = self.mlir_funcOp.entry_block
            else:
                self.blkmap[offset] = self.mlir_funcOp.body.blocks.append()

        # analysis to collect the number of times a variable gets assigned
        # variable gets assigned more than once will be spilled on stack
        # this is to deal with the issue that numba IR is not in SSA form
        local_var_assign_count = {}
        self.collect_var_assign_count(
            self.func_ir.blocks.values(), local_var_assign_count
        )

        for var_name, count in local_var_assign_count.items():
            assert (
                var_name not in self.var_assign_count
            ), f"{var_name} already in var_assign_count"
            self.var_assign_count[var_name] = count

        with ir.InsertionPoint(self.blkmap[self.firstblk]):
            self.allocate_stack_space_for_vars_with_multiple_assigns(
                local_var_assign_count
            )

        self.cfg = analysis.compute_cfg_from_blocks(self.blocks)
        self.usedefs = analysis.compute_use_defs(self.blocks)
        self.live_map = analysis.compute_live_map(
            self.cfg, self.blocks, self.usedefs.usemap, self.usedefs.defmap
        )
        self.dead_map = analysis.compute_dead_maps(
            self.cfg, self.blocks, self.live_map, self.usedefs.defmap
        )

        # Lower all blocks
        for offset, block in sorted(self.blocks.items()):
            self.current_offset = offset
            self.lower_block(offset, block)

    def collect_var_assign_count(self, blocks, var_assign_count):
        trace()
        for block in blocks:
            for inst in block.body:

                def collect_var_assign_count_from_inst(inst):
                    if isinstance(inst, numba_ir.Assign):
                        var = inst.target
                        if var.name in var_assign_count:
                            var_assign_count[var.name] += 1
                        else:
                            var_assign_count[var.name] = 1
                    elif isinstance(inst, numba_ir.SetAttr):
                        # SetAttr on structs counts as a reassignment of the target variable
                        var = inst.target
                        if var.name in var_assign_count:
                            var_assign_count[var.name] += 1
                        else:
                            var_assign_count[var.name] = 1
                    else:
                        pass

                collect_var_assign_count_from_inst(inst)

    def allocate_stack_space_for_vars_with_multiple_assigns(self, var_assign_count):
        trace()
        for var_name, count in var_assign_count.items():
            if count > 1:
                var_type = self.get_numba_type(var_name)
                if isinstance(var_type, types.NoneType):
                    continue
                mlir_type = self.get_mlir_type(var_type)

                if not _is_valid_memref_element_type(mlir_type):
                    self.varmap[var_name] = self.alloca(mlir_type, count=1)
                    trace(
                        f"Allocated LLVM stack space for {type(var_type).__name__} "
                        f"variable {var_name} (mlir type {mlir_type})"
                    )
                else:
                    memref_type = ir.MemRefType.get(shape=[1], element_type=mlir_type)
                    self.varmap[var_name] = memref.alloca(
                        memref=memref_type, dynamic_sizes=[], symbol_operands=[]
                    )
                    self._tag_alloca_for_deferred_dbg_declare(
                        var_name, self.varmap[var_name]
                    )

    @staticmethod
    def _canonical_dbg_var_name(var_name: str) -> str:
        """Normalize Numba SSA names (foo.1 -> foo) for debug metadata lookup."""
        return var_name.split(".")[0] if "." in var_name else var_name

    def _get_numba_type_for_dbg_var(self, var_name: str):
        """Resolve typemap entries for SSA-renamed variable names."""
        base_name = self._canonical_dbg_var_name(var_name)
        return self.fndesc.typemap.get(var_name) or self.fndesc.typemap.get(base_name)

    def _tag_alloca_for_deferred_dbg_declare(self, var_name, alloca_op):
        """Attach debug metadata to allocas for deferred dbg.declare emission.

        Tag the memref.alloca with a NameLoc: ``dbg_var:<name>``. A deferred pass in
        ``mlir_optimization.py`` consumes this tag after base_pipeline.
        """
        if (
            not self._debug_full
            or self._di_builder is None
            or not self._di_builder.valid
            or var_name.startswith("$")
        ):
            return

        base_name = self._canonical_dbg_var_name(var_name)
        numba_type = self._get_numba_type_for_dbg_var(var_name)
        if not isinstance(numba_type, types.Complex):
            return

        var_attr = self._di_builder.di_local_vars.get(base_name)
        if var_attr is None:
            return

        # Remember this variable so lower_to_mlir() can serialize module-local attrs.
        self._deferred_dbg_declare_vars.add(base_name)
        name_loc = ir.Location.name(f"dbg_var:{base_name}")
        op = alloca_op if isinstance(alloca_op, ir.Operation) else alloca_op.owner
        op.location = ir.Location.fused([op.location, name_loc])

    def _materialize_deferred_dbg_declare_attrs(self):
        """Persist deferred DI attrs into the serialized MLIR module.

        optimize() reparses ``mlir_module_str`` and runs deferred dbg.declare
        emission in that parsed module, so these attrs are intentionally retained
        for the lowering->optimize handoff and consumed/stripped in
        ``mlir_optimization._emit_deferred_dbg_declares`` after emission.
        """
        if (
            not self._deferred_dbg_declare_vars
            or not self._debug_full
            or self._di_builder is None
            or not self._di_builder.valid
        ):
            return

        attrs = self.mlir_module.operation.attributes
        attrs["metadata.dbg_declare_expr"] = self._di_builder.di_expression
        for base_name in sorted(self._deferred_dbg_declare_vars):
            var_attr = self._di_builder.di_local_vars.get(base_name)
            if var_attr is None:
                continue
            attrs[f"metadata.dbg_declare_var_{base_name}"] = var_attr

    def lower_block(self, offset, block):
        """
        Lower the given numba block into MLIR.
        """
        trace("offset=%s, block=%s", offset, block)
        with ir.InsertionPoint(self.blkmap[offset]):
            for inst in block.body:
                if self._di_subprogram is not None:
                    # Propagate location info into MLIR per instruction.
                    self.loc = self.get_loc(inst.loc)
                    inst_loc = ir.Location.file(
                        filename=self.loc.filename,
                        line=self.loc.line,
                        col=self.loc.col,
                    )
                    with inst_loc:
                        self.lower_inst(inst)
                else:
                    self.lower_inst(inst)

    def lower_inst(self, inst):
        trace(inst)
        if isinstance(inst, numba_ir.Assign):
            self.lower_assign(inst)
        elif isinstance(inst, (numba_ir.SetItem, numba_ir.StaticSetItem)):
            self.lower_setitem(inst)
        elif isinstance(inst, numba_ir.SetAttr):
            self.lower_setattr(inst)
        elif isinstance(inst, numba_ir.Branch):
            self.lower_branch(inst)
        elif isinstance(inst, numba_ir.Jump):
            self.lower_jump(inst)
        elif isinstance(inst, numba_ir.Del):
            self.lower_del(inst)
        elif isinstance(inst, numba_ir.Return):
            self.lower_return(inst)
        elif isinstance(inst, numba_ir.StaticRaise):
            self.lower_static_raise(inst)
        elif isinstance(inst, numba_ir.Print):
            self.lower_print(inst.args)
        else:
            raise NotImplementedError(
                f"NotImplemented lowering {inst} of type {type(inst)}."
            )

    def lower_print(self, args):
        trace()

        builder = self.get_registered_builder(print, types.void())
        if builder is None:
            raise InternalCompilerError("Print lowering is not implemented!")
        builder(self, None, args, [])

    def lower_static_raise(self, static_raise_inst):
        trace(static_raise_inst)

        exc_class = static_raise_inst.exc_class
        error_code = KERNEL_ERROR_CODES.get(exc_class, 4)  # Default to RuntimeError
        self.set_error_code(error_code)
        return_block = self.blkmap[max(self.blkmap.keys())]
        cf.br([], return_block)

    def lower_assign(self, assign_inst):
        """
        target = value
        """
        target = assign_inst.target
        value = assign_inst.value
        trace("target: %s, value: %s", target, value)
        if isinstance(value, numba_ir.Arg):
            self.lower_arg_assign(target, value)
        elif isinstance(value, numba_ir.Const):
            self.lower_const_assign(target, value)
        elif isinstance(value, numba_ir.Global):
            self.lower_global_assign(target, value)
        elif isinstance(value, numba_ir.Var):
            self.lower_var_assign(target, value)
        elif isinstance(value, numba_ir.Expr):
            self.lower_expr_assign(target, value)
        elif isinstance(value, numba_ir.FreeVar):
            self.lower_free_var_assign(target, value)
        else:
            raise NotImplementedError(
                f"NotImplemented lowering assign value {value} of type {type(value)}"
            )

    def lower_free_var_assign(self, target, free_var):
        trace("target: %s, free_var: %s", target, free_var)
        if free_var.value is None:
            target_type = self.get_numba_type(target.name)
            self.store_var(target, self.get_mlir_type(target_type))
        elif hasattr(free_var.value, "__cuda_array_interface__"):
            self.lower_captured_array_to_memref(target, free_var.value)
        else:
            self.store_var(target, free_var.value)

    def lower_captured_array_to_memref(self, target, pyval):
        """Build an MLIR memref value from __cuda_array_interface__ metadata.

        Bakes the captured device array's pointer, shape, and strides as constants
        into a memref descriptor struct, then casts to the target memref type.
        Registers ``pyval`` on the active code library to keep the underlying
        GPU memory alive while the compiled kernel exists.
        """
        self.context.active_code_library.referenced_objects[id(pyval)] = pyval
        array = pyval.__cuda_array_interface__
        memref_type = self.get_mlir_type(self.get_numba_type(target.name))
        shape = array["shape"]
        ndim = len(shape)
        itemsize = np.dtype(array["typestr"]).itemsize

        byte_strides = array.get("strides")
        if byte_strides:
            strides = [s // itemsize for s in byte_strides]
        else:
            strides, s = [], 1
            for d in reversed(shape):
                strides.insert(0, s)
                s *= d

        if ndim > 0:
            struct_type = ir.Type.parse(
                f"!llvm.struct<(ptr, ptr, i64, array<{ndim} x i64>, array<{ndim} x i64>)>"
            )
        else:
            struct_type = ir.Type.parse("!llvm.struct<(ptr, ptr, i64)>")
        ptr = llvm.inttoptr(
            llvm.PointerType.get(), arith.constant(T.i64(), array["data"][0])
        )
        i64c = lambda v: arith.constant(T.i64(), v)
        ins = lambda d, v, *p: llvm.insertvalue(
            container=d, value=v, position=ir.DenseI64ArrayAttr.get(list(p))
        )

        desc = llvm.UndefOp(struct_type).result
        desc = ins(desc, ptr, 0)
        desc = ins(desc, ptr, 1)
        desc = ins(desc, i64c(0), 2)
        for i, s in enumerate(shape):
            desc = ins(desc, i64c(s), 3, i)
        for i, s in enumerate(strides):
            desc = ins(desc, i64c(s), 4, i)

        self.store_var(
            target, builtin.unrealized_conversion_cast([memref_type], [desc])
        )

    def lower_arg_assign(self, target, arg):
        trace()
        # the arg is a default argument, which is of Omitted numba type
        # create a MLIR constantOp to store the default value.
        # The Numba type info can be retrieved from the typemap
        #
        # For example, if we have a Numba IR instruction:
        #       x = arg(0, name=x)
        # and arg is a default argument, then there is no argument found
        # in the MLIR function argument list. However, the typemap contains
        #       {'arg.x': omitted(default=1.0), 'x': float64}
        # which can be used to construct a MLIR constant operation.
        if _is_omitted_arg(self.get_numba_type("arg." + target.name)):
            arg_type = self.get_numba_type("arg." + target.name)
            target_type = self.get_numba_type(target.name)
            if isinstance(arg_type, types.NoneType):
                self.store_var(target, self.get_mlir_type(target_type))
            elif isinstance(target_type, types.NumberClass):
                # This is the case for passing in argument representing numpy data types
                # For example np.int32
                # The default value constructed is 0 of MLIR type int32
                dtype = target_type.dtype
                dtype_default_const = arith.constant(
                    result=self.get_mlir_type(dtype),
                    value=0.0 if isinstance(dtype, types.Float) else 0,
                )
                self.store_var(target, dtype_default_const)
            else:
                self.store_var(target, arg_type.value)
        # the arg is not a default argument, which is an MLIR function argument
        else:
            arg_type = self.fndesc.argtypes[arg.index]

            if isinstance(arg_type, types.NoneType):
                # NoneType args are excluded from the MLIR function signature;
                # store the MLIR NoneType as a placeholder
                self.store_var(target, ir.NoneType.get())
            elif isinstance(arg_type, types.BaseTuple):
                flat_start_idx = self._get_flat_arg_start_index(arg.index)
                # Reassemble tuple from flattened block arguments
                value = self._reassemble_tuple_from_block_args(arg_type, flat_start_idx)
                self.store_var(target, value)
            else:
                flat_start_idx = self._get_flat_arg_start_index(arg.index)
                # Single argument - get the block argument at the flat index
                block_arg = self.mlir_funcOp.entry_block.arguments[flat_start_idx]
                self.store_var(target, block_arg)

    def lower_const_assign(self, target, const):
        trace()
        target_type = self.get_numba_type(target.name)
        if const.value is None:
            # for NoneType const, there is no 1-on-1 mapping between
            # numba instruction and MLIR op. Therefore, we register the
            # MLIR type in the varmap to unblock lowering of other
            # numba instruction (i.e, assign this NoneType variable to other variables)
            self.store_var(target, self.get_mlir_type(target_type))
        elif isinstance(const.value, (bool, int, float, np.number)):
            value = const.value
            mlir_type = self.get_mlir_type(target_type)

            # Check if the MLIR type supports constants (arith.constant)
            # Only integer and float types can have constants - not pointers, memrefs, etc.
            if not isinstance(
                mlir_type,
                (ir.IntegerType, ir.IndexType, ir.FloatType, ir.F16Type, ir.BF16Type),
            ):
                raise TypeError(
                    f"Cannot create constant of type {mlir_type} for target {target.name} "
                    f"(Numba type: {target_type}). Constants are only supported for "
                    f"integer and float types."
                )

            # Convert numpy scalars to Python primitives for MLIR
            if isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            elif isinstance(value, np.bool_):
                value = bool(value)

            # For large unsigned integers, convert to signed representation for MLIR
            # MLIR integers are signless, but IntegerAttr expects values in signed range
            if isinstance(value, int) and isinstance(target_type, types.Integer):
                if not target_type.signed and value >= (
                    1 << (target_type.bitwidth - 1)
                ):
                    # Convert unsigned to signed two's complement representation
                    value = value - (1 << target_type.bitwidth)

            self.store_var(
                target,
                arith.constant(mlir_type, value),
            )
        elif isinstance(const.value, complex):
            from cusimt.lowering.math import complex_cg

            complex_cg(self, target, [const.value.real, const.value.imag], {})
        elif isinstance(const.value, tuple):
            memref_allocaOp = memref.alloca(
                memref=self.get_mlir_type(target_type),
                dynamic_sizes=[],
                symbol_operands=[],
            )
            for i in range(len(const.value)):
                memref_index = arith.constant(result=ir.IndexType.get(), value=i)
                value = arith.constant(
                    result=self.get_mlir_type(target_type.dtype), value=const.value[i]
                )
                memref.store(
                    value=value,
                    memref=memref_allocaOp,
                    indices=[memref_index],
                )
            self.store_var(target, memref_allocaOp)
        elif isinstance(const.value, (str, bytes)):
            self.store_var(target, const.value)
        else:
            raise NotImplementedError(
                f"NotImplemented lowering const assignment {const}"
            )

    def lower_global_assign(self, target, glob):
        trace()
        target_type = self.get_numba_type(target.name)
        if glob.value is None:
            self.store_var(target, self.get_mlir_type(target_type))
            return
        if isinstance(glob.value, (bool, int, float, np.number)):
            mlir_type = self.get_mlir_type(target_type)
            # Check if the MLIR type supports constants
            if not isinstance(
                mlir_type,
                (ir.IntegerType, ir.IndexType, ir.FloatType, ir.F16Type, ir.BF16Type),
            ):
                raise TypeError(
                    f"Cannot create constant of type {mlir_type} for global {target.name} "
                    f"(Numba type: {target_type}, value: {glob.value}). "
                    f"Constants are only supported for integer and float types."
                )
            # Convert numpy scalars to Python primitives for MLIR
            value = glob.value
            if isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, np.floating):
                value = float(value)
            elif isinstance(value, np.bool_):
                value = bool(value)
            self.store_var(
                target,
                arith.constant(mlir_type, value),
            )
        elif hasattr(glob.value, "__cuda_array_interface__"):
            self.lower_captured_array_to_memref(target, glob.value)
        else:
            self.store_var(target, glob.value)

    def lower_var_assign(self, target, var):
        trace()
        assert self.var_lowered(var), f"Var {var.name} not found in varmap."
        var_value = self.load_var(var)

        match var_value:
            case (
                RangeObject()
                | ArrayIterObject()
                | UniTupleIterObject()
                | IterResult()
                | tuple()
            ):
                self.store_var(target, var_value)
            case ir.NoneType():
                self.store_var(target, var_value)
            case str() | bytes():
                self.store_var(target, var_value)
            case None:
                target_type = self.get_numba_type(target.name)
                self.store_var(target, self.get_mlir_type(target_type))
            case _:
                source_type = self.get_numba_type(var.name)
                target_type = self.get_numba_type(target.name)
                if source_type == target_type:
                    if isinstance(var_value, ir.Value):
                        value_op = self.mlir_convert(
                            var_value, self.get_mlir_type(target_type)
                        )
                        self.store_var(target, value_op)
                    else:
                        self.store_var(target, var_value)
                else:
                    value_op = self.lower_cast(source_type, target_type, var_value)
                    self.store_var(target, value_op)

    def lower_array_literal(self, value: np.ndarray) -> ir.Value:
        from cusimt.lowering_utilities import tensor_to_memref

        with self.alloca_insertion_point():
            dtype = to_mlir_type(value.dtype)
            raveled = value.ravel()
            elems = [self.lower_literal_if_needed(e) for e in raveled]
            mr_type = T.tensor(*value.shape, element_type=dtype)
            mr = tensor.from_elements(mr_type, elems)
            mr = tensor_to_memref(mr)
            return mr

    def lower_literal_if_needed(
        self, value: ir.Value | np.ndarray, numba_type=None
    ) -> ir.Value:
        match value:
            case tuple():
                return tuple(map(self.lower_literal_if_needed, value))
            case np.ndarray():
                return self.lower_array_literal(value)
            case np.number():
                mlir_type = to_mlir_type(value.dtype)
                cst = constant(value, mlir_type)
                return cst
            case bool():
                # Python bool literal - convert to MLIR i1
                return arith.constant(result=T.bool(), value=value)
            case int():
                # Python int literal - convert to MLIR i64
                return arith.constant(result=T.i64(), value=value)
            case float():
                # Python float literal - convert to MLIR f64
                return arith.constant(result=T.f64(), value=value)
            case str():
                from cusimt.lowering_utilities.string import (
                    materialize_string_constant,
                )

                return materialize_string_constant(self.mlir_gpu_module, value)
            case _:
                return value

    def lower_expr_assign(self, target, expr):
        trace("target=%s", target)
        trace("expr=%s", expr)
        match expr.op:
            case "binop" | "inplace_binop":
                self.lower_binop_expr_assign(target, expr.fn, expr.lhs, expr.rhs)
            case "unary":
                self.lower_unary_expr_assign(target, expr.fn, expr.value)
            case "cast":
                self.lower_cast_expr_assign(target, expr.value)
            case "getitem" | "typed_getitem":
                self.lower_getitem_expr_assign(target, expr.value, expr.index)
            case "static_getitem":
                # For static_getitem, use index_var if available, otherwise use the constant index
                index = expr.index_var if expr.index_var is not None else expr.index
                self.lower_getitem_expr_assign(target, expr.value, index)
            case "call":
                args = list(expr.args)
                # Handle vararg (*args) - expand tuple into individual arguments
                if expr.vararg is not None:
                    vararg_type = self.get_numba_type(expr.vararg.name)
                    if isinstance(vararg_type, types.BaseTuple):
                        # Create synthetic variables for each tuple element
                        for i in range(len(vararg_type)):
                            # Get the element from the tuple using static_getitem
                            elem_var = numba_ir.Var(
                                scope=target.scope,
                                name=f"$vararg_{expr.vararg.name}_{i}",
                                loc=target.loc,
                            )
                            elem_type = vararg_type[i]
                            self.fndesc.typemap[elem_var.name] = elem_type
                            # Lower the static_getitem to extract tuple element
                            self.lower_getitem_expr_assign(elem_var, expr.vararg, i)
                            args.append(elem_var)
                self.lower_call_expr_assign(target, expr.func, args, expr.kws, expr)
            case "arrayexpr":
                self.lower_arrayexpr_assign(target, expr.expr[0], expr.expr[1])
            case "build_tuple":
                self.lower_build_tuple_expr_assign(target, expr.items)
            case "getattr":
                self.lower_getattr_assign(target, expr.value, expr.attr)
            case "getiter":
                self.lower_getiter_expr_assign(target, expr.value)
            case "iternext":
                self.lower_iternext_expr_assign(target, expr.value)
            case "pair_first":
                self.lower_pair_first_expr_assign(target, expr.value)
            case "pair_second":
                self.lower_pair_second_expr_assign(target, expr.value)
            case "exhaust_iter":
                self.lower_exhaust_iter_expr_assign(target, expr.value, expr.count)
            case "null":
                # null() represents None - store a dummy value for none-typed targets
                self.lower_null_expr_assign(target)
            case _:
                raise NotImplementedError(
                    f"NotImplemented lowering expression assignment {expr}."
                )

    def lower_exhaust_iter_expr_assign(self, target, value, count):
        """
        Lower exhaust_iter expression used for tuple unpacking.
        The value should already be a tuple (memref), so we just forward it.
        The actual unpacking happens with subsequent static_getitem operations.
        """
        trace()
        assert self.var_lowered(value), f"Value {value.name} not found in varmap."
        # Simply forward the tuple value - the unpacking will be done by static_getitem
        self.store_var(target, self.load_var(value))

    def lower_null_expr_assign(self, target):
        """
        Lower null() expression which represents None or uninitialized values.
        For stack-allocated variables, we store a zero/default value.
        For register variables, we store the MLIR type as a placeholder.
        """
        trace()
        target_type = self.get_numba_type(target.name)
        mlir_type = self.get_mlir_type(target_type)

        if self.var_lowered(target):
            # Stack-allocated variable needs an actual value
            # Create a default/zero value of the appropriate type
            match mlir_type:
                case ir.IntegerType() | ir.IndexType():
                    default_value = arith.constant(mlir_type, 0)
                    self.store_var(target, default_value)
                case ir.FloatType() | ir.F16Type() | ir.BF16Type():
                    default_value = arith.constant(mlir_type, 0.0)
                    self.store_var(target, default_value)
                case _:
                    # For other types (e.g., arrays, structs, Records/llvm.ptr), skip initialization
                    # The variable will be assigned a proper value later
                    pass
        else:
            # Register variable - store type as placeholder
            self.store_var(target, mlir_type)

    def lower_pair_second_expr_assign(self, target, value):
        """
        Extract the loop condition (is_valid) from the (value, is_valid) tuple returned by iternext.
        """
        trace()
        iter_result = self.load_var(value)
        if isinstance(iter_result, IterResult):
            elem1 = self.mlir_convert(iter_result.is_valid, T.bool())
            self.store_var(target, elem1)
        else:
            elem1 = memref.load(memref=iter_result, indices=[index_of(1)])
            elem1 = self.mlir_convert(elem1, T.bool())
            self.store_var(target, elem1)

    def mlir_convert(self, value: ir.Value, target_type: ir.Type) -> ir.Value:
        trace("value=%s target_type=%s", value, target_type)
        return convert(value, target_type)

    def lower_pair_first_expr_assign(self, target, value):
        """
        Extract the yielded value from the (value, is_valid) tuple returned by iternext.
        """
        trace()
        iter_result = self.load_var(value)
        if isinstance(iter_result, IterResult):
            self.store_var(target, iter_result.value)
        else:
            elem0 = memref.load(memref=iter_result, indices=[index_of(0)])
            self.store_var(target, elem0)

    def lower_iternext_expr_assign(self, target, value):
        trace()
        iter_obj = self.load_var(value)
        if isinstance(iter_obj, RangeObject):
            iternext = iter_obj.next()
            self.store_var(target, iternext)
        elif isinstance(iter_obj, (ArrayIterObject, UniTupleIterObject)):
            iternext = iter_obj.next()
            self.store_var(target, iternext)
        else:
            raise InternalCompilerError(
                f"Unsupported iterator object for value {value.name}: {type(iter_obj)}"
            )

    def lower_getiter_expr_assign(self, target, value):
        trace()
        value_type = self.get_numba_type(value.name)

        if isinstance(value_type, types.RangeType):
            ro = self.load_var(value)
            if not isinstance(ro, RangeObject):
                raise InternalCompilerError(
                    f"Range object not found for value {value.name}"
                )
            self.store_var(target, ro)
        elif isinstance(value_type, types.Array) and value_type.ndim == 1:
            array = self.load_var(value)
            element_type = self.get_mlir_type(value_type.dtype)
            length = memref.dim(array, index_of(0))
            aio = ArrayIterObject(self, array, element_type, length)
            self.store_var(target, aio)
        elif isinstance(value_type, types.UniTuple):
            tup = self.load_var(value)
            tuple_memref = self.concretize_tuple(tup)
            element_type = self.get_mlir_type(value_type.dtype)
            utio = UniTupleIterObject(
                self, tuple_memref, value_type.count, element_type
            )
            self.store_var(target, utio)
        elif isinstance(value_type, types.Tuple):
            # Handle Tuple with IntegerLiteral elements (e.g., for stride in (64, 32, 16, ...))
            all_literals = all(
                isinstance(t, (types.Literal, types.IntegerLiteral))
                for t in value_type.types
            )
            if all_literals:
                # Extract literal values from the type
                literal_values = [t.literal_value for t in value_type.types]
                # Use i32 if all values fit, otherwise i64
                max_val = max(abs(v) for v in literal_values)
                if max_val <= 2**31 - 1:
                    element_type = T.i32()
                else:
                    element_type = T.i64()
                # Create memref AND store constants at entry block
                with self.alloca_insertion_point():
                    mr_type = T.memref(value_type.count, element_type=element_type)
                    mr = memref.alloca(
                        memref=mr_type, dynamic_sizes=[], symbol_operands=[]
                    )
                    for i, lit_val in enumerate(literal_values):
                        const = arith.constant(result=element_type, value=lit_val)
                        memref.store(value=const, memref=mr, indices=[index_of(i)])
                # Create iterator object
                utio = UniTupleIterObject(self, mr, value_type.count, element_type)
                self.store_var(target, utio)
            else:
                raise InternalCompilerError(
                    f"Unsupported iterator type: {value_type} (non-literal tuple elements)"
                )
        else:
            raise InternalCompilerError(f"Unsupported iterator type: {value_type}")

    def lower_binop_expr_assign(self, target, fn, lhs, rhs):
        trace("target=%s fn=%s", target, fn)
        trace("lhs=%s rhs=%s", lhs, rhs)
        assert self.var_lowered(lhs), f"LHS var {lhs.name} not found in varmap."
        assert self.var_lowered(rhs), f"RHS var {rhs.name} not found in varmap."

        # target type is used to determined the type of both lhs and rhs operands
        target_type = self.get_numba_type(target.name)
        lhs_type = self.get_numba_type(lhs.name)
        rhs_type = self.get_numba_type(rhs.name)
        if builder := self.get_registered_builder(fn, target_type(lhs_type, rhs_type)):
            builder(self, target, [lhs, rhs], [])
            return
        raise NotImplementedError(f"NotImplemented lowering binop {fn=}")

    def lower_unary_expr_assign(self, target, fn, operand):
        trace("target: %s, fn: %s, operand: %s", target, fn, operand)
        assert self.var_lowered(operand), f"Operand {operand} not found in varmap."

        # target type is used to determined the type of both lhs and rhs operands
        target_type = self.get_numba_type(target.name)
        operand_type = self.get_numba_type(operand.name)

        if builder := self.get_registered_builder(fn, target_type(operand_type)):
            builder(self, target, [operand], [])
            return

        raise NotImplementedError(f"NotImplemented lowering unary {fn=}")

    def lower_cast_expr_assign(self, target, value):
        trace("target: %s, value: %s", target, value)
        assert self.var_lowered(value), f"Cast var {value.name} not found in varmap."
        target_type = self.get_numba_type(target)
        value_type = self.get_numba_type(value)
        mlir_value = self.load_var(value)
        if target_type != value_type:
            mlir_value = self.lower_cast(value_type, target_type, mlir_value)
        self.store_var(target, mlir_value)

    def lower_getitem_expr_assign(self, target, value, index):
        """
        target = getitem(value=.., index=..)
        index can be either a variable or an integer constant
        """
        trace("target: %s, value: %s, index: %s", target, value, index)

        target_type = self.get_numba_type(target.name)
        value_type = self.get_numba_type(value.name)
        match index:
            case int():
                index_type = types.int64
            case str():
                index_type = types.StringLiteral(index)
            case numba_ir.Var():
                index_type = self.get_numba_type(index.name)
            case _:
                raise NotImplementedError(
                    f"lowering getitem {target} = {value}[{index}]"
                )
        signature = target_type(value_type, index_type)
        if builder := self.get_registered_builder(operator.getitem, signature):
            builder(self, target, [value, index], [])
            return
        if builder := self.get_registered_builder("static_getitem", signature):
            builder(self, target, [value, index], [])
            return

        raise NotImplementedError(f"lowering getitem {target} = {value}[{index}]")

    def build_user_defined_function_call(
        self,
        target: numba_ir.Var,
        fn: dispatcher._DispatcherBase,
        args: list[numba_ir.Var],
        kws: list[tuple[str, numba_ir.Var]] = [],
    ):
        target_type = self.get_numba_type(target.name)

        # TODO: Here named arguments are simply considered as positional arguments
        argtypes = tuple(
            [self.get_numba_type(arg.name) for arg in args]
            + [self.get_numba_type(value.name) for (name, value) in kws]
        )
        func_name = generate_mangled_name(fn.py_func.__qualname__, argtypes)

        from cusimt.descriptor import MLIRDispatcher

        # Numba CPU @jit dispatchers (e.g. xoroshiro128p random functions) can appear
        # as device function calls inside CUDA kernels. Re-wrap through
        # cusimt's pipeline so we get a proper MLIR compile result.
        if not isinstance(fn, MLIRDispatcher):
            from cusimt import cuda

            if not hasattr(fn, "_cusimt_device_dispatcher"):
                fn._cusimt_device_dispatcher = cuda.jit(device=True)(fn.py_func)
            fn = fn._cusimt_device_dispatcher
        elif not fn.targetoptions.get("device", False):
            # Ensure non-device dispatchers are recompiled as device functions.
            if not hasattr(fn, "_device_dispatcher"):
                opts = fn.targetoptions.copy()
                opts["device"] = True
                fn._device_dispatcher = MLIRDispatcher(fn.py_func, targetoptions=opts)
            fn = fn._device_dispatcher

        cres = fn.compile(argtypes)

        if callee_linker := cres.metadata.get("linker"):
            self.linker.merge_ltoirs_from(callee_linker)

        module: ir.Module = ir.Module.parse(cres.metadata["mlir_module_str"])
        gpu_module: gpu.GPUModuleOp = list(module.body)[0]
        gpu_module_ops = list(gpu_module.regions[0].blocks[0].operations)
        linkable_ops = [
            op
            for op in gpu_module_ops
            if isinstance(op, (gpu.GPUFuncOp, func.FuncOp, llvm.GlobalOp))
        ]
        assert self.mlir_gpu_module
        insertion_block: ir.Block = self.mlir_gpu_module.regions[0].blocks[0]

        # Collect names already in the target block (from prior compilations
        # or from _get_or_create_error_global).
        from cusimt.lowering_utilities.link import _get_op_sym_name

        existing_names = set(self._cloned_device_funcs)
        for existing_op in insertion_block:
            if n := _get_op_sym_name(existing_op):
                existing_names.add(n)

        callee: gpu.GPUFuncOp | None = None
        with ir.InsertionPoint(insertion_block):
            for op in linkable_ops:
                sym_name = _get_op_sym_name(op)
                if isinstance(op, (gpu.GPUFuncOp, func.FuncOp)):
                    if op.name.value == func_name:
                        callee = op
                if sym_name and sym_name not in existing_names:
                    insertion_block.append(op.clone())
                    existing_names.add(sym_name)
                    self._cloned_device_funcs.add(sym_name)

        assert callee, f"Could not find callee function {func_name} in the module."

        callee_function_type = callee.function_type.value.results
        call_results = func.call(
            result=callee_function_type,
            callee=callee.name.value,
            operands_=[self.load_var(arg) for arg in args]
            + [self.load_var(value) for (name, value) in kws],
        )

        target_type = self.get_numba_type(target.name)
        if isinstance(target_type, types.BaseTuple) and len(callee_function_type) > 1:
            return tuple(call_results)
        return call_results

    def build_recursive_call(
        self,
        target: numba_ir.Var,
        fn_type: types.RecursiveCall,
        args: list[numba_ir.Var],
        kws: list[tuple[str, numba_ir.Var]] | None = None,
    ):
        """Emit a func.call back to the function currently being compiled."""
        _ = target
        _ = fn_type
        if kws is None:
            kws = []
        assert self.mlir_funcOp is not None
        callee_function_type = self.mlir_funcOp.function_type.value.results
        callee_name = self.mlir_funcOp.name.value
        callOp = func.call(
            result=callee_function_type,
            callee=callee_name,
            operands_=[self.load_var(arg) for arg in args]
            + [self.load_var(value) for (name, value) in kws],
        )
        return callOp

    def _get_struct_field_index(self, value_type, attr) -> int | None:
        """Get the struct field index for a make_attribute_wrapper attribute.

        Looks up the data model for value_type and checks if the attribute
        (or without leading underscore) matches a field name.
        """
        try:
            model = self.context.data_model_manager.lookup(value_type)
        except KeyError:
            return None
        if not hasattr(model, "_fields") or not hasattr(model, "get_field_position"):
            return None

        for candidate in (attr, attr.lstrip("_")):
            if candidate in model._fields:
                return model.get_field_position(candidate)
        return None

    def _lower_record_array_field_view(self, target, value, attr):
        """
        Lower record array field access: arr.field_name -> strided array view.

        Creates a memref view into the record array by building an LLVM struct
        descriptor with the field's offset and record-sized strides.
        """
        from cusimt.lowering_utilities import convert

        value_type = self.get_numba_type(value.name)
        target_type = self.get_numba_type(target.name)
        record_type = value_type.dtype
        field_info = record_type.fields[attr]
        field_offset = field_info.offset
        record_size = record_type.size
        rank = value_type.ndim

        array_val = self.load_var(value)
        target_mlir_type = self.get_mlir_type(target_type)

        ptr_as_index = memref.extract_aligned_pointer_as_index(array_val)
        ptr_i64 = arith.index_cast(T.i64(), ptr_as_index)

        if field_offset > 0:
            ptr_i64 = arith.addi(ptr_i64, arith.constant(T.i64(), field_offset))

        ptr = llvm.inttoptr(llvm.PointerType.get(), ptr_i64)

        md = memref.extract_strided_metadata(array_val)

        field_size = field_info.type.bitwidth // 8
        stride_multiplier = record_size // field_size

        sizes_i64 = []
        strides_i64 = []
        for i in range(rank):
            sizes_i64.append(convert(md[2 + i], T.i64()))
            stride_i64 = convert(md[2 + rank + i], T.i64())
            strides_i64.append(
                arith.muli(stride_i64, arith.constant(T.i64(), stride_multiplier))
            )

        struct_type = ir.Type.parse(
            f"!llvm.struct<(ptr, ptr, i64, array<{rank} x i64>, array<{rank} x i64>)>"
        )
        i64c = lambda v: arith.constant(T.i64(), v)
        ins = lambda d, v, *p: llvm.insertvalue(
            container=d, value=v, position=ir.DenseI64ArrayAttr.get(list(p))
        )

        desc = llvm.UndefOp(struct_type).result
        desc = ins(desc, ptr, 0)
        desc = ins(desc, ptr, 1)
        desc = ins(desc, i64c(0), 2)
        for i, s in enumerate(sizes_i64):
            desc = ins(desc, s, 3, i)
        for i, s in enumerate(strides_i64):
            desc = ins(desc, s, 4, i)

        result = builtin.unrealized_conversion_cast([target_mlir_type], [desc])
        self.store_var(target, result)
        trace(
            "Record array field view: %s.%s stored to %s", value.name, attr, target.name
        )

    def _make_bound_receiver_var(self, fn, fn_value, recvr_type):
        """Create a synthetic Var for a bound method's receiver (self)."""
        name = f"$bound_self_{fn.name}_{id(fn)}"
        self.fndesc.typemap[name] = recvr_type
        self.varmap[name] = fn_value
        return numba_ir.Var(scope=fn.scope, name=name, loc=fn.loc)

    def lower_overload_call(
        self,
        target: numba_ir.Var,
        overload_disp: dispatcher.Dispatcher,
        args: list[numba_ir.Var],
        kws: list[tuple[str, numba_ir.Var]] = [],
    ):
        """
        Lower a call to an overloaded function by compiling its implementation
        to MLIR, linking the result, and emitting a func.call.
        """
        from cusimt import cuda
        from cusimt.lowering_utilities import link

        argtypes = tuple(
            [self.get_numba_type(arg.name) for arg in args]
            + [self.get_numba_type(value.name) for (name, value) in kws]
        )

        py_func = overload_disp.py_func
        cuda_func = cuda.jit(device=True)(py_func)
        unique_qualname = f"{py_func.__qualname__}_{id(overload_disp)}"
        func_name = generate_mangled_name(unique_qualname, argtypes)
        cres = cuda_func.compile(argtypes, abi_info={"abi_name": func_name})

        if callee_linker := cres.metadata.get("linker"):
            self.linker.merge_ltoirs_from(callee_linker)

        link.link_inplace(self.mlir_module, cres.metadata["mlir_module_str"])

        name_attr = ir.StringAttr.get(func_name)
        body = self.mlir_gpu_module.regions[0].blocks[0]
        callee = next(
            (
                op
                for op in body
                if isinstance(op, (func.FuncOp, gpu.GPUFuncOp)) and op.name == name_attr
            ),
            None,
        )
        if callee is None:
            raise InternalCompilerError(
                f"Could not find function {func_name} after linking overload MLIR."
            )

        callee_type = get_func_type(callee)
        call_args = [
            convert(val, ty)
            for val, ty in zip(self.load_vars(args), callee_type.inputs)
        ]
        call_result = func.call(
            result=callee_type.results,
            callee=callee.name.value,
            operands_=call_args,
        )

        target_type = self.get_numba_type(target.name)
        if isinstance(target_type, types.BaseTuple):
            result = tuple(call_result)
        else:
            result = call_result
        self.store_var(target, result)

    @typechecked
    def get_registered_builder(
        self,
        fn: numba_ir.Var | _Intrinsic | AnyCallable[PS],
        signature: typing.Signature | tuple[types.Type, ...],
    ) -> Builder | None:
        """
        After doing the work to find the code generator for the function
        and signature if it exists, filter out any builders that do not
        originate in cusimt.

        TODO: Instead, filter _out_ builders coming from Numba by overriding
        methods in the target
        """
        trace("fn: %s, signature: %s", fn, signature)
        if builder := self._get_registered_builder(fn, signature):
            if self._assume_builder_is_usable_for_mlir(builder):
                return builder
        return None

    @lru_cache(maxsize=None)
    def _filter_out_numba_builders(self, fn) -> None:
        overload_selector = self.context._defns[fn]
        overload_selector.versions = [
            version
            for version in overload_selector.versions
            if self._assume_builder_is_usable_for_mlir(version[-1])
        ]

    def _lookup_actual_function(self, fn, signature) -> Builder | None:
        """
        Ensure we have a real function and it exists in the context's registry
        of code generators, and return it if we can find one that matches
        the signature.
        """
        #        if callable(fn) and fn in self.context._defns:
        if fn in self.context._defns:

            self._filter_out_numba_builders(fn)
            try:
                impl = self.context.get_function(fn, signature)
                return impl._callable.func
            except NotImplementedError:
                return None
        return None

    def _extract_callable_from_var(self, var) -> Callable[..., Any] | None:
        """
        This is ugly. There are a few ways callables might be stored in
        numba variables. The best case is that we have stored a function
        or module attribute (also a function) directly to the variable,
        in which case we pull that out directly.

        Otherwise, the typing system may have resolved it to an intrinsic
        via a direct call or overload. In this case, we fall back on the
        _type_ of the variable to get the callable.
        """
        ty = self.get_numba_type(var.name)
        if isinstance(ty, types.NumberClass):
            return types.NumberClass
        if isinstance(ty, types.Function):
            if self.var_lowered(var):
                fn = self.load_var(var)
                if isinstance(fn, types.Function):
                    return fn.typing_key
            return ty.typing_key
        if self.var_lowered(var):
            fn = self.load_var(var)
            if isinstance(fn, DeferredLowering):
                return fn
            if isinstance(fn, types.Function):
                return fn.typing_key
            # BoundFunction vars hold the bound receiver value at runtime,
            # so callable resolution must come from the type/template key.
            if fn is not None and not isinstance(ty, types.BoundFunction):
                return fn
        if isinstance(ty, types.BoundFunction):
            return ty.typing_key
        return None

    def _get_registered_builder(self, fn, signature) -> Builder | None:
        """
        Look up a code generator for the given function and signature.

        Priority order:
        1. Registered lowerings in context._defns (from @lower decorators)
        2. Intrinsic _defn builders (if MLIR-compatible)
        3. DeferredLowering objects
        """
        if isinstance(fn, numba_ir.Var):
            fn = self._extract_callable_from_var(fn)
            if fn is None:
                return None

        if fn is range:
            trace("NOT deferring to special-handling of range built-in")
            # return None

        # Check registered lowerings first — these take priority over
        # intrinsic _defn builders so that cusimt @lower registrations
        # can shadow numba-cuda intrinsics.
        if builder := self._lookup_actual_function(fn, signature):
            return builder

        if isinstance(fn, _Intrinsic):
            has_arrays = any(
                isinstance(arg_ty, types.Array) for arg_ty in signature.args
            )
            if has_arrays:
                return None
            full_sig, maybe_builder = fn._defn(self, *signature.args)
            if maybe_builder and self._assume_builder_is_usable_for_mlir(maybe_builder):
                return maybe_builder

        if isinstance(fn, DeferredLowering):
            return fn

        return None

    def _link_external_function(self, fn_value: ExternFunction):
        if fn_value.use_cooperative:
            self.metadata["use_cooperative"] = True

        if not isinstance(fn_value.link, (tuple, list, set)):
            raise ValueError(
                f"Invalid link attribute for external function {fn_value.name}: expected tuple, list, or set, got {fn_value.link}"
            )

        for link_item in fn_value.link:
            self._link_external_item(link_item)

    def _link_external_item(self, link_item):
        key = self._external_link_item_key(link_item)
        if key in self._linked_external_items:
            return
        self._linked_external_items.add(key)

        if hasattr(link_item, "setup_callback") and link_item.setup_callback:
            self._setup_callbacks.append(link_item.setup_callback)
        if hasattr(link_item, "teardown_callback") and link_item.teardown_callback:
            self._teardown_callbacks.append(link_item.teardown_callback)
        self.linker.add_file_guess_ext(link_item)

    @staticmethod
    def _external_link_item_key(link_item):
        try:
            hash(link_item)
        except TypeError:
            return id(link_item)
        return link_item

    def _link_external_mlir_library(self, other: ExternMLIRLibrary):
        trace("linking %s", other)
        from cusimt.lowering_utilities import link

        if other in self._seen_mlir_libraries:
            trace("already seen %s, skipping", other)
            return
        self._seen_mlir_libraries.add(other)
        link.link_inplace(self.mlir_module, other.source)

    def lower_call_external_mlir_library_function(
        self, target, fn: ExternMLIRLibraryFunction, args, kws
    ):
        trace("lowering call to external MLIR library function %s", fn.name)
        self._link_external_mlir_library(fn.library)
        func_ty = to_mlir_type(fn.sig)
        callee = lookup_callee_in_module(fn.name, func_ty, self.mlir_gpu_module)
        if callee is None:
            logging.error(fn.library.source)
            raise RuntimeError(
                f"Could not find function {fn.name} in MLIR module."
                f" Expected to find function after linking in {fn}."
            )
        args = self.load_vars(args)
        callee_type = get_func_type(callee)
        args = [
            convert(arg, arg_type) for arg, arg_type in zip(args, callee_type.inputs)
        ]
        # Use callee's actual type to preserve layout information
        call_result = func.call(
            result=callee_type.results,
            callee=callee.name.value,
            operands_=args,
        )
        target_type = self.get_numba_type(target.name)
        if isinstance(target_type, types.BaseTuple):
            result = tuple(call_result)
        else:
            result = call_result
        self.store_var(target, result)

    def lower_call_external_function(self, target, fn_value: ExternFunction, args, kws):
        self._link_external_function(fn_value)

        if fn_value.abi == "c":
            self._lower_call_external_c_abi(target, fn_value, args)
        else:
            self._lower_call_external_numba_abi(target, fn_value, args)

    def _lower_call_external_numba_abi(self, target, fn_value: ExternFunction, args):
        external_abi_signature = user_signature_to_external_abi_signature(fn_value.sig)
        external_abi_mlir_type = to_mlir_type(external_abi_signature)
        assert isinstance(external_abi_mlir_type, ir.FunctionType)
        return_type = external_abi_mlir_type.results
        arg_types = tuple(external_abi_mlir_type.inputs)
        args = tuple(map(self.load_var, args))
        user_return_actual_type = fn_value.sig.return_type

        ptr = llvm.PointerType.get()
        if user_return_actual_type != types.void:
            return_mlir_type = to_mlir_type(user_return_actual_type)
        else:
            return_mlir_type = T.i8()
        c1 = arith.constant(result=T.i32(), value=1)
        return_ptr = llvm.alloca(res=ptr, array_size=c1, elem_type=return_mlir_type)
        args = [return_ptr] + list(args)

        operands = []
        for arg, ty in zip(args, arg_types):
            operands.append(convert(arg, ty))

        callee = get_or_insert_function(
            fn_value.name, external_abi_mlir_type, self.mlir_gpu_module
        )
        func.call(result=return_type, callee=callee.name.value, operands_=operands)
        if user_return_actual_type != types.void:
            return_type = to_mlir_type(user_return_actual_type)
            result = llvm.load(res=return_type, addr=return_ptr)
            self.store_var(target, result)

    def _lower_call_external_c_abi(self, target, fn_value: ExternFunction, args):
        user_sig = fn_value.sig
        c_mlir_type = to_mlir_type(user_sig)
        assert isinstance(c_mlir_type, ir.FunctionType)
        arg_types = tuple(c_mlir_type.inputs)
        result_types = list(c_mlir_type.results)
        args = tuple(map(self.load_var, args))

        operands = []
        for arg, ty in zip(args, arg_types):
            operands.append(convert(arg, ty))

        callee = get_or_insert_function(
            fn_value.name, c_mlir_type, self.mlir_gpu_module
        )
        if user_sig.return_type != types.void:
            result = func.call(
                result=result_types, callee=callee.name.value, operands_=operands
            )
            self.store_var(target, result)
        else:
            func.call(result=[], callee=callee.name.value, operands_=operands)

    def lower_call_expr_assign(self, target, fn, args, kws, expr=None):
        trace("target: %s, fn: %s, args: %s, kws: %s", target, fn, args, kws)
        for arg in args:
            assert self.var_lowered(arg), f"Arg {arg} not found in varmap."
        for name, value in kws:
            assert self.var_lowered(
                value
            ), f"Named arg {name}={value} not found in varmap."

        target_type = self.get_numba_type(target.name)
        kwarg_types = [self.get_numba_type(value.name) for (name, value) in kws]
        arg_types = [self.get_numba_type(arg.name) for arg in args]
        signature = target_type(*arg_types, *kwarg_types)

        if expr is not None and expr in self.fndesc.calltypes:
            ct_sig = self.fndesc.calltypes[expr]
            if ct_sig.recvr:
                signature = ct_sig

        fn_value = self.load_var(fn)
        call_args = list(args)

        if signature.recvr:
            call_args = [
                self._make_bound_receiver_var(fn, fn_value, signature.recvr),
                *args,
            ]

        # Handle literal_unroll specially - it just passes through its argument
        from cusimt.numba_cuda.misc.special import literal_unroll

        if fn_value is literal_unroll:
            if len(args) == 1:
                arg_value = self.load_var(args[0])
                self.store_var(target, arg_value)
                return
            else:
                raise InternalCompilerError("literal_unroll expects exactly 1 argument")

        if builder := self.get_registered_builder(fn, signature):
            builder_args = args if isinstance(builder, DeferredLowering) else call_args
            builder(self, target, builder_args, kws)
            return

        fn_type = self.get_numba_type(fn.name)

        if isinstance(fn_type, types.RecursiveCall):
            call = self.build_recursive_call(target, fn_type, args, kws)
            self.store_var(target, call)
            return

        if isinstance(fn_value, dispatcher.Dispatcher):
            call = self.build_user_defined_function_call(target, fn_value, args, kws)
            self.store_var(target, call)
        elif isinstance(fn_value, ExternFunction):
            self.lower_call_external_function(target, fn_value, args, kws)
        elif isinstance(fn_value, ExternMLIRLibraryFunction):
            self.lower_call_external_mlir_library_function(target, fn_value, args, kws)
        elif fn.name in self.user_defined_functions:
            callee = self.user_defined_functions[fn.name]
            callOp = self.build_user_defined_function_call(target, callee, args, kws)
            self.store_var(target, callOp)
        elif overload_builder := self.context.get_overload_builder(fn_type, signature):
            overload_builder(self, target, call_args, kws)
        elif getattr(fn_value, "__cusimt_jitable__", False):
            # Function marked by cusimt's @register_jitable.
            # Cache the dispatcher on the function to avoid recompilation.
            if not hasattr(fn_value, "_cusimt_device_dispatcher"):
                from cusimt import cuda

                fn_value._cusimt_device_dispatcher = cuda.jit(device=True)(fn_value)
            call = self.build_user_defined_function_call(
                target, fn_value._cusimt_device_dispatcher, args, kws
            )
            self.store_var(target, call)
        else:
            fn_module = getattr(fn_value, "__module__", "<unknown module>")
            fn_qualname = getattr(fn_value, "__qualname__", repr(fn_value))
            raise NotImplementedError(
                f"NotImplemented lowering call to {fn_value}\n"
                f"  function: {fn_qualname}\n"
                f"  module:   {fn_module}\n"
                f"  type(fn_value): {type(fn_value)}\n"
                f"  fn_type:  {fn_type}\n"
                f"  signature: {signature}"
            )

    def lower_arrayexpr_assign(self, target, func, args):
        """
        Lower NumPy universal function (ufunc) calls on arrays.

        When NumPy ufuncs like np.abs(array) or np.sqrt(array) are called,
        Numba routes them through this arrayexpr path instead of direct
        function calls. We look up the ufunc in our database and delegate
        to the appropriate MLIR lowering function.
        """
        trace("target: %s, func: %s, args: %s", target, func, args)

        # Import the ufunc database
        from cusimt.ufunc_db import (
            get_ufunc_lowering,
            is_supported_ufunc,
            get_supported_ufuncs,
        )

        # Check if this is a supported ufunc
        if not is_supported_ufunc(func):
            supported = get_supported_ufuncs()
            raise NotImplementedError(
                f"Ufunc {func.__name__ if hasattr(func, '__name__') else func} is not supported. "
                f"Supported ufuncs: {[f.__name__ for f in supported if hasattr(f, '__name__')]}"
            )

        # Get the lowering function and call it
        lowering_fn = get_ufunc_lowering(func)
        if lowering_fn is None:
            name = func.__name__ if hasattr(func, "__name__") else func
            raise NotImplementedError(f"Ufunc {name} is not supported.")
        lowering_fn(self, target, args, {})

    def _get_shared_address_space(self):
        return ir.Attribute.parse("#gpu.address_space<workgroup>")

    def _get_shared_memory_base(self):
        if self._shared_memory_base is None:
            mr_type = memref.MemRefType.get(
                shape=[ir.ShapedType.get_dynamic_size()],
                element_type=T.i8(),
                memory_space=self._get_shared_address_space(),
            )
            assert self.mlir_funcOp
            with ir.InsertionPoint.at_block_begin(self.mlir_funcOp.entry_block):
                self._shared_memory_base = gpu.dynamic_shared_memory(mr_type)
        return self._shared_memory_base

    def _request_shared_memory(
        self, sizes: tuple[ir.Value, ...], mr_type: ir.MemRefType
    ):
        match mr_type.element_type:
            case ir.IntegerType() | ir.FloatType() as t:
                bytes: int = t.width // 8
            case T.index:
                bytes: int = 8
            case _:
                raise NotImplementedError(
                    f"NotImplemented shared memory type {mr_type}."
                )
        assert self.mlir_funcOp
        with ir.InsertionPoint(self.mlir_funcOp.entry_block):
            bytes_op = arith.constant(result=T.index(), value=bytes)
            for size in sizes:
                size = self.mlir_convert(size, T.index())
                bytes_op = arith.muli(lhs=bytes_op, rhs=size)
            shm_base = self._get_shared_memory_base()
            if self._total_shared_memory_bytes is None:
                self._total_shared_memory_bytes = arith.constant(
                    result=T.index(), value=0
                )
            view = memref.view(
                result=mr_type,
                source=shm_base,
                byte_shift=self._total_shared_memory_bytes,
                sizes=sizes,
            )
            self._total_shared_memory_bytes = arith.addi(
                lhs=self._total_shared_memory_bytes, rhs=bytes_op
            )
        return view

    def _get_tuple_element_type(self, target_type):
        """Determine common element type for tuple, using int64 for integer index tuples."""
        if isinstance(target_type, types.Tuple):
            all_integer = all(
                isinstance(target_type.types[i], (types.Integer, types.IntegerLiteral))
                for i in range(target_type.count)
            )
            if all_integer:
                return types.int64, self.get_mlir_type(types.int64)

            mlir_element_type = self.get_mlir_type(target_type.types[0])
            all_same = all(
                self.get_mlir_type(target_type.types[i]) == mlir_element_type
                for i in range(1, target_type.count)
            )
            if not all_same:
                return types.int64, self.get_mlir_type(types.int64)
            return target_type.types[0], mlir_element_type
        else:
            assert isinstance(
                target_type, types.UniTuple
            ), f"Does not support building Memref type from Numba's tuple type {target_type}"
            if isinstance(target_type.dtype, (types.Integer, types.IntegerLiteral)):
                return types.int64, self.get_mlir_type(types.int64)
            return target_type.dtype, self.get_mlir_type(target_type.dtype)

    def _coerce_tuple_item(self, item, common_numba_type, mlir_element_type):
        """Convert tuple item to common type, handling index variables specially."""
        item_value = self.load_var(item)
        return self.mlir_convert(item_value, self.get_mlir_type(common_numba_type))

    def lower_build_tuple_expr_assign(self, target, items):
        trace("target: %s, items: %s", target, items)
        for item in items:
            assert self.var_lowered(item), f"Item {item} not found in varmap."

        target_type = self.get_numba_type(target.name)
        if target_type.count != len(items):
            raise InternalCompilerError(
                f"Target type {target_type} has {target_type.count} elements, but {len(items)} items were provided."
            )

        value = tuple(self.load_var(item) for item in items)
        self.store_var(target, value)

    def _assume_builder_is_usable_for_mlir(self, object):
        module = getattr(object, "__module__", None) or ""
        is_numba = module.startswith("cusimt.numba_cuda.")
        is_cccl = module.startswith("cuda.coop")
        trace(
            "\n\tis object %s defined in numba?: %s, in cccl?: %s",
            object,
            is_numba,
            is_cccl,
        )
        return not (is_numba or is_cccl)

    def _is_extension_cast(self, cast_impl):
        """True only for third-party extension casts (e.g. cuDF).

        Casts from numba.* use llvmlite APIs and are incompatible.
        Casts from cusimt.* use a different calling convention
        (builder, target, args, kwargs) and are handled by their own
        lowering paths.  Only extension casts (neither numba nor cusimt)
        follow the (context, builder, fromty, toty, val) convention we
        call here.
        """
        module = getattr(cast_impl, "__module__", "")
        return not (module.startswith("numba.") or module.startswith("cusimt."))

    def lower_cast(self, source_type, target_type, value):
        """Dispatch a type cast through the extension registry or fall back to MLIR conversion.

        Checks the cast registry for a third-party extension cast (registered
        via ``cusimt.extending.lower_cast``).  If one is found, it is invoked
        with the standard (context, builder, fromty, toty, val) convention.
        Otherwise, falls back to ``mlir_convert`` for built-in MLIR type
        conversions (e.g. int widening, float promotion).
        """
        try:
            cast_impl = self.context._casts.find((source_type, target_type))
        except errors.NumbaNotImplementedError:
            cast_impl = None
        if cast_impl is not None and self._is_extension_cast(cast_impl):
            return cast_impl(self.context, self, source_type, target_type, value)
        return self.mlir_convert(value, self.get_mlir_type(target_type))

    def get_getattr_builder(self, target, value, attr):
        target_type = self.get_numba_type(target.name)
        value_type = self.get_numba_type(value.name)
        trace("target_type=%s, value_type=%s, attr=%s", target_type, value_type, attr)
        try:
            if builder := self.context.get_getattr(value_type, attr):
                if self._assume_builder_is_usable_for_mlir(builder):
                    return builder
        except NotImplementedError:
            # No registered builder - will be handled by explicit lowering in lower_getattr_assign
            pass
        return None

    def lower_getattr_assign(self, target, value, attr):
        target_type = self.get_numba_type(target.name)
        value_type = self.get_numba_type(value.name)
        trace("target_type=%s, value_type=%s, attr=%s", target_type, value_type, attr)

        if builder := self.get_getattr_builder(target, value, attr):
            builder(self.context, self, target, value, attr)
            return

        if isinstance(target_type, types.BoundFunction):
            self.store_var(target, self.load_var(value))
            return

        if isinstance(value_type, types.Module):
            pymod = value_type.pymod
            real_attr = getattr(pymod, attr)
            self.store_var(target, real_attr)
            trace(
                "getattr stored %s = %s (%s, %s)",
                target.name,
                real_attr,
                pymod.__name__,
                attr,
            )
            return

        # Handle array.ctypes - returns an ArrayCTypes wrapper
        if isinstance(value_type, types.Array) and attr == "ctypes":
            # In MLIR, we just pass through the memref - the ArrayCTypes is a
            # wrapper type that we handle specially when accessing .data
            array_value = self.load_var(value)
            self.store_var(target, array_value)
            return

        # Handle ArrayCTypes.data - extract data pointer as integer
        if isinstance(value_type, types.ArrayCTypes) and attr == "data":
            # Get the underlying array/memref
            array_value = self.load_var(value)
            # Extract the aligned pointer as an index
            ptr_as_index = memref.extract_aligned_pointer_as_index(array_value)
            # Convert index to uintp
            result = arith.index_cast(T.i64(), ptr_as_index)
            self.store_var(target, result)
            return

        # Handle EnumClass.member - return the constant value of the enum member
        if isinstance(value_type, types.EnumClass):
            member = getattr(value_type.instance_class, attr)
            dtype = value_type.dtype
            mlir_type = to_mlir_type(dtype)
            const = arith.constant(mlir_type, member.value)
            self.store_var(target, const)
            return

        if (field_idx := self._get_struct_field_index(value_type, attr)) is not None:
            struct_val = self.load_var(value)
            result = llvm.extractvalue(
                res=self.get_mlir_type(target_type),
                container=struct_val,
                position=ir.DenseI64ArrayAttr.get([field_idx]),
            )
            self.store_var(target, result)
            return

        # Handle record array field access: arr.field -> strided array view
        if isinstance(value_type, types.Array) and isinstance(
            value_type.dtype, types.Record
        ):
            record_type = value_type.dtype
            if attr in record_type.fields:
                self._lower_record_array_field_view(target, value, attr)
                return

        raise NotImplementedError(f"getattr({value_type}, {attr=})")

    def lower_setattr(self, setattr_inst):
        """
        target.attr = value

        This handles struct field modification using LLVM insertvalue
        and union variant modification using bitcast.
        """
        trace("setattr_inst: %s", setattr_inst)
        target = setattr_inst.target
        attr = setattr_inst.attr
        value = setattr_inst.value

        target_type = self.get_numba_type(target.name)
        value_type = self.get_numba_type(value.name)
        trace(
            "lower_setattr: %s.%s = %s, target_type=%s",
            target.name,
            attr,
            value.name,
            target_type,
        )

        # Check registry for setattr implementation (e.g., Record types)
        try:
            # get_setattr expects (attr, sig) where sig has (target_type, value_type)
            setattr_sig = types.void(target_type, value_type)
            if builder := self.context.get_setattr(attr, setattr_sig):
                # Check if this builder is from our registry (not upstream numba)
                module = getattr(builder, "__module__", "")
                if not module.startswith("cusimt.numba_cuda."):
                    builder(self, [target, value])
                    return
        except NotImplementedError:
            pass

        # Handle AggregateType (struct) setattr
        if isinstance(target_type, AggregateType):
            # Check if this is a bitfield
            if attr in target_type.field_layout and target_type.field_layout[attr].get(
                "is_bitfield", False
            ):
                # This is a bitfield - use shift/mask operations
                field_info = target_type.field_layout[attr]
                bit_offset = field_info["bit_offset"]
                bit_width = field_info["bit_width"]

                trace(
                    f"Bitfield setattr: {attr} at bit_offset={bit_offset}, bit_width={bit_width}"
                )

                # Load the current struct value
                struct_value = self.load_var(target)

                # Get the storage type for this bitfield struct
                storage_type = target_type.get_bitfield_storage_type()

                # Extract the storage field at position 0
                storage_mlir_type = self.get_mlir_type(storage_type)
                storage_value = llvm.extractvalue(
                    res=storage_mlir_type,
                    container=struct_value,
                    position=ir.DenseI64ArrayAttr.get([0]),
                )

                # Load the new value
                new_value = self.load_var(value)

                # Convert to storage type if needed
                new_value = convert(new_value, storage_mlir_type)

                # Create clear mask using MLIR operations to avoid Python integer overflow
                # clear_mask = ~(((1 << bit_width) - 1) << bit_offset)

                # Create field mask: (1 << bit_width) - 1
                one = arith.constant(result=storage_mlir_type, value=1)
                bit_width_const = arith.constant(
                    result=storage_mlir_type, value=bit_width
                )
                field_mask = arith.shli(one, bit_width_const)
                field_mask = arith.subi(field_mask, one)

                # Compute bit_offset_const once if needed (reused for both mask and value shifting)
                bit_offset_const = (
                    arith.constant(result=storage_mlir_type, value=bit_offset)
                    if bit_offset > 0
                    else None
                )

                # Shift the mask to the correct position
                if bit_offset_const:
                    shifted_mask = arith.shli(field_mask, bit_offset_const)
                else:
                    shifted_mask = field_mask

                # Invert to get clear mask (NOT operation)
                all_ones = arith.constant(result=storage_mlir_type, value=-1)
                clear_mask_mlir = arith.xori(shifted_mask, all_ones)

                # Clear the bits
                cleared_storage = arith.andi(storage_value, clear_mask_mlir)

                # Shift new value to correct position
                if bit_offset_const:
                    shifted_value = arith.shli(new_value, bit_offset_const)
                else:
                    shifted_value = new_value

                # OR with cleared storage
                updated_storage = arith.ori(cleared_storage, shifted_value)

                # Insert back into struct
                updated_struct = llvm.insertvalue(
                    container=struct_value,
                    value=updated_storage,
                    position=ir.DenseI64ArrayAttr.get([0]),
                )

                self.store_var(target, updated_struct)
                trace("Stored bitfield '%s' back to %s", attr, target.name)
                return

            # Regular field (not a bitfield) - use field_layout
            if attr not in target_type.field_layout:
                available_fields = list(target_type.field_layout.keys())
                raise AttributeError(
                    f"Struct type '{target_type.name}' has no field '{attr}'. "
                    f"Available fields: {available_fields}"
                )

            # Get field info from field_layout (single source of truth)
            field_info = target_type.field_layout[attr]
            field_index = field_info["field_index"]
            field_type = field_info["underlying_type"]

            # Load the current struct value
            struct_value = self.load_var(target)
            # Load the new field value
            new_field_value = self.load_var(value)

            # Get the expected field type and convert the value to match
            field_mlir_type = self.get_mlir_type(field_type)

            # Convert the value to the correct type if needed
            new_field_value = self.mlir_convert(new_field_value, field_mlir_type)

            # Use LLVM insertvalue to create a new struct with the modified field
            updated_struct = llvm.insertvalue(
                container=struct_value,
                value=new_field_value,
                position=ir.DenseI64ArrayAttr.get([field_index]),
            )

            # Store the updated struct back to the target variable
            self.store_var(target, updated_struct)
            trace(
                "Stored updated struct with field '%s' (index %s) back to %s",
                attr,
                field_index,
                target.name,
            )
            return

        # Handle UnionType (union) setattr
        if isinstance(target_type, UnionType):
            # Find the variant type
            variant_type = target_type.get_variant_type(attr)
            if variant_type is None:
                raise AttributeError(
                    f"Union type '{target_type.name}' has no variant '{attr}'. "
                    f"Available variants: {[v[0] for v in target_type.variants]}"
                )

            # Load the new value
            new_value = self.load_var(value)

            # Get the union storage type
            union_mlir_type = self.get_mlir_type(target_type)

            # Check if variant is a struct
            if isinstance(variant_type, AggregateType):
                # For structs, extract each field and pack into integer
                union_value = arith.constant(result=union_mlir_type, value=0)

                # Check if this is a bitfield struct
                if variant_type.is_bitfield_struct:
                    # Bitfield struct: struct has a single storage field at position 0
                    # Just extract the storage field and use it as the union value

                    # Get the storage type for this bitfield struct
                    storage_type = variant_type.get_bitfield_storage_type()

                    storage_mlir_type = self.get_mlir_type(storage_type)

                    # Extract storage field from struct (position 0 - the only field)
                    storage_value = llvm.extractvalue(
                        res=storage_mlir_type,
                        container=new_value,
                        position=ir.DenseI64ArrayAttr.get([0]),
                    )

                    # Extend/truncate to union storage size if needed
                    union_value = convert(storage_value, union_mlir_type)
                else:
                    # Regular (non-bitfield) struct: use field_layout for everything
                    for field_name, field_info in variant_type.field_layout.items():
                        # Get field info from field_layout (single source of truth)
                        field_index = field_info["field_index"]
                        field_type = field_info["underlying_type"]
                        bit_offset = field_info["bit_offset"]

                        # Extract field from struct
                        field_value = llvm.extractvalue(
                            res=self.get_mlir_type(field_type),
                            container=new_value,
                            position=ir.DenseI64ArrayAttr.get([field_index]),
                        )

                        # Extend to union storage size if needed
                        field_mlir_type = self.get_mlir_type(field_type)
                        extended = convert(field_value, union_mlir_type)

                        # Shift to correct position
                        if bit_offset > 0:
                            shift_amount = arith.constant(
                                result=union_mlir_type, value=bit_offset
                            )
                            shifted = arith.shli(extended, shift_amount)
                        else:
                            shifted = extended

                        # OR into union value
                        union_value = arith.ori(union_value, shifted)
            else:
                # For primitive types, bitcast works
                union_value = convert(new_value, union_mlir_type)

            # Store the casted value back to the union
            self.store_var(target, union_value)
            trace(
                "Packed value into union storage and stored in %s.%s", target.name, attr
            )
            return

        raise NotImplementedError(
            f"NotImplemented lowering setattr for {target.name}.{attr} = {value.name} (type: {target_type})"
        )

    def lower_setitem(self, setitem_inst):
        """
        target[index] = value
        """
        trace("setitem_inst: %s", setitem_inst)
        target = setitem_inst.target
        index = getattr(setitem_inst, "index_var", setitem_inst.index)
        value = setitem_inst.value
        arg_types = [self.get_numba_type(var.name) for var in setitem_inst.list_vars()]
        signature = types.Any(*arg_types)
        if builder := self.get_registered_builder(operator.setitem, signature):
            builder(self, None, [target, index, value], [])
            return

    def lower_branch(self, branch_inst):
        """
        branch cond, truebr, falsebr
        """
        trace("branch_inst: %s", branch_inst)
        cond = branch_inst.cond
        truebr = branch_inst.truebr
        falsebr = branch_inst.falsebr

        assert self.var_lowered(cond), f"Condition {cond} not found in varmap."
        assert truebr in self.blkmap, f"truebr {truebr} not found in blkmap"
        assert falsebr in self.blkmap, f"falsebr {falsebr} not found in blkmap"

        cf.cond_br(
            condition=self.load_var(cond),
            true_dest_operands=[],
            false_dest_operands=[],
            true_dest=self.blkmap[truebr],
            false_dest=self.blkmap[falsebr],
        )

    def lower_jump(self, jump_inst):
        """
        jump target
        """
        trace("jump_inst=%s", jump_inst)
        target = jump_inst.target

        assert target in self.blkmap, f"target {target} not found in self.blkmap"

        cf.br(dest_operands=[], dest=self.blkmap[target])

    def lower_del(self, del_inst):
        """
        del value
        """
        # # value is of class 'str' and doesn't have 'name' field
        # value = del_inst.value
        # del self.varmap[value]
        pass

    def concretize_tuple(self, value: tuple[ir.Value, ...] | ir.Value) -> ir.Value:
        """
        We try to delay concretization of Python types as long as possible; this
        concretizes tuples whos element types are already values.
        """
        if isinstance(value, ir.Value):
            return value
        if not all(isinstance(x, ir.Value) for x in value):
            raise InternalCompilerError(
                f"Tuple {value} contains non-MLIR values: {value=}."
            )
        with self.alloca_insertion_point():
            dtype = value[0].type
            mr_type = T.memref(len(value), element_type=dtype)
            mr = memref.alloca(memref=mr_type, dynamic_sizes=[], symbol_operands=[])
        for i, v in enumerate(value):
            memref.store(value=ir.Value(v), memref=ir.Value(mr), indices=[index_of(i)])
        return mr

    def lower_return(self, return_inst):
        """
        return value
        """
        trace("return_inst=%s", return_inst)
        value = return_inst.value
        value_type = self.get_numba_type(value.name)
        return_ctor = (
            gpu.ReturnOp
            if isinstance(self.mlir_funcOp, gpu.GPUFuncOp)
            else func.ReturnOp
        )
        if isinstance(value_type, types.NoneType):
            return_ctor([])
        else:
            value = self.load_var(value)
            if isinstance(value, tuple):
                return_ctor(list(value))
            else:
                return_ctor([value])

    def _unwrap_mlir_value(self, value):
        if isinstance(value, ir.OpView):
            return value.result
        if isinstance(value, ir.Value):
            return value
        return None

    def _need_deferred_debug_emission(self, value_type):
        return isinstance(
            value_type,
            (types.Array, types.Complex, ir.ComplexType, ir.MemRefType),
        )

    def _llvm_struct_type(self, elem_types):
        return ir.Type.parse(f"!llvm.struct<({', '.join(str(t) for t in elem_types)})>")

    def _build_llvm_aggregate_value(self, aggregate_type, elems, elem_types):
        if len(elems) != len(elem_types):
            return None

        aggregate = llvm.UndefOp(aggregate_type).result
        for i, (elem, elem_type) in enumerate(zip(elems, elem_types, strict=True)):
            elem = self.mlir_convert(elem, elem_type)
            aggregate = llvm.insertvalue(
                container=aggregate,
                value=elem,
                position=ir.DenseI64ArrayAttr.get([i]),
            )
        return aggregate

    def _materialize_tuple_value(self, value, numba_type):
        if len(value) != numba_type.count:
            return None

        elems = []
        for elem in value:
            elem = self._unwrap_mlir_value(elem)
            if elem is None:
                return None
            elems.append(elem)

        if isinstance(numba_type, types.UniTuple):
            elem_types = [self.get_mlir_type(numba_type.dtype)] * numba_type.count
            if any(self._need_deferred_debug_emission(t) for t in elem_types):
                return None
            aggregate_type = ir.Type.parse(
                f"!llvm.array<{numba_type.count} x {elem_types[0]}>"
            )
        else:
            elem_types = [self.get_mlir_type(t) for t in numba_type.types]
            if any(self._need_deferred_debug_emission(t) for t in elem_types):
                return None
            aggregate_type = self._llvm_struct_type(elem_types)

        return self._build_llvm_aggregate_value(aggregate_type, elems, elem_types)

    def _emit_dbg_value(self, var_name, value):
        """Emit llvm.intr.dbg.value for a local scalar variable if full debug is on.

        Scalar function arguments (except booleans) and supported aggregate locals
        are described via dbg.declare on stack storage for a stable debug location,
        while scalar locals use dbg.value.
        """
        if (
            not self._debug_full
            or self._di_builder is None
            or not self._di_builder.valid
            or var_name.startswith("$")
        ):
            return
        # Numba SSA renames: foo.1, foo.2, etc. Map back to the user name.
        base_name = self._canonical_dbg_var_name(var_name)
        var_attr = self._di_builder.di_local_vars.get(base_name)
        if var_attr is None:
            return
        numba_type = self._get_numba_type_for_dbg_var(var_name)
        if self._need_deferred_debug_emission(numba_type):
            return
        if isinstance(numba_type, types.BaseTuple) and isinstance(value, tuple):
            aggregate = self._materialize_tuple_value(value, numba_type)
            if aggregate is not None:
                self._emit_dbg_declare(base_name, aggregate, var_attr)
            return
        if isinstance(numba_type, types.Record):
            mlir_value = self._unwrap_mlir_value(value)
            if mlir_value is not None:
                llvm.intr_dbg_declare(
                    mlir_value,
                    var_attr,
                    location_expr=self._di_builder.di_expression,
                )
            return
        mlir_value = self._unwrap_mlir_value(value)
        if mlir_value is None:
            return
        is_arg = base_name in self._di_builder.arg_names
        is_boolean = isinstance(numba_type, types.Boolean)
        if is_arg and not is_boolean:
            # Use dbg.declare (alloca+store) for scalar args, except boolean
            # args which use dbg.value to avoid a known NVVM crash.
            self._emit_dbg_declare(base_name, mlir_value, var_attr)
        else:
            llvm.intr_dbg_value(
                mlir_value,
                var_attr,
                location_expr=self._di_builder.di_expression,
            )

    def _emit_dbg_declare(self, var_name, value, var_attr):
        """Emit llvm.intr.dbg.declare for a value materialized in stack storage."""
        alloca_ptr = self.alloca(value.type)
        llvm.store(value, alloca_ptr)
        llvm.intr_dbg_declare(
            alloca_ptr,
            var_attr,
            location_expr=self._di_builder.di_expression,
        )

    def get_loc(self, loc):
        lineinfo = loc
        if not lineinfo.col:
            lineinfo.col = 0
        return lineinfo

    def load_vars(self, vars: Sequence[numba_ir.Var]) -> list[ir.Value]:
        """
        Load the values from the given numba variables.
        """
        return [self.load_var(v) for v in vars]

    def load_var(self, var: numba_ir.Var):
        result = self._load_var(var)
        numba_type = self.get_numba_type(var.name)
        result = self.lower_literal_if_needed(result, numba_type)
        return result

    def _load_var(self, var: numba_ir.Var) -> Any:
        """
        Load the value from the given numba variable.
        """
        trace("var=%s", var)
        if isinstance(var, (list, tuple)):
            return [self.load_var(v) for v in var]

        assert self.var_lowered(var), f"Var {var.name} not found in varmap."

        if var.name in self.var_assign_count and self.var_assign_count[var.name] > 1:
            # if variable is stack allocated (multiple assigned),
            # load the variable from stack
            var_type = self.get_numba_type(var.name)
            slot = self.varmap[var.name]

            if isinstance(slot.type, MemRefType):
                trace("")
                index = index_of(0)
                trace("index=%s", index)
                loadOp = memref.load(memref=slot, indices=[index])
                trace("loadOp=%s", loadOp)
                return loadOp

            trace("Loading %s from LLVM stack slot", type(var_type).__name__)
            return llvm.load(res=self.get_mlir_type(var_type), addr=slot)
        elif var.name in self._debug_forced_alloca:
            # variable forced to memref.alloca for debug info.
            return memref.load(memref=self.varmap[var.name], indices=[index_of(0)])
        else:
            trace("")
            # the variable is promoted to register,
            # load the value from varmap
            return self.varmap[var.name]

    def store_var(self, var, value):
        """
        Store the value (MLIR Op) into the given variable.
        """
        trace("var=%s value=%s", var, value)
        if var.name in self.var_assign_count and self.var_assign_count[var.name] > 1:
            # if variable is stack allocated (multiple assigned),
            # store the value to stack
            assert self.var_lowered(
                var
            ), f"Stack allocated var {var.name} not found in varmap."

            slot = self.varmap[var.name]
            var_type = self.get_numba_type(var.name)

            if isinstance(slot.type, MemRefType):
                memref.store(value=value, memref=slot, indices=[index_of(0)])
            else:
                trace("Storing %s to LLVM stack slot", type(var_type).__name__)
                llvm.store(value=value, addr=slot)
        else:
            # the value can be safely stored in register,
            # register the value in varmap
            assert not self.var_lowered(
                var
            ), f"Var {var.name} already defined in varmap."
            numba_type = self._get_numba_type_for_dbg_var(var.name)
            if (
                self._debug_full
                and isinstance(numba_type, types.Complex)
                and isinstance(value, (ir.Value, ir.OpView))
            ):
                # Force single-assign complex vars onto stack so deferred dbg.declare
                # has a stable pointer location after memref->LLVM lowering.
                mlir_value = value.result if isinstance(value, ir.OpView) else value
                memref_type = ir.MemRefType.get(shape=[1], element_type=mlir_value.type)
                alloca_op = memref.alloca(
                    memref=memref_type, dynamic_sizes=[], symbol_operands=[]
                )
                memref.store(value=mlir_value, memref=alloca_op, indices=[index_of(0)])
                self.varmap[var.name] = alloca_op
                self._debug_forced_alloca.add(var.name)
                self._tag_alloca_for_deferred_dbg_declare(var.name, alloca_op)
            else:
                self.varmap[var.name] = value

        self._emit_dbg_value(var.name, value)

    def get_numba_type(self, var):
        """
        Get the numba type of a Numba variable.
        """
        match var:
            case numba_ir.Var():
                return self.fndesc.typemap[var.name]
            case str():
                return self.fndesc.typemap[var]
            case _:
                raise TypeError(f"Cannot get numba type from {type(var)=}")

    def get_mlir_type(self, ty):
        """
        Get the MLIR type from a Numba type.
        """
        match ty:
            case typing.Signature() as sig:
                if sig.return_type in (types.none, types.void):
                    ret_ty = tuple()
                else:
                    ret_ty = [self.get_mlir_type(sig.return_type)]
                return ir.FunctionType.get(
                    list(map(self.get_mlir_type, sig.args)),
                    ret_ty,
                )
            case numba_ir.Var():
                return self.get_mlir_type(self.get_numba_type(ty))
            case MLIRDispatcherType():
                return ir.NoneType.get()
            case types.StringLiteral():
                return self.context.get_value_type(types.unicode_type)
            case types.CharSeq():
                return ir.Type.parse(f"!llvm.array<{ty.count} x i8>")
            case types.UnicodeCharSeq():
                import numpy as np

                char_bits = np.dtype("U1").itemsize * 8
                return ir.Type.parse(f"!llvm.array<{ty.count} x i{char_bits}>")
            case _:
                try:
                    return self.context.get_value_type(ty)
                except KeyError as e:
                    raise TypeError(
                        f"Cannot convert type {str(ty)} to MLIR type."
                    ) from e

    def var_lowered(self, var):
        return var.name in self.varmap

    def _count_flat_elements(self, numba_type) -> int:
        """Count how many flat MLIR arguments a type expands to."""
        if isinstance(numba_type, types.NoneType):
            return 0
        if isinstance(numba_type, types.BaseTuple):
            if isinstance(numba_type, types.UniTuple):
                return numba_type.count * self._count_flat_elements(numba_type.dtype)
            else:
                return sum(self._count_flat_elements(t) for t in numba_type.types)
        else:
            return 1

    def _flatten_type(self, numba_type) -> list:
        """Recursively flatten a Numba type to a list of scalar/array types."""
        if isinstance(numba_type, types.NoneType):
            return []
        if isinstance(numba_type, types.BaseTuple):
            if isinstance(numba_type, types.UniTuple):
                elem_flat = self._flatten_type(numba_type.dtype)
                return elem_flat * numba_type.count
            else:
                result = []
                for t in numba_type.types:
                    result.extend(self._flatten_type(t))
                return result
        else:
            return [numba_type]

    def _get_flat_arg_start_index(self, arg_index: int) -> int:
        """Get the starting index in flattened block arguments for a given original arg index."""
        flat_idx = 0
        for i in range(arg_index):
            if not _is_omitted_arg(self.fndesc.argtypes[i]):
                flat_idx += self._count_flat_elements(self.fndesc.argtypes[i])
        return flat_idx

    def _reassemble_tuple_from_block_args(self, numba_type, start_idx: int) -> tuple:
        """Reassemble a tuple from flattened block arguments. Returns Python tuple of ir.Values."""
        if isinstance(numba_type, types.UniTuple):
            elements = []
            idx = start_idx
            for _ in range(numba_type.count):
                if isinstance(numba_type.dtype, types.BaseTuple):
                    elem = self._reassemble_tuple_from_block_args(numba_type.dtype, idx)
                    idx += self._count_flat_elements(numba_type.dtype)
                else:
                    elem = self.mlir_funcOp.entry_block.arguments[idx]
                    idx += 1
                elements.append(elem)
            return tuple(elements)
        elif isinstance(numba_type, types.BaseTuple):
            elements = []
            idx = start_idx
            for elem_type in numba_type.types:
                if isinstance(elem_type, types.BaseTuple):
                    elem = self._reassemble_tuple_from_block_args(elem_type, idx)
                    idx += self._count_flat_elements(elem_type)
                else:
                    elem = self.mlir_funcOp.entry_block.arguments[idx]
                    idx += 1
                elements.append(elem)
            return tuple(elements)
        else:
            return self.mlir_funcOp.entry_block.arguments[start_idx]

    def verify_mlir_module(self):
        trace("Verifying MLIR module:\n%s", str(self.mlir_module))
        try:
            self.mlir_module.operation.verify()
        except Exception as e:
            print(self.mlir_module.operation.get_asm())
            raise InternalCompilerError(f"Invalid MLIR module: {e}") from e
