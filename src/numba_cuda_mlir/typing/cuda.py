# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import lru_cache
import inspect
from numba_cuda_mlir.errors import ForceLiteralArg
from numba_cuda_mlir.numba_cuda.typing import typeof
from numba_cuda_mlir.numba_cuda.misc.special import literal_unroll
from numba_cuda_mlir.numba_cuda.misc.special import literally
from numba_cuda_mlir.numba_cuda.misc.special import (
    literal_unroll as cuda_literal_unroll,
)
from numba_cuda_mlir.numba_cuda.typing.templates import (
    CallableTemplate,
    AttributeTemplate,
    ConcreteTemplate,
    AbstractTemplate,
    Registry,
    signature,
)
from numba_cuda_mlir import types
from numba_cuda_mlir.cuda.vector_types import (
    vector_types_by_name,
    vector_types_by_alias,
)

from numba_cuda_mlir.numba_cuda import cudadecl

registry = Registry()


@registry.register
class PrintFunctionTemplate(AbstractTemplate):
    from numba_cuda_mlir.cuda.print import print as print_intrinsic

    key = print_intrinsic

    def is_valid_print_argument_type(self, arg_type):
        from numba_cuda_mlir.numba_cuda.types.ext_types import Dim3

        if isinstance(arg_type, types.BaseTuple):
            return all(self.is_valid_print_argument_type(t) for t in arg_type.types)
        return isinstance(
            arg_type,
            (
                types.Number,
                types.IntegerLiteral,
                types.Boolean,
                types.Array,
                types.UnicodeType,
                types.StringLiteral,
                Dim3,
            ),
        )

    def generic(self, args, kws):
        for arg in args:
            if not self.is_valid_print_argument_type(arg):
                raise TypeError(f"Invalid argument type for print: {arg=} ({type(arg)=})")

        if "end" in kws:
            end = kws.pop("end")
            if not isinstance(end, (types.StringLiteral, types.UnicodeType)):
                raise TypeError(f"Invalid end type for print: {end=} ({type(end)=})")

        if "sep" in kws:
            sep = kws.pop("sep")
            if not isinstance(sep, (types.StringLiteral, types.UnicodeType)):
                raise TypeError(f"Invalid sep type for print: {sep=} ({type(sep)=})")

        return signature(types.none, *args)


@registry.register
class BuiltinPrintFunctionTemplate(PrintFunctionTemplate):
    key = print


class InlinePTXFunctionTemplate(AbstractTemplate):
    from numba_cuda_mlir.cuda import inline_ptx

    key = inline_ptx

    def generic(self, args, kws):
        from numba_cuda_mlir.lowering_utilities.type_conversions import (
            inline_ptx_type_constraint_to_numba_type,
        )

        if len(args) == 0:
            raise TypeError(f"Invalid arguments to inline_ptx: {args=}")
        force_args_to_be_literal = set()

        format, *rest = args
        if not isinstance(format, types.Literal):
            force_args_to_be_literal.add(0)

        if len(rest) % 2 != 0:
            raise TypeError(f"Invalid arguments to inline_ptx: {args=}")

        result_types = []
        arg_types = [types.string]

        for i, (fmt, _arg) in enumerate(zip(rest[::2], rest[1::2])):
            if not isinstance(fmt, types.Literal):
                force_args_to_be_literal.add(1 + i * 2)
            else:
                fmt_string = fmt.literal_value
                arg_type = inline_ptx_type_constraint_to_numba_type(fmt_string[-1])
                arg_types.extend([fmt, arg_type])
                if "=" in fmt_string:
                    # write-only arguments are returned
                    result_types.append(arg_type)

        if len(force_args_to_be_literal) > 0:
            from numba_cuda_mlir.numba_cuda.core.errors import ForceLiteralArg

            raise ForceLiteralArg(force_args_to_be_literal)

        if len(result_types) == 0:
            result_type = types.none
        elif len(result_types) == 1:
            # Single output: return scalar type
            result_type = result_types[0]
        else:
            # Multiple outputs: return tuple
            result_type = types.Tuple(result_types)
        return signature(result_type, *arg_types)


@registry.register
class PointerCastTemplate(AbstractTemplate):
    from numba_cuda_mlir.types import ptr

    key = ptr

    def generic(self, args, kws):
        if len(args) != 1:
            raise TypeError(f"ptr() takes exactly 1 argument ({len(args)} given)")
        if isinstance(args[0], (types.Array, types.CPointer, types.Integer, types.AggregateType)):
            return signature(types.CPointer(types.none), args[0])
        else:
            raise TypeError(f"Invalid argument type for pointer cast: {args[0]=}")


class SyncthreadsTemplate(ConcreteTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.syncthreads
    cases = [types.none()]


class NumbaSyncthreadsTemplate(SyncthreadsTemplate):
    from numba_cuda_mlir.numba_cuda import syncthreads

    key = syncthreads


class ThisGridTemplate(ConcreteTemplate):
    from numba_cuda_mlir.numba_cuda.cg import this_grid
    from numba_cuda_mlir.numba_cuda.types.ext_types import grid_group

    key = this_grid
    cases = [signature(grid_group)]


class GridGroupSyncTemplate(AbstractTemplate):
    key = "GridGroup.sync"

    def generic(self, args, kws):
        if len(args) == 0:
            return signature(types.int32, recvr=self.this)
        return None


class ShflSyncTemplateBase(AbstractTemplate):
    def generic(self, args, kws):
        if len(args) != 3:
            return None
        mask, value, src_lane = args
        if not isinstance(mask, types.Integer):
            return None
        if not isinstance(value, (types.Integer, types.Float)):
            return None
        if not isinstance(src_lane, types.Integer):
            return None
        return signature(value, mask, value, src_lane)


class ShflSyncTemplate(ShflSyncTemplateBase):
    from numba_cuda_mlir import cuda

    key = cuda.shfl_sync


class ShflUpSyncTemplate(ShflSyncTemplateBase):
    from numba_cuda_mlir import cuda

    key = cuda.shfl_up_sync


class ShflDownSyncTemplate(ShflSyncTemplateBase):
    from numba_cuda_mlir import cuda

    key = cuda.shfl_down_sync


class ShflXorSyncTemplate(ShflSyncTemplateBase):
    from numba_cuda_mlir import cuda

    key = cuda.shfl_xor_sync


class GenericArrayTemplate(CallableTemplate):
    # Subclasses can set this to True to allow non-literal shapes (e.g., for dynamic shared memory)
    allow_dynamic_shape = False

    def generic(self):
        allow_dynamic = self.allow_dynamic_shape

        def typer(shape, dtype, alignment=None, alignas=None):
            # By default, only integer literals and tuples of integer literals are valid shapes
            # This matches numba-cuda's Cuda_array_decl behavior for static allocations.
            # Shared arrays can use dynamic shapes when allow_dynamic_shape=True.
            if isinstance(shape, types.Integer):
                if not allow_dynamic and not isinstance(shape, types.IntegerLiteral):
                    return None
                ndim = 1
            elif isinstance(shape, (types.Tuple, types.UniTuple)):
                if not allow_dynamic and any(
                    not isinstance(s, types.IntegerLiteral) for s in shape
                ):
                    return None
                ndim = len(shape)
            else:
                return None

            # Support both 'alignment' (numba-cuda) and 'alignas' (numba_cuda_mlir) keywords
            align_val = alignment if alignment is not None else alignas

            if align_val is not None:
                permitted = (types.IntegerLiteral, types.NoneType)
                if not isinstance(align_val, permitted):
                    from numba_cuda_mlir.numba_cuda.core.errors import (
                        RequireLiteralValue,
                    )

                    raise RequireLiteralValue("alignment must be a constant integer")

            # Parse dtype - handle DTypeSpec, TypeRef, and StringLiteral
            if isinstance(dtype, types.DTypeSpec):
                nb_dtype = dtype.dtype
            elif isinstance(dtype, types.TypeRef):
                nb_dtype = dtype.instance_type
            elif isinstance(dtype, types.StringLiteral):
                import numpy as np
                from numba_cuda_mlir.numba_cuda.core.errors import TypingError
                from numba_cuda_mlir.numba_cuda.np.numpy_support import from_dtype

                try:
                    dt = np.dtype(dtype.literal_value)
                except TypeError:
                    raise TypingError(f"Invalid NumPy dtype specified: '{dtype.literal_value}'")
                nb_dtype = from_dtype(dt)
            else:
                return None

            if nb_dtype is not None and ndim is not None:
                return types.Array(dtype=nb_dtype, ndim=ndim, layout="C")

            return None

        return typer


class ConstArrayLikeTemplate(AbstractTemplate):
    import numba_cuda_mlir.cuda

    key = numba_cuda_mlir.cuda.const.array_like

    def generic(self, args, kws):
        if len(args) == 1 and isinstance(args[0], types.Array):
            arr = args[0]
            # const.array_like returns an array with same shape and dtype
            return signature(types.Array(dtype=arr.dtype, ndim=arr.ndim, layout=arr.layout), arr)
        return None


@registry.register
class ConstArrayTemplate(ConstArrayLikeTemplate):
    pass


@registry.register
class LocalArrayTemplate(GenericArrayTemplate):
    import numba_cuda_mlir.cuda

    key = numba_cuda_mlir.cuda.local.array


@registry.register_attr
class CUDALocalModule(AttributeTemplate):
    import numba_cuda_mlir.cuda.local as local

    key = local

    def resolve_array(self, mod):
        return types.Function(LocalArrayTemplate)


@registry.register
class SharedArrayTemplate(GenericArrayTemplate):
    import numba_cuda_mlir.cuda

    key = numba_cuda_mlir.cuda.shared.array
    # Enable dynamic shapes for shared arrays (uses extern/dynamic shared memory at runtime)
    allow_dynamic_shape = True


@registry.register_attr
class CUDASharedModule(AttributeTemplate):
    import numba_cuda_mlir.cuda

    key = numba_cuda_mlir.cuda.shared

    def resolve_array(self, mod):
        return types.Function(SharedArrayTemplate)


@registry.register_attr
class CudaSharedModuleTemplate(AttributeTemplate):
    import numba_cuda_mlir.cuda

    key = types.Module(numba_cuda_mlir.cuda.shared)

    def resolve_array(self, mod):
        return types.Function(SharedArrayTemplate)


@registry.register_attr
class CudaConstModuleTemplate(AttributeTemplate):
    import numba_cuda_mlir.cuda

    key = types.Module(numba_cuda_mlir.cuda.const)

    def resolve_array_like(self, mod):
        return types.Function(ConstArrayTemplate)


@registry.register_attr
class CudaLocalModuleTemplate(AttributeTemplate):
    import numba_cuda_mlir.cuda

    key = types.Module(numba_cuda_mlir.cuda.local)

    def resolve_array(self, mod):
        return types.Function(LocalArrayTemplate)


@registry.register_attr
class CudaFP16ModuleTemplate(AttributeTemplate):
    from numba_cuda_mlir.cuda import fp16

    key = types.Module(fp16)

    @lru_cache
    def resolve(self, typ: types.Module, attr: str):
        func = getattr(typ.pymod, attr)
        sig = inspect.signature(func)
        params = sig.parameters.values()
        param_types = [param.annotation for param in params]
        return_type = sig.return_annotation

        class FP16FunctionTemplate(ConcreteTemplate):
            key = func
            cases = [signature(return_type, *param_types)]

        return types.Function(FP16FunctionTemplate)


@registry.register
class CudaGridTemplate(AbstractTemplate):
    import numba_cuda_mlir.cuda

    key = numba_cuda_mlir.cuda.grid

    def generic(self, args, kws):
        assert not kws
        assert len(args) == 1
        if not isinstance(args[0], types.Literal):
            raise ForceLiteralArg(arg_indices={0})
        if not isinstance(args[0], types.Integer):
            return None
        value = args[0].literal_value
        if value == 1:
            return signature(types.int32, types.int32)
        ret_types = [types.int32 for i in range(value)]
        ret_ty = types.Tuple(ret_types)
        return signature(ret_ty, types.int32)


@registry.register
class CudaGridsizeTemplate(CudaGridTemplate):
    import numba_cuda_mlir.cuda

    key = numba_cuda_mlir.cuda.gridsize


class BitIntrinsicTemplate(AbstractTemplate):
    """Base template for bit manipulation intrinsics (clz, ffs, popc).

    These intrinsics take an integer and return int32.
    The input is NOT widened - operation happens at the input's bitwidth.
    """

    def generic(self, args, kws):
        if len(args) != 1:
            return None
        arg = args[0]
        if not isinstance(arg, types.Integer):
            return None
        return signature(types.int32, arg)


class ClzTemplate(BitIntrinsicTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.clz


class FfsTemplate(BitIntrinsicTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.ffs


class BrevTemplate(AbstractTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.brev

    def generic(self, args, kws):
        if len(args) != 1:
            return None
        arg = args[0]
        if not isinstance(arg, types.Integer):
            return None
        # brev returns the same type as input (reversed bits)
        return signature(arg, arg)


class PopcTemplate(BitIntrinsicTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.popc


class SelpTemplate(AbstractTemplate):
    """Select based on predicate: returns a if cond is true, else b."""

    from numba_cuda_mlir import cuda

    key = cuda.selp

    def generic(self, args, kws):
        if len(args) != 3:
            return None
        cond, a, b = args
        # cond must be integer or boolean
        if not isinstance(cond, (types.Integer, types.Boolean)):
            return None
        # a and b must be same type (or compatible)
        if isinstance(a, types.Integer) and isinstance(b, types.Integer):
            # Return the wider type
            if a.bitwidth >= b.bitwidth:
                return signature(a, cond, a, b)
            else:
                return signature(b, cond, a, b)
        elif isinstance(a, types.Float) and isinstance(b, types.Float):
            if a.bitwidth >= b.bitwidth:
                return signature(a, cond, a, b)
            else:
                return signature(b, cond, a, b)
        elif type(a) == type(b):
            return signature(a, cond, a, b)
        return None


def register_bit_intrinsics():
    from numba_cuda_mlir import cuda

    registry.register_global(cuda.clz)(ClzTemplate)
    registry.register_global(cuda.ffs)(FfsTemplate)
    registry.register_global(cuda.brev)(BrevTemplate)
    registry.register_global(cuda.popc)(PopcTemplate)
    registry.register_global(cuda.selp)(SelpTemplate)


register_bit_intrinsics()


@registry.register_attr
class Cuda_stub_resolver(cudadecl.CudaModuleTemplate, AttributeTemplate):
    import numba_cuda_mlir.cuda

    key = types.Module(numba_cuda_mlir.cuda)

    def resolve_grid(self, mod):
        return types.Function(CudaGridTemplate)

    def resolve_gridsize(self, mod):
        return types.Function(CudaGridsizeTemplate)

    def resolve_fp16(self, mod):
        from numba_cuda_mlir.cuda import fp16

        return types.Module(fp16)

    def resolve_local(self, mod):
        import numba_cuda_mlir.cuda

        return types.Module(numba_cuda_mlir.cuda.local)

    def resolve_syncthreads(self, mod):
        return types.Function(SyncthreadsTemplate)

    def resolve_shared(self, mod):
        import numba_cuda_mlir.cuda

        return types.Module(numba_cuda_mlir.cuda.shared)

    def resolve_const(self, mod):
        import numba_cuda_mlir.cuda

        return types.Module(numba_cuda_mlir.cuda.const)

    def resolve_shared_array(self, mod):
        return types.Function(SharedArrayTemplate)

    def resolve_local_array(self, mod):
        return types.Function(LocalArrayTemplate)

    def resolve_experimental(self, mod):
        import numba_cuda_mlir.cuda.experimental as experimental

        return types.Module(experimental)

    def resolve_intrin(self, mod):
        import numba_cuda_mlir.cuda.intrin as intrin

        return types.Module(intrin)

    def resolve_print(self, mod):
        return types.Function(PrintFunctionTemplate)

    def resolve_libdevice(self, mod):
        import numba_cuda_mlir.cuda.libdevice as libdevice

        return types.Module(libdevice)

    def resolve_shfl_sync(self, mod):
        return types.Function(ShflSyncTemplate)

    def resolve_shfl_up_sync(self, mod):
        return types.Function(ShflUpSyncTemplate)

    def resolve_shfl_down_sync(self, mod):
        return types.Function(ShflDownSyncTemplate)

    def resolve_shfl_xor_sync(self, mod):
        return types.Function(ShflXorSyncTemplate)

    def resolve_clz(self, mod):
        return types.Function(ClzTemplate)

    def resolve_ffs(self, mod):
        return types.Function(FfsTemplate)

    def resolve_brev(self, mod):
        return types.Function(BrevTemplate)

    def resolve_popc(self, mod):
        return types.Function(PopcTemplate)

    def resolve_selp(self, mod):
        return types.Function(SelpTemplate)

    def resolve_warpsize(self, mod):
        return types.int32

    def resolve_laneid(self, mod):
        return types.int32

    def resolve_all_sync(self, mod):
        return types.Function(AllSyncTemplate)

    def resolve_any_sync(self, mod):
        return types.Function(AnySyncTemplate)

    def resolve_eq_sync(self, mod):
        return types.Function(EqSyncTemplate)

    def resolve_ballot_sync(self, mod):
        return types.Function(BallotSyncTemplate)

    def resolve_match_any_sync(self, mod):
        return types.Function(MatchAnySyncTemplate)

    def resolve_match_all_sync(self, mod):
        return types.Function(MatchAllSyncTemplate)

    def resolve_activemask(self, mod):
        return types.Function(ActivemaskTemplate)

    def resolve_ldca(self, mod):
        return types.Function(LdcaTemplate)

    def resolve_ldcg(self, mod):
        return types.Function(LdcgTemplate)

    def resolve_ldcs(self, mod):
        return types.Function(LdcsTemplate)

    def resolve_ldlu(self, mod):
        return types.Function(LdluTemplate)

    def resolve_ldcv(self, mod):
        return types.Function(LdcvTemplate)

    def resolve_stcg(self, mod):
        return types.Function(StcgTemplate)

    def resolve_stcs(self, mod):
        return types.Function(StcsTemplate)

    def resolve_stwb(self, mod):
        return types.Function(StwbTemplate)

    def resolve_stwt(self, mod):
        return types.Function(StwtTemplate)

    def resolve_vector(self, mod):
        from numba_cuda_mlir.cuda import vector

        return types.Module(vector)

    def resolve_cg(self, mod):
        import numba_cuda_mlir.numba_cuda.cg as cg

        return types.Module(cg)

    def generic_resolve(self, mod, attr):
        from numba_cuda_mlir.numba_cuda.typing.typeof import typeof

        # Handle vector type constructors (float64x4, float32x2, int32x4, etc.)
        stub = vector_types_by_name.get(attr) or vector_types_by_alias.get(attr)
        if stub is not None:
            return typeof(stub)

        return None


@registry.register_attr
class CudaModuleTemplate(Cuda_stub_resolver):
    from numba_cuda_mlir import cuda

    key = types.Module(cuda)


@registry.register_attr
class RealNumbaCudaModuleTemplate(Cuda_stub_resolver):
    import numba_cuda_mlir.numba_cuda as numba_cuda_module

    key = types.Module(numba_cuda_module)


@registry.register_attr
class CudaExperimentalModuleTemplate(AttributeTemplate):
    import numba_cuda_mlir.cuda.experimental as experimental

    key = types.Module(experimental)

    def resolve_inline_ptx(self, mod):
        return types.Function(InlinePTXFunctionTemplate)

    def resolve_intrin(self, mod):
        import numba_cuda_mlir.cuda.intrin as intrin

        return types.Module(intrin)


@registry.register_attr
class CgModuleTemplate(AttributeTemplate):
    import numba_cuda_mlir.numba_cuda.cg as cg

    key = types.Module(cg)

    def resolve_this_grid(self, mod):
        return types.Function(ThisGridTemplate)


@registry.register_attr
class GridGroupAttributeTemplate(AttributeTemplate):
    from numba_cuda_mlir.numba_cuda.types.ext_types import GridGroup as GridGroupClass

    key = GridGroupClass

    def resolve_sync(self, grid_group_type):
        return types.BoundFunction(GridGroupSyncTemplate, grid_group_type)


def register_syncthreads_variants():
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir._mlir.dialects import nvvm

    for intrin, _reduction_op in [
        (cuda.syncthreads_and, nvvm.BarrierReduction.AND),
        (cuda.syncthreads_or, nvvm.BarrierReduction.OR),
        (cuda.syncthreads_count, nvvm.BarrierReduction.POPC),
    ]:

        class STTy(ConcreteTemplate):
            cases = [types.i4(types.i4)]

        registry.register_global(intrin)(STTy)

    registry.register_global(cuda.syncthreads)(SyncthreadsTemplate)


register_syncthreads_variants()


def register_shfl_sync_intrinsics():
    from numba_cuda_mlir import cuda

    registry.register_global(cuda.shfl_sync)(ShflSyncTemplate)
    registry.register_global(cuda.shfl_up_sync)(ShflUpSyncTemplate)
    registry.register_global(cuda.shfl_down_sync)(ShflDownSyncTemplate)
    registry.register_global(cuda.shfl_xor_sync)(ShflXorSyncTemplate)


register_shfl_sync_intrinsics()


class VoteSyncPredicateTemplate(AbstractTemplate):
    """Template for vote_sync operations that return a boolean predicate (all_sync, any_sync, eq_sync)."""

    def generic(self, args, kws):
        if len(args) != 2:
            return None
        mask, predicate = args
        if not isinstance(mask, types.Integer):
            return None
        return signature(types.boolean, mask, predicate)


class AllSyncTemplate(VoteSyncPredicateTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.all_sync


class AnySyncTemplate(VoteSyncPredicateTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.any_sync


class EqSyncTemplate(VoteSyncPredicateTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.eq_sync


class BallotSyncTemplate(AbstractTemplate):
    """Template for ballot_sync which returns a mask of threads."""

    from numba_cuda_mlir import cuda

    key = cuda.ballot_sync

    def generic(self, args, kws):
        if len(args) != 2:
            return None
        mask, predicate = args
        if not isinstance(mask, types.Integer):
            return None
        return signature(types.uint32, mask, predicate)


class MatchAnySyncTemplate(AbstractTemplate):
    """Template for match_any_sync which returns a mask of threads with same value."""

    from numba_cuda_mlir import cuda

    key = cuda.match_any_sync

    def generic(self, args, kws):
        if len(args) != 2:
            return None
        mask, value = args
        if not isinstance(mask, types.Integer):
            return None
        if not isinstance(value, (types.Integer, types.Float)):
            return None
        return signature(types.uint32, mask, value)


class MatchAllSyncTemplate(AbstractTemplate):
    """Template for match_all_sync which returns (mask, predicate) tuple."""

    from numba_cuda_mlir import cuda

    key = cuda.match_all_sync

    def generic(self, args, kws):
        if len(args) != 2:
            return None
        mask, value = args
        if not isinstance(mask, types.Integer):
            return None
        if not isinstance(value, (types.Integer, types.Float)):
            return None
        return signature(types.Tuple([types.uint32, types.boolean]), mask, value)


class ActivemaskTemplate(AbstractTemplate):
    """Template for activemask which returns the mask of active threads."""

    from numba_cuda_mlir import cuda

    key = cuda.activemask

    def generic(self, args, kws):
        if len(args) != 0:
            return None
        return signature(types.uint32)


def register_vote_sync_intrinsics():
    from numba_cuda_mlir import cuda

    registry.register_global(cuda.all_sync)(AllSyncTemplate)
    registry.register_global(cuda.any_sync)(AnySyncTemplate)
    registry.register_global(cuda.eq_sync)(EqSyncTemplate)
    registry.register_global(cuda.ballot_sync)(BallotSyncTemplate)
    registry.register_global(cuda.match_any_sync)(MatchAnySyncTemplate)
    registry.register_global(cuda.match_all_sync)(MatchAllSyncTemplate)
    registry.register_global(cuda.activemask)(ActivemaskTemplate)


register_vote_sync_intrinsics()


# Cache hint constraint map for PTX inline assembly
CACHE_HINT_CONSTRAINT_MAP = {1: "b", 8: "r", 16: "h", 32: "r", 64: "l", 128: "q"}


def _validate_cache_hint_args(instruction, array, index):
    """Validate arguments for cache hint load/store operations."""
    from numba_cuda_mlir.numba_cuda.core.errors import TypingError

    is_array = isinstance(array, types.Array)
    is_pointer = isinstance(array, types.CPointer)
    if not (is_array or is_pointer):
        raise TypingError(f"{instruction} operates on arrays or pointers. Got type {array}")

    if isinstance(index, types.Integer):
        if is_array and array.ndim != 1:
            raise TypingError(f"Expected {array.ndim} indices, got a scalar")
        return True

    if isinstance(index, types.UniTuple):
        if is_pointer:
            raise TypingError(f"Pointers only support scalar indexing, got tuple of {index.count}")
        if index.count != array.ndim:
            raise TypingError(f"Expected {array.ndim} indices, got {index.count}")
        if isinstance(index.dtype, types.Integer):
            return True

    raise TypingError(f"{index} is not a valid index")


def _validate_cache_hint_dtype(instruction, array):
    """Validate dtype for cache hint operations."""
    from numba_cuda_mlir.numba_cuda.core.errors import TypingError

    dtype = array.dtype
    if not isinstance(dtype, (types.Integer, types.Float)):
        raise TypingError(f"{instruction} requires array of integer or float type, got {dtype}")
    bitwidth = dtype.bitwidth
    if bitwidth not in CACHE_HINT_CONSTRAINT_MAP:
        valid_widths = sorted(CACHE_HINT_CONSTRAINT_MAP.keys())
        raise TypingError(
            f"{instruction} requires array dtype with bitwidth in {valid_widths}, "
            f"got bitwidth {bitwidth}"
        )


class CacheHintLoadTemplate(AbstractTemplate):
    """Base template for cache hint load operations (ldca, ldcg, ldcs, ldlu, ldcv)."""

    instruction = None  # Override in subclasses

    def generic(self, args, kws):
        if len(args) != 2:
            return None
        array, index = args
        _validate_cache_hint_args(self.instruction, array, index)
        _validate_cache_hint_dtype(self.instruction, array)
        return signature(array.dtype, array, index)


class CacheHintStoreTemplate(AbstractTemplate):
    """Base template for cache hint store operations (stcg, stcs, stwb, stwt)."""

    instruction = None  # Override in subclasses

    def generic(self, args, kws):
        if len(args) != 3:
            return None
        array, index, value = args
        _validate_cache_hint_args(self.instruction, array, index)
        _validate_cache_hint_dtype(self.instruction, array)
        return signature(types.void, array, index, value)


class LdcaTemplate(CacheHintLoadTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.ldca
    instruction = "ldca"


class LdcgTemplate(CacheHintLoadTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.ldcg
    instruction = "ldcg"


class LdcsTemplate(CacheHintLoadTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.ldcs
    instruction = "ldcs"


class LdluTemplate(CacheHintLoadTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.ldlu
    instruction = "ldlu"


class LdcvTemplate(CacheHintLoadTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.ldcv
    instruction = "ldcv"


class StcgTemplate(CacheHintStoreTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.stcg
    instruction = "stcg"


class StcsTemplate(CacheHintStoreTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.stcs
    instruction = "stcs"


class StwbTemplate(CacheHintStoreTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.stwb
    instruction = "stwb"


class StwtTemplate(CacheHintStoreTemplate):
    from numba_cuda_mlir import cuda

    key = cuda.stwt
    instruction = "stwt"


def register_cache_hint_intrinsics():
    from numba_cuda_mlir import cuda

    registry.register_global(cuda.ldca)(LdcaTemplate)
    registry.register_global(cuda.ldcg)(LdcgTemplate)
    registry.register_global(cuda.ldcs)(LdcsTemplate)
    registry.register_global(cuda.ldlu)(LdluTemplate)
    registry.register_global(cuda.ldcv)(LdcvTemplate)
    registry.register_global(cuda.stcg)(StcgTemplate)
    registry.register_global(cuda.stcs)(StcsTemplate)
    registry.register_global(cuda.stwb)(StwbTemplate)
    registry.register_global(cuda.stwt)(StwtTemplate)


register_cache_hint_intrinsics()


# Register cuda.bindings.driver.CUtensorMap if available
try:
    from cuda.bindings import driver

    @typeof.typeof_impl.register(driver.CUtensorMap)
    def typeof_cutensormap(val, c):
        """Type inference for cuda.bindings.driver.CUtensorMap objects

        Returns the numba-cuda-mlir CUTensorMap type so these objects can be passed
        to kernels with grid_constant(CUTensorMap) signatures.
        """
        return types.CUTensorMap

except ImportError:
    # cuda-python not available, skip registration
    pass


@registry.register_global(literally)
class LiterallyTemplate(AbstractTemplate):
    """Template for literally() - forces compile-time constant"""

    def generic(self, args, kws):
        if len(args) == 1:
            arg = args[0]
            # The argument must be a literal
            if not isinstance(arg, types.Literal):
                raise ForceLiteralArg(set([0]))
            # Returns the same literal type
            return signature(arg, arg)
        return None


@registry.register_global(literal_unroll)
class LiteralUnrollTemplate(AbstractTemplate):
    """Template for literal_unroll() - forces loop unrolling"""

    def generic(self, args, kws):
        if len(args) == 1:
            arg = args[0]
            # Simply pass through the argument (typically a range or iterable)
            return signature(arg, arg)
        return None


# Also register for numba.cuda.misc.special.literal_unroll (different object)
@registry.register_global(cuda_literal_unroll)
class CudaLiteralUnrollTemplate(AbstractTemplate):
    """Template for cuda literal_unroll() - forces loop unrolling"""

    def generic(self, args, kws):
        if len(args) == 1:
            arg = args[0]
            return signature(arg, arg)
        return None
