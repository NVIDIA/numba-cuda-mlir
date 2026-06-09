# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from numba_cuda_mlir.errors import ForceLiteralArg
import operator
import numpy as np
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AttributeTemplate,
    AbstractTemplate,
    signature,
)
from numba_cuda_mlir import types
from numba_cuda_mlir.numba_cuda.typing import npydecl
from numba_cuda_mlir.numba_cuda.typing.npydecl import parse_dtype
from numba_cuda_mlir.lowering_utilities import type_conversions
from numba_cuda_mlir.numba_cuda.np.unsafe.ndarray import to_fixed_tuple

registry = npydecl.registry


@registry.register_global(to_fixed_tuple)
class ToFixedTupleTemplate(AbstractTemplate):
    """Typing template for to_fixed_tuple(array, length)"""

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        array, length = args

        if not isinstance(array, types.Array) or array.ndim != 1:
            return None

        if not isinstance(length, types.IntegerLiteral):
            raise ForceLiteralArg({"length": types.IntegerLiteral})

        tuple_size = int(length.literal_value)
        tuple_type = types.UniTuple(dtype=array.dtype, count=tuple_size)
        return signature(tuple_type, array, length)


@registry.register_global(np.empty)
class NumpyEmptyTemplate(AbstractTemplate):
    """Typing template for np.empty"""

    def generic(self, args, kws):
        if len(args) < 1:
            return None

        shape = args[0]
        dtype = args[1] if len(args) > 1 else types.float64

        # Determine dimensionality from shape
        if isinstance(shape, types.Integer):
            ndim = 1
        elif isinstance(shape, (types.UniTuple, types.Tuple)):
            ndim = shape.count if isinstance(shape, types.UniTuple) else len(shape.types)
        else:
            return None

        # Handle dtype
        if isinstance(dtype, types.DTypeSpec):
            element_type = dtype.dtype
        else:
            element_type = dtype

        restype = types.Array(dtype=element_type, ndim=ndim, layout="C")
        return signature(restype, *args)


@registry.register_global(np.zeros)
class NumpyZerosTemplate(AbstractTemplate):
    """Typing template for np.zeros"""

    def generic(self, args, kws):
        if len(args) < 1:
            return None

        shape = args[0]
        dtype = args[1] if len(args) > 1 else types.float64

        if isinstance(shape, types.Integer):
            ndim = 1
        elif isinstance(shape, (types.UniTuple, types.Tuple)):
            ndim = shape.count if isinstance(shape, types.UniTuple) else len(shape.types)
        else:
            return None

        if isinstance(dtype, types.DTypeSpec):
            element_type = dtype.dtype
        else:
            element_type = dtype

        restype = types.Array(dtype=element_type, ndim=ndim, layout="C")
        return signature(restype, *args)


@registry.register_global(np.ones)
class NumpyOnesTemplate(AbstractTemplate):
    """Typing template for np.ones"""

    def generic(self, args, kws):
        if len(args) < 1:
            return None

        shape = args[0]
        dtype = args[1] if len(args) > 1 else types.float64

        if isinstance(shape, types.Integer):
            ndim = 1
        elif isinstance(shape, (types.UniTuple, types.Tuple)):
            ndim = shape.count if isinstance(shape, types.UniTuple) else len(shape.types)
        else:
            return None

        if isinstance(dtype, types.DTypeSpec):
            element_type = dtype.dtype
        else:
            element_type = dtype

        restype = types.Array(dtype=element_type, ndim=ndim, layout="C")
        return signature(restype, *args)


@registry.register_global(np.full)
class NumpyFullTemplate(AbstractTemplate):
    """Typing template for np.full"""

    def generic(self, args, kws):
        if len(args) < 2:
            return None

        shape = args[0]
        value = args[1]
        dtype = args[2] if len(args) > 2 else types.float64

        if isinstance(shape, types.Integer):
            ndim = 1
        elif isinstance(shape, (types.UniTuple, types.Tuple)):
            ndim = shape.count if isinstance(shape, types.UniTuple) else len(shape.types)
        else:
            return None

        if isinstance(dtype, types.DTypeSpec):
            element_type = dtype.dtype
        else:
            element_type = dtype

        restype = types.Array(dtype=element_type, ndim=ndim, layout="C")
        return signature(restype, *args)


# ============================================================================
# NumPy Binary Operations (np.add, np.subtract, np.multiply, np.divide)
# ============================================================================


@registry.register
class NumpyAddTemplate(AbstractTemplate):
    """Typing template for np.add"""

    key = np.add

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        # Handle scalar + scalar
        if isinstance(lhs, types.Number) and isinstance(rhs, types.Number):
            target_dtype = lhs.unify(self.context, rhs)
            if target_dtype is None:
                return None
            return signature(target_dtype, lhs, rhs)

        # Handle array operations
        if isinstance(lhs, types.Array) or isinstance(rhs, types.Array):
            lhs_ndim = lhs.ndim if isinstance(lhs, types.Array) else 0
            rhs_ndim = rhs.ndim if isinstance(rhs, types.Array) else 0
            target_ndim = max(lhs_ndim, rhs_ndim)

            if isinstance(lhs, types.Number):
                target_dtype = rhs.dtype.unify(self.context, lhs)
                target_layout = rhs.layout
            elif isinstance(rhs, types.Number):
                target_dtype = lhs.dtype.unify(self.context, rhs)
                target_layout = lhs.layout
            else:
                target_dtype = lhs.dtype.unify(self.context, rhs.dtype)
                target_layout = lhs.layout

            if target_dtype is None:
                return None

            restype = types.Array(dtype=target_dtype, ndim=target_ndim, layout=target_layout)
            return signature(restype, lhs, rhs)

        return None


@registry.register
class NumpySubtractTemplate(AbstractTemplate):
    """Typing template for np.subtract"""

    key = np.subtract

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        if isinstance(lhs, types.Number) and isinstance(rhs, types.Number):
            target_dtype = lhs.unify(self.context, rhs)
            if target_dtype is None:
                return None
            return signature(target_dtype, lhs, rhs)

        if isinstance(lhs, types.Array) or isinstance(rhs, types.Array):
            lhs_ndim = lhs.ndim if isinstance(lhs, types.Array) else 0
            rhs_ndim = rhs.ndim if isinstance(rhs, types.Array) else 0
            target_ndim = max(lhs_ndim, rhs_ndim)

            if isinstance(lhs, types.Number):
                target_dtype = rhs.dtype.unify(self.context, lhs)
                target_layout = rhs.layout
            elif isinstance(rhs, types.Number):
                target_dtype = lhs.dtype.unify(self.context, rhs)
                target_layout = lhs.layout
            else:
                target_dtype = lhs.dtype.unify(self.context, rhs.dtype)
                target_layout = lhs.layout

            if target_dtype is None:
                return None

            restype = types.Array(dtype=target_dtype, ndim=target_ndim, layout=target_layout)
            return signature(restype, lhs, rhs)

        return None


@registry.register
class NumpyMultiplyTemplate(AbstractTemplate):
    """Typing template for np.multiply"""

    key = np.multiply

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        if isinstance(lhs, types.Number) and isinstance(rhs, types.Number):
            target_dtype = lhs.unify(self.context, rhs)
            if target_dtype is None:
                return None
            return signature(target_dtype, lhs, rhs)

        if isinstance(lhs, types.Array) or isinstance(rhs, types.Array):
            lhs_ndim = lhs.ndim if isinstance(lhs, types.Array) else 0
            rhs_ndim = rhs.ndim if isinstance(rhs, types.Array) else 0
            target_ndim = max(lhs_ndim, rhs_ndim)

            if isinstance(lhs, types.Number):
                target_dtype = rhs.dtype.unify(self.context, lhs)
                target_layout = rhs.layout
            elif isinstance(rhs, types.Number):
                target_dtype = lhs.dtype.unify(self.context, rhs)
                target_layout = lhs.layout
            else:
                target_dtype = lhs.dtype.unify(self.context, rhs.dtype)
                target_layout = lhs.layout

            if target_dtype is None:
                return None

            restype = types.Array(dtype=target_dtype, ndim=target_ndim, layout=target_layout)
            return signature(restype, lhs, rhs)

        return None


@registry.register
class NumpyDivideTemplate(AbstractTemplate):
    """Typing template for np.divide"""

    key = np.divide

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        if isinstance(lhs, types.Number) and isinstance(rhs, types.Number):
            target_dtype = lhs.unify(self.context, rhs)
            if target_dtype is None:
                return None
            return signature(target_dtype, lhs, rhs)

        if isinstance(lhs, types.Array) or isinstance(rhs, types.Array):
            lhs_ndim = lhs.ndim if isinstance(lhs, types.Array) else 0
            rhs_ndim = rhs.ndim if isinstance(rhs, types.Array) else 0
            target_ndim = max(lhs_ndim, rhs_ndim)

            if isinstance(lhs, types.Number):
                target_dtype = rhs.dtype.unify(self.context, lhs)
                target_layout = rhs.layout
            elif isinstance(rhs, types.Number):
                target_dtype = lhs.dtype.unify(self.context, rhs)
                target_layout = lhs.layout
            else:
                target_dtype = lhs.dtype.unify(self.context, rhs.dtype)
                target_layout = lhs.layout

            if target_dtype is None:
                return None

            restype = types.Array(dtype=target_dtype, ndim=target_ndim, layout=target_layout)
            return signature(restype, lhs, rhs)

        return None


@registry.register
class NumpyNegativeTemplate(AbstractTemplate):
    """Typing template for np.negative"""

    key = np.negative

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        x = args[0]

        if isinstance(x, types.Array):
            restype = types.Array(dtype=x.dtype, ndim=x.ndim, layout=x.layout)
            return signature(restype, x)

        return None


# ============================================================================
# NumPy Unary Math Functions (absolute, ceil, floor, log, exp)
# ============================================================================


@registry.register
class NumpyAbsoluteTemplate(AbstractTemplate):
    """Typing template for np.absolute"""

    key = np.absolute

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        x = args[0]

        if isinstance(x, types.Number):
            return signature(x, x)
        elif isinstance(x, types.Array):
            restype = types.Array(dtype=x.dtype, ndim=x.ndim, layout=x.layout)
            return signature(restype, x)

        return None


@registry.register
class NumpyCeilTemplate(AbstractTemplate):
    """Typing template for np.ceil"""

    key = np.ceil

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        x = args[0]

        if isinstance(x, types.Number):
            return signature(x, x)
        elif isinstance(x, types.Array):
            restype = types.Array(dtype=x.dtype, ndim=x.ndim, layout=x.layout)
            return signature(restype, x)

        return None


@registry.register
class NumpyFloorTemplate(AbstractTemplate):
    """Typing template for np.floor"""

    key = np.floor

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        x = args[0]

        if isinstance(x, types.Number):
            return signature(x, x)
        elif isinstance(x, types.Array):
            restype = types.Array(dtype=x.dtype, ndim=x.ndim, layout=x.layout)
            return signature(restype, x)

        return None


@registry.register
class NumpyLogTemplate(AbstractTemplate):
    """Typing template for np.log"""

    key = np.log

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        x = args[0]

        if isinstance(x, types.Number):
            return signature(types.float64, x)
        elif isinstance(x, types.Array):
            restype = types.Array(dtype=types.float64, ndim=x.ndim, layout=x.layout)
            return signature(restype, x)

        return None


@registry.register
class NumpyExpTemplate(AbstractTemplate):
    """Typing template for np.exp"""

    key = np.exp

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        x = args[0]

        if isinstance(x, types.Number):
            return signature(types.float64, x)
        elif isinstance(x, types.Array):
            restype = types.Array(dtype=types.float64, ndim=x.ndim, layout=x.layout)
            return signature(restype, x)

        return None


# ============================================================================
# NumPy Matrix Operations (matmul, dot, transpose)
# ============================================================================


@registry.register
class NumpyMatmulTemplate(AbstractTemplate):
    """Typing template for np.matmul"""

    key = np.matmul

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        if isinstance(lhs, types.Array) and isinstance(rhs, types.Array):
            if lhs.ndim == 2 and rhs.ndim == 2:
                target_dtype = lhs.dtype.unify(self.context, rhs.dtype)
                if target_dtype is None:
                    return None

                restype = types.Array(dtype=target_dtype, ndim=2, layout=lhs.layout)
                return signature(restype, lhs, rhs)

        return None


@registry.register_global(np.dot)
class NumpyDotTemplate(AbstractTemplate):
    """Typing template for np.dot"""

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        # Scalar * scalar
        if isinstance(lhs, types.Number) and isinstance(rhs, types.Number):
            target_dtype = lhs.unify(self.context, rhs)
            if target_dtype is None:
                return None
            return signature(target_dtype, lhs, rhs)

        # Scalar * array or array * scalar
        if (isinstance(lhs, types.Number) and isinstance(rhs, types.Array)) or (
            isinstance(rhs, types.Number) and isinstance(lhs, types.Array)
        ):
            if isinstance(lhs, types.Number):
                array_val = rhs
                scalar_val = lhs
            else:
                array_val = lhs
                scalar_val = rhs

            target_dtype = array_val.dtype.unify(self.context, scalar_val)
            if target_dtype is None:
                return None

            restype = types.Array(dtype=target_dtype, ndim=array_val.ndim, layout=array_val.layout)
            return signature(restype, lhs, rhs)

        # Array operations
        if isinstance(lhs, types.Array) and isinstance(rhs, types.Array):
            # 1D @ 1D -> scalar
            if lhs.ndim == 1 and rhs.ndim == 1:
                target_dtype = lhs.dtype.unify(self.context, rhs.dtype)
                if target_dtype is None:
                    return None
                return signature(target_dtype, lhs, rhs)

            # 2D @ 2D -> 2D
            if lhs.ndim == 2 and rhs.ndim == 2:
                target_dtype = lhs.dtype.unify(self.context, rhs.dtype)
                if target_dtype is None:
                    return None
                restype = types.Array(dtype=target_dtype, ndim=2, layout=lhs.layout)
                return signature(restype, lhs, rhs)

            # ND @ 1D or 1D @ ND or ND @ MD
            if lhs.ndim >= 2 and rhs.ndim == 1:
                restype = types.Array(dtype=lhs.dtype, ndim=lhs.ndim - 1, layout=lhs.layout)
                return signature(restype, lhs, rhs)

            if lhs.ndim >= 1 and rhs.ndim >= 2:
                restype = types.Array(
                    dtype=lhs.dtype, ndim=lhs.ndim + rhs.ndim - 2, layout=lhs.layout
                )
                return signature(restype, lhs, rhs)

        return None


@registry.register_global(np.transpose)
class NumpyTransposeTemplate(AbstractTemplate):
    """Typing template for np.transpose"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        x = args[0]

        if isinstance(x, types.Array):
            restype = types.Array(dtype=x.dtype, ndim=x.ndim, layout=x.layout)
            return signature(restype, x)

        return None


# ============================================================================
# NumPy Reduction and Statistical Functions
# ============================================================================


@registry.register_global(np.sum)
class NumpySumTemplate(AbstractTemplate):
    """Typing template for np.sum"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        arr = args[0]

        if isinstance(arr, types.Array):
            # Sum returns scalar of same dtype
            return signature(arr.dtype, arr)

        return None


@registry.register_global(np.any)
class NumpyAnyTemplate(AbstractTemplate):
    """Typing template for np.any"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        arr = args[0]

        if isinstance(arr, types.Array):
            return signature(types.boolean, arr)

        return None


@registry.register_global(np.all)
class NumpyAllTemplate(AbstractTemplate):
    """Typing template for np.all"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        arr = args[0]

        if isinstance(arr, types.Array):
            return signature(types.boolean, arr)

        return None


@registry.register_global(np.var)
class NumpyVarTemplate(AbstractTemplate):
    """Typing template for np.var"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        arr = args[0]

        if isinstance(arr, types.Array):
            return signature(arr.dtype, arr)

        return None


@registry.register_global(np.mean)
class NumpyMeanTemplate(AbstractTemplate):
    """Typing template for np.mean"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        arr = args[0]

        if isinstance(arr, types.Array):
            return signature(arr.dtype, arr)

        return None


@registry.register_global(np.min)
class NumpyMinTemplate(AbstractTemplate):
    """Typing template for np.min"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        arr = args[0]

        if isinstance(arr, types.Array):
            return signature(arr.dtype, arr)

        return None


@registry.register_global(np.max)
class NumpyMaxTemplate(AbstractTemplate):
    """Typing template for np.max"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        arr = args[0]

        if isinstance(arr, types.Array):
            return signature(arr.dtype, arr)

        return None


@registry.register_global(np.prod)
class NumpyProdTemplate(AbstractTemplate):
    """Typing template for np.prod"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        arr = args[0]

        if isinstance(arr, types.Array):
            return signature(arr.dtype, arr)

        return None


@registry.register_global(np.nanmin)
class NumpyNanMinTemplate(AbstractTemplate):
    """Typing template for np.nanmin"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None
        arr = args[0]
        if isinstance(arr, types.Array):
            return signature(arr.dtype, arr)
        return None


@registry.register_global(np.nanmax)
class NumpyNanMaxTemplate(AbstractTemplate):
    """Typing template for np.nanmax"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None
        arr = args[0]
        if isinstance(arr, types.Array):
            return signature(arr.dtype, arr)
        return None


@registry.register_global(np.nansum)
class NumpyNanSumTemplate(AbstractTemplate):
    """Typing template for np.nansum"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None
        arr = args[0]
        if isinstance(arr, types.Array):
            return signature(arr.dtype, arr)
        return None


@registry.register_global(np.nanprod)
class NumpyNanProdTemplate(AbstractTemplate):
    """Typing template for np.nanprod"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None
        arr = args[0]
        if isinstance(arr, types.Array):
            return signature(arr.dtype, arr)
        return None


@registry.register_global(np.nanmean)
class NumpyNanMeanTemplate(AbstractTemplate):
    """Typing template for np.nanmean"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None
        arr = args[0]
        if isinstance(arr, types.Array):
            return signature(arr.dtype, arr)
        return None


@registry.register
class NumpySqrtTemplate(AbstractTemplate):
    """Typing template for np.sqrt"""

    key = np.sqrt

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        x = args[0]

        if isinstance(x, types.Array):
            restype = types.Array(dtype=x.dtype, ndim=x.ndim, layout=x.layout)
            return signature(restype, x)

        return None


@registry.register
class NumpyAbsTemplate(AbstractTemplate):
    """Typing template for np.abs"""

    key = np.abs

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        x = args[0]

        if isinstance(x, types.Array):
            restype = types.Array(dtype=x.dtype, ndim=x.ndim, layout=x.layout)
            return signature(restype, x)

        return None


@registry.register_global(np.reshape)
class NumpyReshapeTemplate(AbstractTemplate):
    """Typing template for np.reshape"""

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        arr, shape = args

        if isinstance(arr, types.Array) and isinstance(shape, (types.Tuple, types.Array)):
            # Reshape returns same dtype, potentially different ndim
            # For now, return same array type
            return signature(arr, arr, shape)

        return None


@registry.register_global(len)
class LenTemplate(AbstractTemplate):
    """Typing template for len() with arrays and tuples"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        arg = args[0]

        if isinstance(arg, (types.Array, types.Tuple, types.UniTuple)):
            return signature(types.intp, arg)

        return None


# ============================================================================
# NumPy Array Methods
# ============================================================================


@registry.register_attr
class ArrayAttributeTemplate(AttributeTemplate):
    """Typing template for array methods"""

    key = types.Array

    def resolve_size(self, arr):
        """Type arr.size -> int (property, not method)"""
        return types.intp

    def resolve_sum(self, arr):
        """Type arr.sum() -> scalar of same dtype"""
        return types.BoundFunction(NumpyArraySumMethodTemplate, arr)

    def resolve_prod(self, arr):
        """Type arr.prod() -> scalar of same dtype"""
        return types.BoundFunction(NumpyArrayProdMethodTemplate, arr)

    def resolve_min(self, arr):
        """Type arr.min() -> scalar of same dtype"""
        return types.BoundFunction(NumpyArrayMinMethodTemplate, arr)

    def resolve_max(self, arr):
        """Type arr.max() -> scalar of same dtype"""
        return types.BoundFunction(NumpyArrayMaxMethodTemplate, arr)

    def resolve_reshape(self, arr):
        """Type arr.reshape(shape) -> array"""
        return types.BoundFunction(NumpyArrayReshapeMethodTemplate, arr)

    def resolve_view(self, arr):
        """Type arr.view(dtype) -> array"""
        return types.BoundFunction(NumpyArrayViewMethodTemplate, arr)

    def resolve_ctypes(self, arr):
        """Type arr.ctypes -> ArrayCTypes"""
        return types.ArrayCTypes(arr)

    def generic_resolve(self, arr, attr):
        """Handle record array field access: arr.field_name -> array view"""
        if isinstance(arr.dtype, types.Record):
            record_type = arr.dtype
            if attr in record_type.fields:
                field_info = record_type.fields[attr]
                field_dtype = field_info.type
                return types.Array(
                    dtype=field_dtype,
                    ndim=arr.ndim,
                    layout="A",
                )
        return None


@registry.register_attr
class ArrayCTypesAttributeTemplate(AttributeTemplate):
    """Typing template for ArrayCTypes attributes"""

    key = types.ArrayCTypes

    def resolve_data(self, ctinfo):
        """Type ctinfo.data -> uintp (pointer as integer)"""
        return types.uintp


class NumpyArraySumMethodTemplate(AbstractTemplate):
    """Typing for arr.sum() method"""

    key = "Array.sum"

    def generic(self, args, kws):
        # arr.sum() takes no arguments
        if len(args) == 0:
            # Get the this (array) from the bound function
            arr = self.this
            return signature(arr.dtype, recvr=self.this)
        return None


class NumpyArrayProdMethodTemplate(AbstractTemplate):
    """Typing for arr.prod() method"""

    key = "Array.prod"

    def generic(self, args, kws):
        if len(args) == 0:
            arr = self.this
            return signature(arr.dtype, recvr=self.this)
        return None


class NumpyArrayMinMethodTemplate(AbstractTemplate):
    """Typing for arr.min() method"""

    key = "Array.min"

    def generic(self, args, kws):
        if len(args) == 0:
            arr = self.this
            return signature(arr.dtype, recvr=self.this)
        return None


class NumpyArrayMaxMethodTemplate(AbstractTemplate):
    """Typing for arr.max() method"""

    key = "Array.max"

    def generic(self, args, kws):
        if len(args) == 0:
            arr = self.this
            return signature(arr.dtype, recvr=self.this)
        return None


class NumpyArrayReshapeMethodTemplate(AbstractTemplate):
    """Typing for arr.reshape(shape) method"""

    key = "Array.reshape"

    def generic(self, args, kws):
        if len(args) == 1:
            arr = self.this
            shape = args[0]
            # Return same array type for now
            return signature(arr, shape, recvr=self.this)
        return None


class NumpyArrayViewMethodTemplate(AbstractTemplate):
    """Typing for arr.view(dtype) method"""

    key = "Array.view"

    def generic(self, args, kws):
        if len(args) == 1:
            arr = self.this
            dtype = args[0]
            element_type = dtype.dtype if isinstance(dtype, types.DTypeSpec) else dtype
            ret = types.Array(element_type, arr.ndim, arr.layout)
            return signature(ret, dtype, recvr=self.this)
        return None


# ============================================================================
# Python Operator Overloads for Arrays
# ============================================================================


@registry.register_global(operator.add)
class OperatorAddTemplate(AbstractTemplate):
    """Typing template for operator.add with arrays"""

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        # If one of the operands is an array, handle array addition
        if isinstance(lhs, types.Array) or isinstance(rhs, types.Array):
            lhs_ndim = lhs.ndim if isinstance(lhs, types.Array) else 0
            rhs_ndim = rhs.ndim if isinstance(rhs, types.Array) else 0
            target_ndim = max(lhs_ndim, rhs_ndim)

            # Determine target dtype and layout
            if isinstance(lhs, types.Number):
                target_dtype = rhs.dtype.unify(self.context, lhs)
                target_layout = rhs.layout
            elif isinstance(rhs, types.Number):
                target_dtype = lhs.dtype.unify(self.context, rhs)
                target_layout = lhs.layout
            else:
                target_dtype = lhs.dtype.unify(self.context, rhs.dtype)
                target_layout = lhs.layout

            if target_dtype is None:
                return None

            restype = types.Array(dtype=target_dtype, ndim=target_ndim, layout=target_layout)
            return signature(restype, lhs, rhs)

        return None


@registry.register_global(operator.iadd)
class OperatorIAddTemplate(AbstractTemplate):
    """Typing template for operator.iadd with arrays"""

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        if isinstance(lhs, types.Array) or isinstance(rhs, types.Array):
            lhs_ndim = lhs.ndim if isinstance(lhs, types.Array) else 0
            rhs_ndim = rhs.ndim if isinstance(rhs, types.Array) else 0
            target_ndim = max(lhs_ndim, rhs_ndim)

            if isinstance(lhs, types.Number):
                target_dtype = rhs.dtype.unify(self.context, lhs)
                target_layout = rhs.layout
            elif isinstance(rhs, types.Number):
                target_dtype = lhs.dtype.unify(self.context, rhs)
                target_layout = lhs.layout
            else:
                target_dtype = lhs.dtype.unify(self.context, rhs.dtype)
                target_layout = lhs.layout

            if target_dtype is None:
                return None

            restype = types.Array(dtype=target_dtype, ndim=target_ndim, layout=target_layout)
            return signature(restype, lhs, rhs)

        return None


@registry.register_global(operator.sub)
class OperatorSubTemplate(AbstractTemplate):
    """Typing template for operator.sub with arrays"""

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        if isinstance(lhs, types.Array) or isinstance(rhs, types.Array):
            lhs_ndim = lhs.ndim if isinstance(lhs, types.Array) else 0
            rhs_ndim = rhs.ndim if isinstance(rhs, types.Array) else 0
            target_ndim = max(lhs_ndim, rhs_ndim)

            if isinstance(lhs, types.Number):
                target_dtype = rhs.dtype.unify(self.context, lhs)
                target_layout = rhs.layout
            elif isinstance(rhs, types.Number):
                target_dtype = lhs.dtype.unify(self.context, rhs)
                target_layout = lhs.layout
            else:
                target_dtype = lhs.dtype.unify(self.context, rhs.dtype)
                target_layout = lhs.layout

            if target_dtype is None:
                return None

            restype = types.Array(dtype=target_dtype, ndim=target_ndim, layout=target_layout)
            return signature(restype, lhs, rhs)

        return None


@registry.register_global(operator.isub)
class OperatorISubTemplate(AbstractTemplate):
    """Typing template for operator.isub with arrays"""

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        if isinstance(lhs, types.Array) or isinstance(rhs, types.Array):
            lhs_ndim = lhs.ndim if isinstance(lhs, types.Array) else 0
            rhs_ndim = rhs.ndim if isinstance(rhs, types.Array) else 0
            target_ndim = max(lhs_ndim, rhs_ndim)

            if isinstance(lhs, types.Number):
                target_dtype = rhs.dtype.unify(self.context, lhs)
                target_layout = rhs.layout
            elif isinstance(rhs, types.Number):
                target_dtype = lhs.dtype.unify(self.context, rhs)
                target_layout = lhs.layout
            else:
                target_dtype = lhs.dtype.unify(self.context, rhs.dtype)
                target_layout = lhs.layout

            if target_dtype is None:
                return None

            restype = types.Array(dtype=target_dtype, ndim=target_ndim, layout=target_layout)
            return signature(restype, lhs, rhs)

        return None


@registry.register_global(operator.mul)
class OperatorMulTemplate(AbstractTemplate):
    """Typing template for operator.mul with arrays"""

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        if isinstance(lhs, types.Array) or isinstance(rhs, types.Array):
            lhs_ndim = lhs.ndim if isinstance(lhs, types.Array) else 0
            rhs_ndim = rhs.ndim if isinstance(rhs, types.Array) else 0
            target_ndim = max(lhs_ndim, rhs_ndim)

            if isinstance(lhs, types.Number):
                target_dtype = rhs.dtype.unify(self.context, lhs)
                target_layout = rhs.layout
            elif isinstance(rhs, types.Number):
                target_dtype = lhs.dtype.unify(self.context, rhs)
                target_layout = lhs.layout
            else:
                target_dtype = lhs.dtype.unify(self.context, rhs.dtype)
                target_layout = lhs.layout

            if target_dtype is None:
                return None

            restype = types.Array(dtype=target_dtype, ndim=target_ndim, layout=target_layout)
            return signature(restype, lhs, rhs)

        return None


@registry.register_global(operator.imul)
class OperatorIMulTemplate(AbstractTemplate):
    """Typing template for operator.imul with arrays"""

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        if isinstance(lhs, types.Array) or isinstance(rhs, types.Array):
            lhs_ndim = lhs.ndim if isinstance(lhs, types.Array) else 0
            rhs_ndim = rhs.ndim if isinstance(rhs, types.Array) else 0
            target_ndim = max(lhs_ndim, rhs_ndim)

            if isinstance(lhs, types.Number):
                target_dtype = rhs.dtype.unify(self.context, lhs)
                target_layout = rhs.layout
            elif isinstance(rhs, types.Number):
                target_dtype = lhs.dtype.unify(self.context, rhs)
                target_layout = lhs.layout
            else:
                target_dtype = lhs.dtype.unify(self.context, rhs.dtype)
                target_layout = lhs.layout

            if target_dtype is None:
                return None

            restype = types.Array(dtype=target_dtype, ndim=target_ndim, layout=target_layout)
            return signature(restype, lhs, rhs)

        return None


@registry.register_global(operator.truediv)
class OperatorTruedivTemplate(AbstractTemplate):
    """Typing template for operator.truediv with arrays"""

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        if isinstance(lhs, types.Array) or isinstance(rhs, types.Array):
            lhs_ndim = lhs.ndim if isinstance(lhs, types.Array) else 0
            rhs_ndim = rhs.ndim if isinstance(rhs, types.Array) else 0
            target_ndim = max(lhs_ndim, rhs_ndim)

            # Get dtypes
            lhs_dtype = lhs.dtype if isinstance(lhs, types.Array) else lhs
            rhs_dtype = rhs.dtype if isinstance(rhs, types.Array) else rhs

            # True division always returns float
            # Get the unified type first
            target_dtype = lhs_dtype.unify(self.context, rhs_dtype)
            if target_dtype is None:
                return None

            # Convert to float if integer
            if isinstance(target_dtype, types.Integer):
                # Use at least float32 for integer division
                bitwidth = max(target_dtype.bitwidth, 32)
                target_dtype = type_conversions.float_of_width(bitwidth)

            # Get layout
            if isinstance(lhs, types.Array):
                target_layout = lhs.layout
            elif isinstance(rhs, types.Array):
                target_layout = rhs.layout
            else:
                target_layout = "C"

            restype = types.Array(dtype=target_dtype, ndim=target_ndim, layout=target_layout)
            return signature(restype, lhs, rhs)

        return None


@registry.register_global(operator.itruediv)
class OperatorITruedivTemplate(AbstractTemplate):
    """Typing template for operator.itruediv with arrays"""

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        if isinstance(lhs, types.Array) or isinstance(rhs, types.Array):
            lhs_ndim = lhs.ndim if isinstance(lhs, types.Array) else 0
            rhs_ndim = rhs.ndim if isinstance(rhs, types.Array) else 0
            target_ndim = max(lhs_ndim, rhs_ndim)

            if isinstance(lhs, types.Number):
                target_dtype = rhs.dtype.unify(self.context, lhs)
                target_layout = rhs.layout
            elif isinstance(rhs, types.Number):
                target_dtype = lhs.dtype.unify(self.context, rhs)
                target_layout = lhs.layout
            else:
                target_dtype = lhs.dtype.unify(self.context, rhs.dtype)
                target_layout = lhs.layout

            if target_dtype is None:
                return None

            restype = types.Array(dtype=target_dtype, ndim=target_ndim, layout=target_layout)
            return signature(restype, lhs, rhs)

        return None


@registry.register_global(operator.neg)
class OperatorNegTemplate(AbstractTemplate):
    """Typing template for operator.neg with arrays"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        x = args[0]

        if isinstance(x, types.Array):
            restype = types.Array(dtype=x.dtype, ndim=x.ndim, layout=x.layout)
            return signature(restype, x)

        return None


@registry.register_global(abs)
class AbsTemplate(AbstractTemplate):
    """Typing template for abs() with arrays"""

    def generic(self, args, kws):
        if len(args) != 1:
            return None

        x = args[0]

        if isinstance(x, types.Array):
            restype = types.Array(dtype=x.dtype, ndim=x.ndim, layout=x.layout)
            return signature(restype, x)

        return None


@registry.register_global(operator.matmul)
class OperatorMatmulTemplate(AbstractTemplate):
    """Typing template for operator.matmul (@) with arrays"""

    def generic(self, args, kws):
        if len(args) != 2:
            return None

        lhs, rhs = args

        # Only handle array @ array
        if isinstance(lhs, types.Array) and isinstance(rhs, types.Array):
            # Handle 1D @ 1D and 2D @ 2D cases
            if (lhs.ndim == 1 and rhs.ndim == 1) or (lhs.ndim == 2 and rhs.ndim == 2):
                target_dtype = lhs.dtype.unify(self.context, rhs.dtype)
                if target_dtype is None:
                    return None

                restype = types.Array(dtype=target_dtype, ndim=lhs.ndim, layout=lhs.layout)
                return signature(restype, lhs, rhs)

        return None
