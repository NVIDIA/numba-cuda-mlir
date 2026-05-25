# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba_cuda_mlir.extending import as_numba_type, overload, typing_registry
from numba_cuda_mlir.numba_cuda.core.errors import NumbaTypeError
from numba_cuda_mlir import types
import functools
import operator

overload = functools.partial(overload, typing_registry=typing_registry)


@overload(operator.truth)
def ol_truth(val):
    if isinstance(val, types.Boolean):

        def impl(val):
            return val

        return impl


@overload(bool)
def bool_sequence(x):
    valid_types = (
        types.CharSeq,
        types.UnicodeCharSeq,
        types.UnicodeType,
    )

    if isinstance(x, valid_types):

        def bool_impl(x):
            return len(x) > 0

        return bool_impl


@overload(bool, inline="always")
def bool_none(x):
    if isinstance(x, types.NoneType) or x is None:
        return lambda x: False


@overload(str)
def ol_str_generic(object=""):
    def impl(object=""):
        attr = "__str__"
        if hasattr(object, attr):
            return getattr(object, attr)()
        else:
            return repr(object)

    return impl


@overload(isinstance)
def ol_isinstance(var, typs):
    def true_impl(var, typs):
        return True

    def false_impl(var, typs):
        return False

    var_ty = as_numba_type(var)

    if isinstance(var_ty, types.Optional):
        msg = f'isinstance cannot handle optional types. Found: "{var_ty}"'
        raise NumbaTypeError(msg)

    # NOTE: The current implementation of `isinstance` restricts the type of the
    # instance variable to types that are well known and in common use. The
    # danger of unrestricted type comparison is that a "default" of `False` is
    # required and this means that if there is a bug in the logic of the
    # comparison tree `isinstance` returns False! It's therefore safer to just
    # reject the compilation as untypable!
    supported_var_ty = (
        types.Number,
        types.Bytes,
        types.RangeType,
        types.Tuple,
        types.UniTuple,
        types.Function,
        types.UnicodeType,
        types.NoneType,
        types.Array,
        types.Boolean,
        types.Float,
        types.UnicodeCharSeq,
        types.Complex,
        types.NPDatetime,
        types.NPTimedelta,
    )
    if not isinstance(var_ty, supported_var_ty):
        msg = f'isinstance() does not support variables of type "{var_ty}".'
        raise NumbaTypeError(msg)

    t_typs = typs

    # Check the types that the var can be an instance of, it'll be a scalar,
    # a unituple or a tuple.
    if isinstance(t_typs, types.UniTuple):
        # corner case - all types in isinstance are the same
        t_typs = t_typs.key[0]

    if not isinstance(t_typs, types.Tuple):
        t_typs = (t_typs,)

    for typ in t_typs:
        if isinstance(typ, types.Function):
            key = typ.key[0]  # functions like int(..), float(..), str(..)
        else:
            key = typ.key

        # corner cases for bytes, range, ...
        # avoid registering those types on `as_numba_type`
        types_not_registered = {
            bytes: types.Bytes,
            range: types.RangeType,
            tuple: types.BaseTuple,
        }
        if key in types_not_registered:
            if isinstance(var_ty, types_not_registered[key]):
                return true_impl
            continue

        if isinstance(typ, types.TypeRef):
            # Use of Numba type classes is in general not supported as they do
            # not work when the jit is disabled.
            if key not in (types.ListType, types.DictType):
                msg = "Numba type classes (except numba.typed.* container types) are not supported."
                raise NumbaTypeError(msg)
            # Case for TypeRef (i.e. isinstance(var, typed.List))
            #      var_ty == ListType[int64] (instance)
            #         typ == types.ListType  (class)
            return true_impl if type(var_ty) is key else false_impl
        else:
            numba_typ = as_numba_type(key)
            if var_ty == numba_typ:
                return true_impl
            elif isinstance(numba_typ, (types.NPDatetime, types.NPTimedelta)):
                if isinstance(var_ty, type(numba_typ)):
                    return true_impl
            elif (
                isinstance(numba_typ, types.ClassType)
                and isinstance(var_ty, types.ClassInstanceType)
                and var_ty.key == numba_typ.instance_type.key
            ):
                # check for jitclasses
                return true_impl
            elif isinstance(numba_typ, types.Container) and numba_typ.key[0] == types.undefined:
                # check for containers (list, tuple, set, ...)
                if isinstance(var_ty, numba_typ.__class__) or (
                    isinstance(var_ty, types.BaseTuple) and isinstance(numba_typ, types.BaseTuple)
                ):
                    return true_impl

    return false_impl
