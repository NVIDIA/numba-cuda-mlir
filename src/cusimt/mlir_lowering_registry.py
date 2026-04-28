# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from functools import lru_cache
from typing import Callable, override
from numba.core.imputils import Registry
from numba.core import types
import functools


def _decorate_getattr(impl, ty, attr):
    """
    The upstream Numba version of the getattr lowering registration
    does not use functools.wraps, but we would like to filter out the
    upstream numba lowerings and only apply our own.
    """
    real_impl = impl

    if attr is not None:

        @functools.wraps(real_impl)
        def res(context, builder, typ, value, attr):
            return real_impl(context, builder, typ, value)

    else:

        @functools.wraps(real_impl)
        def res(context, builder, typ, value, attr):
            return real_impl(context, builder, typ, value, attr)

    res.signature = (ty,)
    res.attr = attr
    return res


def _decorate_setattr(impl, ty, attr):
    """
    Similar to _decorate_getattr but for setattr lowerings.
    Preserves __module__ so our lowerings aren't filtered out.
    """
    real_impl = impl

    if attr is not None:

        @functools.wraps(real_impl)
        def res(context, builder, sig, args, attr):
            return real_impl(context, builder, sig, args)

    else:

        @functools.wraps(real_impl)
        def res(context, builder, sig, args, attr):
            return real_impl(context, builder, sig, args, attr)

    # Signature is (target_type, value_type) for setattr lookup
    # Use types.Any for value_type to match any value type
    res.signature = ty, types.Any
    res.attr = attr
    return res


class MLIRLoweringRegistry(Registry):
    """
    Registry for MLIR-based lowering implementations.

    Each lowering module should create its own instance of this registry:
        registry = MLIRLoweringRegistry()
        lower = registry.lower
    """

    def __init__(self):
        super().__init__(name="cusimt")

    def lower_getattr(self, ty, attr):
        """
        Decorate an implementation of __getattr__ for type *ty* and
        the attribute *attr*.

        The decorated implementation will have the signature
        (context, builder, typ, val).
        """

        def decorate(impl):
            return self._decorate_attr(impl, ty, attr, self.getattrs, _decorate_getattr)

        return decorate

    def lower_getattr_generic(self, ty):
        """
        Decorate a generic implementation of __getattr__ for type *ty*.

        This overrides Numba's Registry.lower_getattr_generic to use our
        custom _decorate_getattr which preserves __module__ so that our
        lowerings aren't filtered out by _filter_numba_lowerings.

        The decorated implementation will have the signature
        (context, builder, typ, val, attr).
        """

        def decorate(impl):
            # Use attr=None to indicate generic getattr (handles any attribute)
            return self._decorate_attr(impl, ty, None, self.getattrs, _decorate_getattr)

        return decorate

    def lower_setattr_generic(self, ty):
        """
        Decorate a generic implementation of __setattr__ for type *ty*.

        This overrides Numba's Registry.lower_setattr_generic to use our
        custom _decorate_setattr which preserves __module__ so that our
        lowerings aren't filtered out.

        The decorated implementation will have the signature
        (context, builder, sig, args, attr).
        """

        def decorate(impl):
            # Use attr=None to indicate generic setattr (handles any attribute)
            return self._decorate_attr(impl, ty, None, self.setattrs, _decorate_setattr)

        return decorate
