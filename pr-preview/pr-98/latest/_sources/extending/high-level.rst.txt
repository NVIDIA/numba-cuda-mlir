..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. _high-level-extending:

High-level extension API
========================

Decorators in the High-level API are used to implement compilation of new
Python functions, methods, and attributes without writing any MLIR by hand.
Each decorator registers an *implementation function*: a Python function that
is itself compiled by the same pipeline used for ``@cuda.jit`` kernels. Any
Python code supported by Numba-CUDA-MLIR can be used inside an implementation
function.

Implementation functions are `overloaded`: that is, they are invoked at compile
time with the Numba types of their callable's arguments, and they return a
function implementing the callable for those types. The term `overloaded`
refers to the ability for different implementation functions to be provided for
different input types; the chosen implementation for a given function and set
of argument types is referred to as an `overload` of that function.

All the High-level API decorators are members of
:py:mod:`numba_cuda_mlir.extending`.

Implementing functions
----------------------

The :py:func:`~numba_cuda_mlir.extending.overload` decorator is used to provide
the implementations of a Python callable to be used in a kernel or device
function.

An example of the use of ``@overload``:

.. code-block:: python

   from numba_cuda_mlir import cuda, extending, types

   # A pure Python function. Normally usable only from within Python code; we
   # will make it usable in kernels and device functions with the overloaded
   # implementation below.
   def my_func(x):
       if isinstance(x, int):
           return x + 1
       elif isinstance(x, float):
           return x * 2.0
       else:
           raise NotImplementedError

   # The decorated implementation function.
   @extending.overload(my_func)
   def my_func_overload(x):
       # Different implementations are returned for different argument types.
       # This is because a single unique typing is required for each variable
       # in a Python function. Providing a single implementation for both types
       # would result in `x` being promoted to a float value in the integer case.
       if isinstance(x, types.Integer):
           def impl(x):
               return x + 1
           return impl
       elif isinstance(x, types.Float):
           def impl(x):
               return x * 2.0
           return impl

   @cuda.jit
   def kernel(int_out, float_out):
       int_out[0] = my_func(int_out[0])
       float_out[0] = my_func(float_out[0])

In the above example, the implementation function returns *different* overloads
for different argument types. When there is no implementation for a given set of
argument types, the implementation function returns ``None`` to decline the
overload. This lets Numba-CUDA-MLIR try other implementation functions, until it
finds a matching implementation. If no matching overload is found, then a
compilation error occurs.


Implementing methods
--------------------

:py:func:`~numba_cuda_mlir.extending.overload_method` registers an
implementation function for a method on instances of types supported by
Numba-CUDA-MLIR:

.. code-block:: python

   from numba_cuda_mlir import cuda, extending, types
   import numpy as np

   @extending.overload_method(types.Array, "doubled_first")
   def array_doubled_first(arr):
       def impl(arr):
           return arr[0] * 2
       return impl

   @cuda.jit
   def kernel(arr, out):
       out[0] = arr.doubled_first()

The first argument of the implementation function is the ``self`` object; any
additional parameters become method arguments:

.. code-block:: python

   @extending.overload_method(types.Array, "elem_plus")
   def array_elem_plus(arr, idx, val):
       def impl(arr, idx, val):
           return arr[idx] + val
       return impl

Implementing attributes
-----------------------

:py:func:`~numba_cuda_mlir.extending.overload_attribute` registers an
implementation function for a read-only attribute on types supported by
Numba-CUDA-MLIR. The implementation function takes the type of the ``self``
object and must return a function that computes the attribute value:

.. code-block:: python

   @extending.overload_attribute(types.Array, "doubled_size")
   def array_doubled_size(arr):
       def get(arr):
           return arr.size * 2
       return get

To expose a writable attribute, register a lowering for ``setattr`` through
the low-level API; see :ref:`lowering-getattr-setattr`.

Registering helper functions
----------------------------

:py:func:`~numba_cuda_mlir.extending.register_jitable` marks a regular Python
function as compilable from device code. It is the simplest way to factor
shared logic out of multiple kernels or implementations without having to
write a full overload:

.. code-block:: python

   @extending.register_jitable
   def triple(x):
       return x * 3

   @cuda.jit
   def kernel(arr):
       arr[0] = triple(arr[0])

   # Also works; prints "6"
   print(triple(2))

A ``@register_jitable`` function may itself call other ``@register_jitable``
functions, ``@cuda.jit`` device functions, and any built-in or overloaded
operation supported by Numba-CUDA-MLIR.

.. _high-level-intrinsic:

Implementing intrinsics
-----------------------

The :py:func:`~numba_cuda_mlir.extending.intrinsic` decorator turns a Python
function into a *compiler intrinsic*: a function called at compile time to
both type the call and emit code for it. Intrinsics are the bridge between
the high-level API (writing implementations in Python) and the low-level API
(emitting MLIR directly).

An ``@intrinsic`` implementation function is called with a typing context in
addition to the argument types for the implementation it returns. It must return
a tuple ``(signature, codegen)``, where:

- The ``signature`` object should be a
  :class:`numba_cuda_mlir.cuda.typing.templates.Signature` object.
- The ``codegen`` callable has the same signature as a :ref:`lowering function
  <lowering-functions>` — ``(builder, target, args, kwargs)`` — and is
  responsible for emitting MLIR for the call.

An example of an intrinsic:

.. code-block:: python

   from numba_cuda_mlir import cuda, extending, types
   from numba_cuda_mlir._mlir.dialects import cf, arith
   from numba_cuda_mlir._mlir.extras import types as T

   @extending.intrinsic
   def do_nothing(typingctx, x):
       def codegen(builder, target, args, kwargs):
           true = arith.constant(result=T.bool(), value=1)
           cf.assert_(true, "This should not be executed")
           builder.store_var(target, builder.load_var(args[0]))

       return x(x), codegen

The signature ``x(x)`` constructs a Numba ``Signature`` from the argument
type ``x`` (the return type) and the parameter types ``(x,)``. Inside the
``codegen``, ``builder.load_var`` and ``builder.store_var`` are the canonical
ways to read inputs and write the result; see :ref:`lowering-builder` for the
full builder API.


Type inference for callables
----------------------------

The :py:func:`~numba_cuda_mlir.extending.type_callable` decorator registers a
type-only inference rule for a callable. Unlike ``@overload``, it does not
provide an implementation — it only tells the compiler what the result type
should be. Pair it with a separate ``lowering_registry.lower`` registration
to provide the implementation. This split is useful when the typing logic is
trivial but the lowering is best written in MLIR directly, for example for
constructors of custom types:

.. code-block:: python

   from numba_cuda_mlir import extending, types

   def make_boxed_int(x):
       raise NotImplementedError("only callable inside a kernel")

   @extending.type_callable(make_boxed_int)
   def _type_make_boxed_int(context):
       def typer(x):
           if isinstance(x, types.Integer):
               return my_boxed_int
       return typer

The lowering for ``make_boxed_int`` is registered separately through
:py:attr:`~numba_cuda_mlir.extending.lowering_registry`; a complete worked
example appears in :ref:`lower-cast-example`.

Dispatching on type information
-------------------------------

All of the implementation functions described above run at compile time with the
Numba-CUDA-MLIR types of the call arguments. This is the right place to:

* Inspect ``arr.ndim``, ``arr.dtype``, ``arr.layout`` and similar attributes
  to return specialised implementations.
* Validate inputs and raise a typing error when the call is unsupported (use
  ``raise TypeError(...)`` or ``numba_cuda_mlir.numba_cuda.errors.TypingError``).
* Return ``None`` to decline the overload — another registered overload, or
  the compiler's default, will then be tried.

API reference
-------------

.. py:module:: numba_cuda_mlir.extending

.. py:function:: overload(func, jit_options=MappingProxyType({}), strict=True, inline="never", prefer_literal=False, **kwargs)

   Register an implementation for ``func``. The decorated function is the
   implementation function: it is called at compile time with the Numba types of
   the arguments and must return a Python function (the implementation), or
   ``None`` to decline. The implementation is compiled by Numba-CUDA-MLIR's
   pipeline.

   :param func: The Python callable being overloaded.
   :param jit_options: Options forwarded to ``cuda.jit`` when compiling the
       implementation.
   :param strict: If ``True``, raise when the implementation cannot be
       compiled. If ``False``, the failure is silenced (useful for
       :py:func:`register_jitable`).
   :param inline: Inlining policy: ``"never"``, ``"always"``, or a
       cost-model callable.
   :param prefer_literal: If ``True``, prefer literal-typed arguments when
       resolving the overload.

.. py:function:: overload_method(typ, meth, **kwargs)

   Register an implementation for the method ``meth`` on the type ``typ``. The
   decorated function is an implementation function with the same contract as
   :py:func:`overload`; its first parameter is the ``self`` object.

.. py:function:: overload_attribute(typ, attr, **kwargs)

   Register an implementation for a read-only attribute ``attr`` on the
   type ``typ``. The decorated function is an implementation function whose only
   parameter is the ``self`` object and whose returned implementation is a
   function of the receiver that produces the attribute value.

.. py:function:: register_jitable(*args, **kwargs)

   Mark a Python function as compilable from device code. The function is
   registered as a non-strict overload of itself, so calls to it from inside
   a kernel or device function dispatch to the original Python source
   compiled by Numba-CUDA-MLIR.

.. py:function:: intrinsic(func)

   Register ``func`` as a compiler intrinsic. ``func`` must accept a typing
   context followed by the argument types, and return a tuple
   ``(signature, codegen)`` where ``codegen`` is a lowering function of the
   form ``(builder, target, args, kwargs)``.

   ``intrinsic`` is re-exported from
   :py:mod:`numba_cuda_mlir.numba_cuda.extending`. In Numba-CUDA-MLIR, the
   ``codegen`` callable emits MLIR through :py:mod:`numba_cuda_mlir._mlir`,
   not LLVM IR through ``llvmlite``.

.. py:function:: type_callable(func)

   Register a type-only inference rule for ``func``. The decorated function
   is called at compile time with the typing context and must return a
   ``typer`` function. The typer is invoked with the argument types and must
   return the call's result type, or ``None`` to decline. Combine with a
   separate ``lowering_registry.lower`` registration to provide the
   implementation.
