..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. _low-level-extending:

Low-level extension API
=======================

The low-level API exposes Numba-CUDA-MLIR's extension phases directly:

* **Planning** — after device functions have been inlined and before type
  inference, whole-function planners can inspect and transform the complete
  untyped Numba IR.
* **Typing** — running before code generation, type inference resolves each
  variable, attribute, and call to a type. Extension authors register new
  typing rules with the :py:mod:`~numba_cuda_mlir.numba_cuda.typing` template
  machinery.
* **Lowering** — once every value has a type, lowering emits MLIR for each
  operation. Extension authors register lowering implementations with
  :py:attr:`~numba_cuda_mlir.extending.lowering_registry`.

Typing and lowering also underpin the high-level API. The high-level
decorators are convenience wrappers that arrange those phases for you; use
the low-level API when you need whole-function planning, a brand-new type,
direct MLIR emission, or anything else that is difficult to express using the
high-level API.

The lowering half of this API differs significantly from Numba's. Numba's
lowering callbacks receive an ``llvmlite.ir.IRBuilder`` and emit LLVM IR.
Numba-CUDA-MLIR's lowering callbacks receive an MLIR builder
(:py:class:`numba_cuda_mlir.mlir_lowering.MLIRLower`) and emit MLIR through
the bindings in :py:mod:`numba_cuda_mlir._mlir`.

Whole-function planning
-----------------------

Extensions that need to inspect or transform a complete function before typing
can register a :py:class:`~numba_cuda_mlir.extending.WholeFunctionPlanner`.
Planners run once after inlineable device functions have been merged into the
caller and before type inference. This makes calls introduced by device
helpers visible without rerunning unrelated rewrite registries.

Implement ``run()`` and return ``True`` only when the planner changed
``state.func_ir``. Before the first planner and after each change,
Numba-CUDA-MLIR simplifies the control flow graph, refreshes postprocessing
state and definitions, and verifies every block. Planners run for both kernels
and device-function compilations; a planner that only applies to kernels
should check ``self.is_device_function`` and decline device functions.

.. code-block:: python

   from numba_cuda_mlir.extending import (
       WholeFunctionPlanner,
       register_planner,
   )

   @register_planner
   class MyPlanner(WholeFunctionPlanner):
       def run(self):
           modified = rewrite_my_extension_calls(self.state.func_ir)
           return bool(modified)

Registration order is preserved, and registering the same planner class more
than once has no effect. The planner pass snapshots its registry before
running, so planners registered after that snapshot, including from a running
planner, start with the next compilation. Planners operate on untyped Numba
IR; typing and MLIR lowering extensions remain responsible for the later
compilation phases.

Persistent disk-cache load and save are bypassed while planners are
registered because planner identity and implementation are not part of the
cache key. In-memory overload reuse is unchanged.

.. autoclass:: numba_cuda_mlir.extending.WholeFunctionPlanner
   :members: is_device_function, run

.. autofunction:: numba_cuda_mlir.extending.register_planner

Typing
------

Typing rules in Numba-CUDA-MLIR are written exactly as they are in Numba: you
subclass a typing *template* and register it on a typing registry. The relevant
building blocks are in :py:mod:`numba_cuda_mlir.numba_cuda.typing.templates`:

* :py:class:`AbstractTemplate` — most general; implement ``generic(args, kws)``
  and return a :py:func:`signature` (or ``None``) based on the argument types.
* :py:class:`ConcreteTemplate` — for a fixed list of supported signatures.
* :py:class:`AttributeTemplate` — for typing attribute access.

Each lowering module in Numba-CUDA-MLIR should hold its own typing registry.
For your own extensions, use the shared registry exposed from
:py:mod:`numba_cuda_mlir.extending`:

.. code-block:: python

   from numba_cuda_mlir.extending import typing_registry
   from numba_cuda_mlir.numba_cuda.typing.templates import (
       AbstractTemplate, signature,
   )
   from numba_cuda_mlir import types
   import operator

   @typing_registry.register_global(operator.setitem)
   class Uint64PointerSetitemTemplate(AbstractTemplate):
       def generic(self, args, kws):
           if len(args) != 3:
               return None
           array, idx, value = args
           if (isinstance(array, types.Array)
                   and array.dtype == types.uint64
                   and isinstance(idx, types.Integer)
                   and value is my_pointer_type):
               return signature(types.none, array, idx, value)
           return None

Two common forms of registration on the typing registry are:

* ``@typing_registry.register_global(callable_or_op)`` — register a template
  that types a global callable or operator (e.g. ``operator.setitem``).
* ``@typing_registry.register_attr`` — register an :py:class:`AttributeTemplate`
  for attribute access on a type.

Inside a template, ``signature(return_type, *arg_types)`` constructs the
Numba ``Signature`` for a successful match. Returning ``None`` declines the
match (other templates are tried). To force a literal to be resolved before
typing proceeds, raise
:py:class:`numba_cuda_mlir.errors.ForceLiteralArg`.


Defining new types
------------------

A new type is a subclass of :py:class:`numba_cuda_mlir.types.Type`. For most
user-defined types it is enough to override ``__init__`` and provide a ``key``
property:

.. code-block:: python

   from numba_cuda_mlir import types
   from numba_cuda_mlir.numba_cuda.typeconv import Conversion

   class MyBoxedIntType(types.Type):
       def __init__(self):
           super().__init__(name="MyBoxedInt")

       @property
       def key(self):
           return self.__class__

       def can_convert_to(self, typingctx, other):
           if isinstance(other, types.Integer):
               return Conversion.safe
           return None

   my_boxed_int = MyBoxedIntType()

The optional ``can_convert_to`` hook tells type inference which implicit
conversions are permitted, with their cost (one of ``Conversion.safe``,
``Conversion.promote``, or ``Conversion.unsafe``). When inference inserts an
implicit cast based on ``can_convert_to``, the corresponding ``@lower_cast``
implementation is invoked to produce the MLIR (see :ref:`lowering-cast`).

Data models
~~~~~~~~~~~

Every type must have a *data model* that describes how it is represented in
MLIR. Register one with :py:func:`numba_cuda_mlir.models.register_model`:

.. code-block:: python

   from numba_cuda_mlir.models import PrimitiveModel, register_model
   from numba_cuda_mlir._mlir import ir as mlir_ir
   from numba_cuda_mlir._mlir.dialects import llvm

   @register_model(MyBoxedIntType)
   class MyBoxedIntModel(PrimitiveModel):
       def __init__(self, dmm, fe_type):
           be_type = llvm.StructType.get_literal(
               [mlir_ir.IntegerType.get_signless(64)]
           )
           super().__init__(dmm, fe_type, be_type)

:py:class:`PrimitiveModel` is the simplest model and represents the type as
a single MLIR value. For an aggregate type with named fields, use
:py:class:`~numba_cuda_mlir.models.AggregateTypeModel`; for an
NRT-managed type wrapping a pointer, see the experimental data models in
:py:mod:`numba_cuda_mlir.models`.

.. _lowering-functions:

Lowering
--------

A lowering function is a Python callable registered against a high-level
operation (``operator.add`` on a pair of types, a call to ``np.sum``, a
constructor for a custom type, …) that emits the MLIR for that operation.

The shared registry for user-supplied lowerings is
:py:attr:`numba_cuda_mlir.extending.lowering_registry`. It is an instance of
:py:class:`~numba_cuda_mlir.mlir_lowering_registry.MLIRLoweringRegistry`
and exposes the decorators ``lower``, ``lower_getattr``,
``lower_getattr_generic``, ``lower_setattr``, ``lower_setattr_generic``,
``lower_cast``, and ``lower_constant``. For convenience,
:py:func:`~numba_cuda_mlir.extending.lower_cast` is re-exported from
``numba_cuda_mlir.extending`` directly.

.. note::

   Internal lowering modules in Numba-CUDA-MLIR each create their own
   :py:class:`MLIRLoweringRegistry` instance and install it from
   :py:meth:`MLIRTargetContext.load_additional_registries`. Extension authors
   should instead use the shared registry exposed from
   :py:mod:`numba_cuda_mlir.extending`, which is wired up automatically. A
   privately-created registry will not be consulted during compilation unless
   you also patch ``load_additional_registries``, which is not part of the
   supported API.

.. _lower-decorator:

Registering a function lowering
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``lowering_registry.lower(callable, *argtys)`` registers an implementation
for a call to ``callable`` with arguments matching ``argtys``. The argument
types may be:

* Type *classes* (e.g. ``types.Integer``) to match any instance.
* Type *instances* (e.g. ``types.int64``) to match exactly.
* ``types.Any`` to match anything.
* ``types.VarArg(...)`` to match a variadic tail.

.. code-block:: python

   from numba_cuda_mlir.extending import lowering_registry
   from numba_cuda_mlir.lowering_utilities import convert
   from numba_cuda_mlir._mlir import ir as mlir_ir
   from numba_cuda_mlir._mlir.dialects import llvm

   @lowering_registry.lower(make_boxed_int, types.Integer)
   def _lower_make_boxed_int(builder, target, args, kwargs):
       val = builder.load_var(args[0])
       struct_ty = builder.get_mlir_type(my_boxed_int)
       i64_ty = mlir_ir.IntegerType.get_signless(64)
       val = convert(val, i64_ty)
       undef = llvm.UndefOp(struct_ty)
       result = llvm.insertvalue(
           container=undef,
           value=val,
           position=mlir_ir.DenseI64ArrayAttr.get([0]),
       )
       builder.store_var(target, result)

Every lowering callback has the same four-argument shape:

.. code-block:: python

   def lower_impl(builder, target, args, kwargs):
       ...

* ``builder`` is the MLIR builder
  (:py:class:`~numba_cuda_mlir.mlir_lowering.MLIRLower`). See
  :ref:`lowering-builder` for the methods it exposes.
* ``target`` is the variable that should receive the result. Write
  to it with ``builder.store_var(target, value)``. Side-effecting lowerings
  that produce no return value may leave the target unwritten.
* ``args`` is a list of variables holding the call arguments. Read
  them with ``builder.load_var(var)`` or ``builder.load_vars(vars)``.
* ``kwargs`` is a list of ``(name, var)`` tuples for keyword arguments.

A lowering may register against multiple signatures by stacking decorators:

.. code-block:: python

   @lower(operator.not_, types.Number)
   @lower(operator.not_, types.Boolean)
   def lower_not(builder, target, args, kwargs):
       ...

.. _lowering-builder:

The MLIR builder
~~~~~~~~~~~~~~~~

:py:class:`~numba_cuda_mlir.mlir_lowering.MLIRLower` is the lowering builder
for Numba-CUDA-MLIR. It exposes helpers for generating MLIR inside the lowering
implementations. The methods most commonly used inside lowerings are:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Purpose
   * - ``builder.load_var(var)``
     - Load the MLIR value currently bound to a Numba IR variable.
   * - ``builder.load_vars(vars)``
     - Load several variables in one call (convenience for binary/n-ary
       lowerings).
   * - ``builder.store_var(var, value)``
     - Bind a Numba IR variable to an MLIR value. Used to record the result
       of a lowering into the ``target``.
   * - ``builder.get_numba_type(var_or_name)``
     - Look up the Numba type of an IR variable. Use this to branch on
       literal types or to dispatch on dtype.
   * - ``builder.get_mlir_type(numba_type)``
     - Translate a Numba type into the MLIR type that represents its data
       model in lowered code.
   * - ``builder.mlir_convert(value, target_mlir_type)``
     - Emit MLIR for an explicit type conversion between MLIR types. Prefer
       this over hand-emitting ``arith`` casts when going between numeric
       types.
   * - ``builder.lower_overload_call(target, disp, args)``
     - Invoke an overload's compiled implementation as part of a higher-level
       lowering. Used internally by
       :py:func:`~numba_cuda_mlir.extending.overload_attribute` and
       :py:func:`~numba_cuda_mlir.extending.overload_method`.
   * - ``builder.lower_literal_if_needed(value, numba_type=None)``
     - Materialise a Python literal or NumPy scalar as an MLIR value.
   * - ``builder.alloca(ty, count=1)``
     - Emit a stack allocation in the function's entry block.
   * - ``builder.alloca_insertion_point()``
     - Context manager that places the builder's insertion point at the
       function entry, suitable for hoisting allocas.
   * - ``builder.incref(typ, value)`` / ``builder.decref(typ, value)``
     - Manipulate the reference count of an NRT-managed value.
   * - ``builder.mlir_gpu_module``
     - The enclosing ``gpu.module`` operation; needed when declaring external
       functions or device intrinsics.


Helpers in ``lowering_utilities``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many lowerings reuse helpers from
:py:mod:`numba_cuda_mlir.lowering_utilities`. These are higher-level than
calling individual dialect ops and should be preferred where applicable:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Helper
     - Purpose
   * - ``convert(value, target_type)``
     - Emit an MLIR conversion to ``target_type``, picking the right ``arith``
       op or LLVM cast for the source/destination pair.
   * - ``constant(py_value, target_type)``
     - Build an MLIR constant of ``target_type`` from a Python value.
   * - ``int_of(value)``, ``index_of(value)``
     - Shortcuts for ``i64`` and ``index`` constants.
   * - ``i32_of(v)``, ``i64_of(v)``, ``f32_of(v)``, ``f64_of(v)``
     - Typed constants for the common scalar types.
   * - ``broadcast_shapes_for_binary_op(lhs, rhs, builder)``
     - Broadcast two ranked tensors / memrefs to a common shape before
       elementwise lowering.
   * - ``memref_to_tensor(v)``, ``tensor_to_memref(v)``
     - Convert between MLIR ``memref`` and ``tensor`` forms.
   * - ``get_or_insert_function(name, fn_type, gpu_module)``
     - Declare (or look up) an external ``func.func`` symbol in the current
       GPU module — needed when calling libdevice or any other externally
       defined function from a lowering.

.. _lowering-getattr-setattr:

Lowering attribute access
~~~~~~~~~~~~~~~~~~~~~~~~~

Attribute access has its own lowering decorators because the call shape is
different (no ``args``/``kwargs``):

.. code-block:: python

   @lowering_registry.lower_getattr(MyType, "field")
   def lower_field(context, builder, typ, val):
       ...

* ``context`` — the target context.
* ``builder`` — the MLIR builder, as for ``@lower``.
* ``typ`` — the Numba type of the receiver.
* ``val`` — the Numba IR variable (or MLIR value, depending on caller)
  holding the ``self`` object.

For a fallback lowering that handles *any* attribute on a type, use
``lower_getattr_generic`` whose callback receives an extra ``attr``
parameter:

.. code-block:: python

   @lowering_registry.lower_getattr_generic(MyType)
   def lower_any_attr(context, builder, typ, val, attr):
       ...

Setattr is registered with ``lower_setattr`` / ``lower_setattr_generic``;
the callback signature is ``(context, builder, sig, args)`` (or with an
extra ``attr`` for the generic form), where ``args`` is ``[target, value]``.

.. _lowering-cast:

Lowering implicit conversions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When type inference inserts an implicit cast — for example to unify the
branches of an ``if`` expression, or to coerce a value being stored into an
array — the lowering layer consults the cast registry before falling back to
the default ``mlir_convert`` path. Register a custom implicit conversion with
:py:func:`~numba_cuda_mlir.extending.lower_cast`:

.. code-block:: python

   from numba_cuda_mlir.extending import lower_cast
   from numba_cuda_mlir._mlir import ir as mlir_ir
   from numba_cuda_mlir._mlir.dialects import llvm

   @lower_cast(MyBoxedIntType, types.Integer)
   def _cast_boxed_to_int(context, builder, fromty, toty, val):
       result_ty = builder.get_mlir_type(toty)
       return llvm.extractvalue(
           res=result_ty,
           container=val,
           position=mlir_ir.DenseI64ArrayAttr.get([0]),
       )

A ``@lower_cast`` callback takes:

* ``context`` — the target context.
* ``builder`` — the MLIR builder.
* ``fromty`` — the source Numba type.
* ``toty`` — the destination Numba type.
* ``val`` — the MLIR value to be converted.

It must *return* the converted MLIR value (it does not write to a target).

For the conversion to be considered by type inference in the first place,
make sure the source type's ``can_convert_to`` returns a non-``None``
:py:class:`Conversion` for the target type (see
:ref:`Defining new Numba types <low-level-extending>`).

.. _lower-cast-example:

Worked example: a custom boxed integer
--------------------------------------

This example combines all of the pieces above into a single complete extension.
It defines a new Numba type ``MyBoxedInt`` (a one-field struct wrapping an
``int64``), gives it a data model, registers a constructor with both typing and
lowering, and registers an implicit conversion back to ``int64`` so that the
type unifies cleanly with regular integers at branch joins. The source —
including its integration test — is in ``tests/test_extending_lower_cast.py``.

.. code-block:: python

   import numpy as np
   from numba_cuda_mlir import cuda, extending, types
   from numba_cuda_mlir.extending import lower_cast, lowering_registry
   from numba_cuda_mlir.lowering_utilities import convert
   from numba_cuda_mlir.models import PrimitiveModel, register_model
   from numba_cuda_mlir._mlir import ir as mlir_ir
   from numba_cuda_mlir._mlir.dialects import llvm
   from numba_cuda_mlir.numba_cuda.typeconv import Conversion

   # 1. A new Numba type, with an opt-in implicit conversion to Integer.
   class MyBoxedIntType(types.Type):
       def __init__(self):
           super().__init__(name="MyBoxedInt")

       @property
       def key(self):
           return self.__class__

       def can_convert_to(self, typingctx, other):
           if isinstance(other, types.Integer):
               return Conversion.safe
           return None

   my_boxed_int = MyBoxedIntType()

   # 2. Data model: represent the type as an LLVM struct containing one i64.
   @register_model(MyBoxedIntType)
   class MyBoxedIntModel(PrimitiveModel):
       def __init__(self, dmm, fe_type):
           be_type = llvm.StructType.get_literal(
               [mlir_ir.IntegerType.get_signless(64)]
           )
           super().__init__(dmm, fe_type, be_type)

   # 3. A constructor callable, with typing and lowering.
   def make_boxed_int(x):
       raise NotImplementedError("only callable inside a numba_cuda_mlir kernel")

   @extending.type_callable(make_boxed_int)
   def _type_make_boxed_int(context):
       def typer(x):
           if isinstance(x, types.Integer):
               return my_boxed_int
       return typer

   @lowering_registry.lower(make_boxed_int, types.Integer)
   def _lower_make_boxed_int(builder, target, args, kwargs):
       val = builder.load_var(args[0])
       struct_ty = builder.get_mlir_type(my_boxed_int)
       i64_ty = mlir_ir.IntegerType.get_signless(64)
       val = convert(val, i64_ty)
       undef = llvm.UndefOp(struct_ty)
       result = llvm.insertvalue(
           container=undef, value=val,
           position=mlir_ir.DenseI64ArrayAttr.get([0]),
       )
       builder.store_var(target, result)

   # 4. Implicit conversion MyBoxedInt -> int64.
   @lower_cast(MyBoxedIntType, types.Integer)
   def _cast_boxed_to_int(context, builder, fromty, toty, val):
       result_ty = builder.get_mlir_type(toty)
       return llvm.extractvalue(
           res=result_ty, container=val,
           position=mlir_ir.DenseI64ArrayAttr.get([0]),
       )

   # 5. A kernel where branch unification forces the cast.
   @cuda.jit
   def kernel(flag, out):
       if flag[0]:
           x = make_boxed_int(42)
       else:
           x = np.int64(99)
       out[0] = x

In the kernel above, the two branches of the ``if`` produce values of
different Numba types (``MyBoxedInt`` and ``int64``). Type inference
consults ``MyBoxedIntType.can_convert_to`` and decides to unify on
``int64``, inserting an implicit cast on the ``MyBoxedInt`` branch. The cast
is then lowered by ``_cast_boxed_to_int`` into a single ``llvm.extractvalue``
that pulls the underlying integer out of the struct.

This same pattern — type, data model, constructor (typing + lowering),
optional conversions — generalises to any custom type you may want to add.

Putting it together with the high-level API
-------------------------------------------

The high-level decorators in :ref:`high-level-extending` go through exactly
the same typing and lowering registries described above. They are
re-implementations of Numba's ``@overload`` family that arrange for the
implementation function to be compiled by Numba-CUDA-MLIR's MLIR pipeline,
and they register typing templates and ``lower_getattr`` callbacks on the
shared registries.

This means you can freely mix the two tiers. A common pattern is:

* Use ``@overload`` or ``@overload_method`` for the bulk of the
  implementation, where pure Python on top of supported NumPy and array
  primitives is enough.
* Drop down to ``@intrinsic`` or ``lowering_registry.lower`` for the
  operations that need direct MLIR emission — for example calling a
  libdevice routine, emitting inline PTX, or wrapping an aggregate type
  field access.
