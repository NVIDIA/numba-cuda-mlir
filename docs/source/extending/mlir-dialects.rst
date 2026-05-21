..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. _mlir-dialects-primer:

MLIR dialect primer
===================

Numba-CUDA-MLIR emits MLIR through the Python bindings accessible in
:py:mod:`numba_cuda_mlir._mlir`. This page is a short tour of the dialects
that lowering code reaches for most often, with pointers to the upstream
`MLIR Python bindings <https://mlir.llvm.org/docs/Bindings/Python/>`_ and
`Dialects <https://mlir.llvm.org/docs/Dialects/>`_ documentation for full
details.

You do not need to be deeply familiar with MLIR to use the lowering API —
the dialect operations behave a lot like calls to typed constructors that
return ``ir.Value`` objects. Each call inserts an operation into the
*current insertion point* of the MLIR builder and returns its result(s).
Insertion points are managed for you when lowering inside a kernel body; if
you need to manipulate them directly, use the
:py:class:`numba_cuda_mlir._mlir.ir.InsertionPoint` context manager.

The :py:mod:`numba_cuda_mlir._mlir` package
-------------------------------------------

The most useful members of :py:mod:`numba_cuda_mlir._mlir` for extension
authors are:

* :py:mod:`numba_cuda_mlir._mlir.ir` — core IR classes: ``Module``,
  ``Block``, ``Operation``, ``OpView``, ``InsertionPoint``, ``Value``,
  ``Type``, ``Attribute``, and the concrete type classes used by individual
  dialects (``IntegerType``, ``FloatType``, ``MemRefType``,
  ``RankedTensorType``, ``DenseI64ArrayAttr``, …).
* :py:mod:`numba_cuda_mlir._mlir.dialects` — one submodule per MLIR dialect
  (``arith``, ``memref``, ``llvm``, ``math``, ``nvvm``, ``gpu``, ``func``,
  ``scf``, ``vector``, ``linalg``, ``tensor``, …). Each exposes the dialect's
  operations as Python callables.
* :py:mod:`numba_cuda_mlir._mlir.extras.types` — usually imported as ``T``;
  short factory functions for the common MLIR types: ``T.i32()``, ``T.f64()``,
  ``T.bool()``, ``T.index()``, ``T.memref(...)``, ``T.tensor(...)``,
  ``T.vector(...)``.

Numba-CUDA-MLIR also ships a thin layer of *dialect extensions* at
:py:mod:`numba_cuda_mlir.mlir.dialect_exts`. These wrap a handful of dialects
(``scf``, ``llvm``, ``arith``, ``memref``, ``func``, ``math``, ``tensor``)
with Python helpers — for example the
:py:func:`if_ctx_manager`/:py:func:`else_ctx_manager` context managers for
``scf.if`` — that are more ergonomic from Python than the raw bindings.
Where a dialect_ext helper is available it is generally the preferred entry
point.

The dialects below are the ones used most frequently by the lowering code in
:py:mod:`numba_cuda_mlir.lowering`. They are listed roughly in order of how
often they appear.

arith — arithmetic and comparison
---------------------------------

The ``arith`` dialect carries the core arithmetic, comparison, and numeric
conversion operations. It is the most-used dialect in the lowering code.
Common operations:

* Constants — ``arith.constant(result=T.i64(), value=0)``.
* Binary arithmetic — ``arith.addi``, ``arith.subi``, ``arith.muli``,
  ``arith.addf``, ``arith.subf``, ``arith.mulf``, ``arith.divf``, etc. The
  ``*i`` variants operate on integer values, ``*f`` variants on floats.
* Comparison — ``arith.cmpi(predicate, lhs, rhs)`` and ``arith.cmpf`` return
  ``i1``; predicates are imported from the dialect.
* Selection — ``arith.select(cond, then_value, else_value)`` is MLIR's
  equivalent of LLVM's ``select``.
* Type conversions — ``arith.extsi``, ``arith.extui``, ``arith.trunci``,
  ``arith.extf``, ``arith.truncf``, ``arith.fptosi``, ``arith.fptoui``,
  ``arith.sitofp``, ``arith.uitofp``, ``arith.bitcast``.
* Bitwise — ``arith.andi``, ``arith.ori``, ``arith.xori``, ``arith.shli``,
  ``arith.shrsi``, ``arith.shrui``.

In most cases you should prefer
:py:func:`numba_cuda_mlir.lowering_utilities.convert` over hand-emitting an
``arith`` cast; it picks the right op for the source/destination pair.

Upstream reference: `arith dialect <https://mlir.llvm.org/docs/Dialects/ArithOps/>`_.

memref — strided in-memory arrays
---------------------------------

The ``memref`` dialect models ranked, strided in-memory arrays. Python arrays
are represented as memrefs at the MLIR level once they have been lowered.
Common operations:

* ``memref.alloc(memref_type)`` / ``memref.alloca(memref_type)`` — heap and
  stack allocation.
* ``memref.load(memref, indices)`` / ``memref.store(value, memref, indices)``
  — element access.
* ``memref.dim(memref, index)`` — runtime dimension extent.
* ``memref.subview(memref, offsets, sizes, strides)`` — strided sub-arrays.
* ``memref.cast`` — change the static type of a memref (e.g. add/drop
  dynamic dimensions).
* ``memref.reshape`` — runtime reshape.
* ``memref.extract_aligned_pointer_as_index(memref)`` — get the base data
  pointer as an ``index``; used when bridging to ``llvm.getelementptr``.

Upstream reference: `memref dialect <https://mlir.llvm.org/docs/Dialects/MemRef/>`_.

llvm — low-level pointer and struct ops
---------------------------------------

The ``llvm`` dialect mirrors the LLVM IR instruction set inside MLIR. It is
used in Numba-CUDA-MLIR for pointer arithmetic and aggregate manipulation
that does not fit into the higher-level dialects above. Common operations:

* ``llvm.load(res, ptr)`` / ``llvm.store(value, ptr)`` — raw load/store at
  an ``!llvm.ptr``.
* ``llvm.getelementptr(res, base, indices, gep_indices, element_type, ...)``
  — compute a pointer to a field or element. The pattern in
  :py:mod:`~numba_cuda_mlir.lowering_utilities` uses the magic value
  ``GEP_DYNAMIC_INDEX = -2147483648`` to indicate a runtime-dynamic index.
* ``llvm.extractvalue(res, container, position)`` /
  ``llvm.insertvalue(container, value, position)`` — field access on
  aggregates (LLVM structs). ``position`` is a
  ``DenseI64ArrayAttr`` whose entries are the field indices.
* ``llvm.UndefOp(struct_type)`` — produce an undefined value of an
  aggregate type, typically the starting point for building a struct field
  by field with ``llvm.insertvalue``.
* ``llvm.inttoptr`` / ``llvm.ptrtoint`` — convert between integer and
  pointer types.
* ``llvm.atomicrmw`` — atomic read-modify-write operations.

The MLIR pointer type is constructed with ``llvm.PointerType.get()``. MLIR
uses opaque pointers, so the pointee type is supplied at the GEP/load/store
site as ``element_type``.

Upstream reference: `LLVM dialect <https://mlir.llvm.org/docs/Dialects/LLVM/>`_.

math — transcendental and library functions
-------------------------------------------

The ``math`` dialect provides floating-point library functions that map
directly to PTX or libdevice intrinsics on the device. Examples used by the
lowering code include ``math.log``, ``math.exp``, ``math.sin``, ``math.cos``,
``math.tan``, ``math.sqrt``, ``math.atan``, ``math.asin``, ``math.acos``,
``math.tanh``, ``math.sinh``, ``math.cosh``, ``math.pow``, ``math.fma``,
``math.gamma``, ``math.lgamma``, ``math.isnan``, ``math.isinf``,
``math.isfinite``, ``math.modf``, ``math.fmod``, ``math.remainder``.

Complex-valued variants are in the ``complex`` dialect (imported as
``complex_dialect`` to avoid clashing with the standard library ``complex``).

Upstream reference: `math dialect <https://mlir.llvm.org/docs/Dialects/MathOps/>`_.

nvvm — NVIDIA GPU intrinsics
----------------------------

The ``nvvm`` dialect exposes NVPTX intrinsics. Used in Numba-CUDA-MLIR for
the operations that have no portable analogue. Common operations:

* ``nvvm.inline_ptx(asm)`` — emit an inline PTX instruction. Use this for
  device-specific operations that have no higher-level wrapper.
* Warp-level synchronisation — ``nvvm.vote_sync``, ``nvvm.match_sync``,
  ``nvvm.shfl_sync``, ``nvvm.bar_warp_sync``.
* Barriers — ``nvvm.barrier``.
* Special registers — ``nvvm.read_ptx_sreg_warpsize``,
  ``nvvm.read_ptx_sreg_laneid``.
* ``nvvm.nanosleep`` — thread sleep.

The intrinsic wrappers in
:py:mod:`numba_cuda_mlir.lowering.mlir.nvvm` (e.g. ``breakpoint``,
``nanosleep``) provide higher-level Python entry points over ``nvvm`` ops
and are usually the preferred call site from extension code.

Upstream reference: `NVVM dialect <https://mlir.llvm.org/docs/Dialects/NVVMDialect/>`_.

gpu — GPU module and kernel intrinsics
--------------------------------------

The ``gpu`` dialect carries the structural pieces of a GPU program — kernel
declarations, thread/block index queries, and a few I/O helpers:

* Thread index — ``gpu.thread_id(gpu.Dimension.x)``, ``gpu.block_id``,
  ``gpu.block_dim``, ``gpu.grid_dim``.
* Synchronisation — ``gpu.barrier``.
* Output — ``gpu.printf(format_string, *args)``.
* ``gpu.address_space`` — memory-space attribute used when constructing
  memrefs that live in shared or constant memory.

Upstream reference: `GPU dialect <https://mlir.llvm.org/docs/Dialects/GPU/>`_.

func — function definitions and calls
-------------------------------------

The ``func`` dialect provides ``func.func`` (function definition) and
``func.call`` (function call). Lowerings that need to call an externally
declared function — for example a libdevice routine — declare it with the
``get_or_insert_function`` helper from
:py:mod:`numba_cuda_mlir.lowering_utilities` and call it with
``func.call(result=[ret_ty], callee=name, operands_=[arg, ...])``.

Upstream reference: `func dialect <https://mlir.llvm.org/docs/Dialects/Func/>`_.

scf — structured control flow
-----------------------------

The ``scf`` dialect provides structured loops and conditionals: ``scf.for``,
``scf.if``, ``scf.while``, ``scf.forall``, ``scf.index_switch``.

The bindings in :py:mod:`numba_cuda_mlir._mlir.dialects.scf` mirror the raw
MLIR operations, but the extension layer in
:py:mod:`numba_cuda_mlir.mlir.dialect_exts.scf` exposes Python-friendly
context managers that are usually easier to use from a lowering:

.. code-block:: python

   from numba_cuda_mlir.mlir.dialect_exts import scf
   from numba_cuda_mlir.mlir.dialect_exts.scf import (
       if_ctx_manager as if_,
       else_ctx_manager as else_,
   )

   with if_(predicate, results=[]) as if_op:
       # ... body ...
       scf.yield_([])
   with else_(if_op):
       # ... body ...
       scf.yield_([])

Use ``scf.yield_(values)`` to terminate a region; pass an empty list when
the region produces no values.

Upstream reference: `scf dialect <https://mlir.llvm.org/docs/Dialects/SCFDialect/>`_.

vector — SIMD vector operations
-------------------------------

The ``vector`` dialect provides SIMD vector load/store, lane extraction, and
elementwise operations. It is used in the CUDA vector-type lowerings
(``int2``, ``float4``, …):

* ``vector.load`` / ``vector.store`` — vectorised memory access.
* ``vector.transfer_read`` / ``vector.transfer_write`` — scalar/vector
  movement with optional masking.
* ``vector.from_elements`` / ``vector.extract`` — build a vector from
  scalars and pull a lane back out.

Upstream reference: `vector dialect <https://mlir.llvm.org/docs/Dialects/Vector/>`_.

tensor — value-semantic tensors
-------------------------------

The ``tensor`` dialect is the value-semantic counterpart to ``memref``. It
is used in Numba-CUDA-MLIR primarily as the result type of higher-level
elementwise operations before they are bufferised back into memrefs.
Operations of interest: ``tensor.empty``, ``tensor.splat``,
``tensor.extract``, ``tensor.dim``, ``tensor.generate``,
``tensor.collapse_shape``.

Upstream reference: `tensor dialect <https://mlir.llvm.org/docs/Dialects/TensorOps/>`_.

linalg — high-level array kernels
---------------------------------

The ``linalg`` dialect provides structured operations like ``linalg.map``
and ``linalg.reduce`` that take regions describing per-element behaviour.
It is the dialect of choice for elementwise and reduction lowerings that
should benefit from upstream MLIR optimisation passes:

.. code-block:: python

   from numba_cuda_mlir._mlir.dialects import linalg

   @linalg.map(result=[dtype], inputs=[lhs, rhs], init=empty)
   def binop(lhs, rhs, init):
       # ... emit the per-element body ...
       return ...

Upstream reference: `linalg dialect <https://mlir.llvm.org/docs/Dialects/Linalg/>`_.

Less-frequently used dialects
-----------------------------

The following dialects appear in only one or two lowering files but are
worth knowing exist for esoteric use cases:

* ``cf`` — unstructured control flow (``cf.br``, ``cf.cond_br``,
  ``cf.assert``). Used by ``@intrinsic`` codegen that needs to drop in a
  device-side assertion (see the ``@intrinsic`` example in
  :ref:`high-level-intrinsic`).
* ``complex`` — operations on MLIR ``complex`` types. The Python import
  alias used in the lowering code is ``complex_dialect`` to avoid shadowing
  the built-in ``complex``.
* ``index`` — operations on the ``index`` type used for memref subscripts.
* ``bufferization`` — used by transforms that move between value-semantic
  tensors and side-effecting memrefs.
* ``shape`` — value-level shape computations
  (``shape.shape_of``, ``shape.get_extent``).
* ``nvgpu`` — higher-level NVIDIA GPU intrinsics complementing the ``nvvm``
  dialect.

All of these are accessed in the same way: ``from
numba_cuda_mlir._mlir.dialects import X`` and call operations as
``X.op_name(...)``.

Constructing MLIR types
-----------------------

The most convenient way to construct MLIR types in a lowering is through
:py:mod:`numba_cuda_mlir._mlir.extras.types`, traditionally imported as
``T``:

.. code-block:: python

   from numba_cuda_mlir._mlir.extras import types as T

   T.i32()                  # !i32
   T.i64()                  # !i64
   T.f32()                  # !f32
   T.f64()                  # !f64
   T.bool()                 # !i1
   T.index()                # !index
   T.memref(10, T.f32())    # !memref<10xf32>
   T.tensor(10, T.f32())    # !tensor<10xf32>
   T.vector(4, T.i32())     # !vector<4xi32>

For types that need to be constructed from a Numba type (so that they pick
up the right data model), prefer ``builder.get_mlir_type(numba_type)`` over
hand-constructing them with ``T``.

Working with attributes
-----------------------

Many MLIR operations take MLIR *attributes* alongside their operands.
Attributes are constants attached to an operation — most commonly array,
integer, and string constants for indices, sizes, and dialect-specific
configuration. The Python bindings expose these via classes on
:py:mod:`numba_cuda_mlir._mlir.ir`:

* ``DenseI64ArrayAttr.get([0, 1])`` — used for field-index arrays in
  ``llvm.insertvalue`` and ``llvm.extractvalue``.
* ``IntegerAttr.get(T.i64(), 5)`` — typed integer constant.
* ``StringAttr.get("foo")`` — string constant.

Upstream documentation for the attribute system lives at
`MLIR Built-in Attributes <https://mlir.llvm.org/docs/Dialects/Builtin/#attributes>`_.
