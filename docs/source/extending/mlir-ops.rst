..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. _mlir-ops-listing:

MLIR generation reference
=========================

This appendix lists the operations and helpers most often reached for from a
Numba-CUDA-MLIR lowering, grouped by dialect or by where they live. It is
intended for discoverability when writing a new lowering: when you know
*what* you want to emit but not *which* op or helper to call.

For semantic detail of each MLIR operation, follow the dialect link at the
top of each section to the upstream MLIR documentation.

The lowering builder
--------------------

Methods on the
:py:class:`~numba_cuda_mlir.mlir_lowering.MLIRLower` instance passed to
every lowering callback as ``builder``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Method
     - Purpose
   * - ``load_var(var)``
     - Load the MLIR value currently bound to a Numba IR variable.
   * - ``load_vars(vars)``
     - Load several variables in one call.
   * - ``store_var(var, value)``
     - Bind a Numba IR variable to an MLIR value (writes the result of the
       current lowering into ``target``).
   * - ``get_numba_type(var_or_name)``
     - Look up the Numba type of an IR variable.
   * - ``get_mlir_type(numba_type)``
     - Translate a Numba type into the MLIR type that represents its data
       model.
   * - ``mlir_convert(value, target_mlir_type)``
     - Emit an explicit MLIR type conversion between MLIR types.
   * - ``lower_overload_call(target, dispatcher, args)``
     - Invoke an overload's compiled implementation as part of a lowering.
   * - ``lower_literal_if_needed(value, numba_type=None)``
     - Materialise a Python literal or NumPy scalar as an MLIR value.
   * - ``alloca(ty, count=1)``
     - Emit a stack allocation in the function entry block.
   * - ``alloca_insertion_point()``
     - Context manager that places the insertion point at the function
       entry; suitable for hoisting allocas.
   * - ``incref(typ, value)`` / ``decref(typ, value)``
     - Adjust the reference count of an NRT-managed value.
   * - ``var_lowered(var)``
     - Check whether a variable has been lowered yet.
   * - ``mlir_gpu_module``
     - The enclosing ``gpu.module`` operation, needed when declaring
       external functions or device intrinsics.

Helpers in ``numba_cuda_mlir.lowering_utilities``
-------------------------------------------------

These higher-level helpers wrap common patterns and should usually be
preferred over emitting raw dialect ops:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Helper
     - Purpose
   * - ``convert(value, target_type)``
     - Emit an MLIR conversion to ``target_type``, picking the right
       ``arith`` op or LLVM cast for the source/destination pair.
   * - ``constant(py_value, target_type)``
     - Build an MLIR constant from a Python value.
   * - ``int_of(value)`` / ``index_of(value)``
     - Shortcuts for ``i64`` and ``index`` constants.
   * - ``i32_of`` / ``i64_of`` / ``f32_of`` / ``f64_of``
     - Typed constant constructors for the common scalar types.
   * - ``coerce_numpy_scalars_for_binary_op(lhs, rhs, builder)``
     - Promote two operands to a common NumPy-style type before lowering a
       binary operation.
   * - ``broadcast_shapes_for_binary_op(lhs, rhs, builder)``
     - Broadcast two ranked tensors / memrefs to a common shape.
   * - ``memref_to_tensor(v)`` / ``tensor_to_memref(v)``
     - Convert between MLIR ``memref`` and ``tensor`` forms.
   * - ``memref_to_llvm_ptr(array, indices, element_type)``
     - Compute an ``!llvm.ptr`` to an element of a memref.
   * - ``get_or_insert_function(name, fn_type, gpu_module)``
     - Declare (or look up) an external ``func.func`` symbol in the
       current GPU module.
   * - ``lookup_callee_in_module(name, gpu_module)``
     - Find an already-declared callee symbol.
   * - ``type_conversions.np_dtype_to_mlir_type``,
       ``type_conversions.to_mlir_type``
     - Map a NumPy dtype or arbitrary type to its MLIR representation.

The ``unverified_convert`` and ``equal`` helpers (also in
``lowering_utilities``) are useful for custom-type conversions and
comparisons respectively.

Type constructors — ``_mlir.extras.types``
------------------------------------------

Conventionally imported as ``T``:

.. list-table::
   :header-rows: 1

   * - Factory
     - Result
   * - ``T.i1()`` / ``T.bool()``
     - 1-bit integer
   * - ``T.i8()``, ``T.i16()``, ``T.i32()``, ``T.i64()``
     - Signless integers
   * - ``T.f16()``, ``T.f32()``, ``T.f64()``
     - Floating point
   * - ``T.bf16()``
     - bfloat16
   * - ``T.index()``
     - Platform-sized integer used for memref subscripts
   * - ``T.memref(*shape, element_type)``
     - Ranked ``memref`` type
   * - ``T.tensor(*shape, element_type)``
     - Ranked ``tensor`` type
   * - ``T.vector(*shape, element_type)``
     - Ranked ``vector`` type

For composite types that depend on a Numba data model (custom types,
records, NumPy arrays) prefer ``builder.get_mlir_type(numba_type)``.

arith dialect
-------------

`Upstream reference <https://mlir.llvm.org/docs/Dialects/ArithOps/>`__

.. list-table::
   :header-rows: 1

   * - Operation
     - Purpose
   * - ``arith.constant(result, value)``
     - Constant of any scalar/vector/tensor type.
   * - ``arith.addi``, ``arith.subi``, ``arith.muli``,
       ``arith.divsi``, ``arith.divui``,
       ``arith.remsi``, ``arith.remui``
     - Integer arithmetic.
   * - ``arith.addf``, ``arith.subf``, ``arith.mulf``,
       ``arith.divf``, ``arith.remf``
     - Floating-point arithmetic.
   * - ``arith.cmpi(predicate, lhs, rhs)``
     - Integer comparison; returns ``i1``.
   * - ``arith.cmpf(predicate, lhs, rhs)``
     - Floating-point comparison.
   * - ``arith.select(cond, then_value, else_value)``
     - Value selection.
   * - ``arith.andi``, ``arith.ori``, ``arith.xori``
     - Bitwise logical operations.
   * - ``arith.shli``, ``arith.shrsi``, ``arith.shrui``
     - Shifts.
   * - ``arith.extsi``, ``arith.extui``, ``arith.trunci``
     - Integer width conversions.
   * - ``arith.extf``, ``arith.truncf``
     - Floating-point width conversions.
   * - ``arith.fptosi``, ``arith.fptoui``,
       ``arith.sitofp``, ``arith.uitofp``
     - Integer/float conversions.
   * - ``arith.bitcast``
     - Reinterpret cast between types of the same bit width.
   * - ``arith.negf``
     - Floating-point negation.

memref dialect
--------------

`Upstream reference <https://mlir.llvm.org/docs/Dialects/MemRef/>`__

.. list-table::
   :header-rows: 1

   * - Operation
     - Purpose
   * - ``memref.alloc(memref_type)``
     - Heap allocation of a memref.
   * - ``memref.alloca(memref_type)``
     - Stack allocation of a memref.
   * - ``memref.load(memref, indices)``
     - Element read.
   * - ``memref.store(value, memref, indices)``
     - Element write.
   * - ``memref.dim(memref, index)``
     - Runtime dimension extent.
   * - ``memref.subview(memref, offsets, sizes, strides)``
     - Strided sub-array.
   * - ``memref.cast(memref, target_type)``
     - Static memref type cast (e.g. add/drop dynamic dims).
   * - ``memref.reshape(source, shape)``
     - Reshape via a shape memref.
   * - ``memref.collapse_shape(source, reassociation)``
     - Static rank reduction.
   * - ``memref.extract_aligned_pointer_as_index(memref)``
     - Base pointer as ``index``; used when bridging to ``llvm.getelementptr``.
   * - ``memref.extract_strided_metadata(memref)``
     - Pull out base pointer, offset, sizes, and strides as separate values.
   * - ``memref.copy(src, dst)``
     - Bulk memref copy.

llvm dialect
------------

`Upstream reference <https://mlir.llvm.org/docs/Dialects/LLVM/>`__

.. list-table::
   :header-rows: 1

   * - Operation
     - Purpose
   * - ``llvm.load(res, ptr)``
     - Load from ``!llvm.ptr``.
   * - ``llvm.store(value, ptr)``
     - Store to ``!llvm.ptr``.
   * - ``llvm.getelementptr(res, base, indices, gep_indices, element_type, ...)``
     - Pointer arithmetic.
   * - ``llvm.extractvalue(res, container, position)``
     - Read aggregate field.
   * - ``llvm.insertvalue(container, value, position)``
     - Write aggregate field.
   * - ``llvm.UndefOp(struct_type)``
     - Undefined value of an aggregate type.
   * - ``llvm.inttoptr(res, value)``
     - Integer-to-pointer conversion.
   * - ``llvm.ptrtoint(res, arg)``
     - Pointer-to-integer conversion.
   * - ``llvm.atomicrmw(...)``
     - Atomic read-modify-write.
   * - ``llvm.mlir_zero``
     - Zero value of an LLVM type.

Pointer types are constructed with ``llvm.PointerType.get()``. Aggregate
position attributes use ``mlir_ir.DenseI64ArrayAttr.get([...])``.

math dialect
------------

`Upstream reference <https://mlir.llvm.org/docs/Dialects/MathOps/>`__

.. list-table::
   :header-rows: 1

   * - Group
     - Operations
   * - Logarithm / exponent
     - ``math.log``, ``math.log2``, ``math.log10``, ``math.exp``,
       ``math.exp2``
   * - Power and roots
     - ``math.pow``, ``math.sqrt``, ``math.rsqrt``, ``math.cbrt``
   * - Trigonometry
     - ``math.sin``, ``math.cos``, ``math.tan``,
       ``math.asin``, ``math.acos``, ``math.atan``, ``math.atan2``
   * - Hyperbolic
     - ``math.sinh``, ``math.cosh``, ``math.tanh``
   * - Floating-point manipulation
     - ``math.ldexp``, ``math.frexp``, ``math.modf``,
       ``math.fmod``, ``math.remainder``, ``math.copysign``, ``math.fma``
   * - Classification
     - ``math.isnan``, ``math.isinf``, ``math.isfinite``
   * - Gamma
     - ``math.gamma``, ``math.lgamma``
   * - Complex
     - ``math.rect``, ``math.polar``, ``math.phase`` (note: also see the
       ``complex`` dialect for ops on MLIR ``complex`` values)

nvvm dialect
------------

`Upstream reference <https://mlir.llvm.org/docs/Dialects/NVVMDialect/>`__

.. list-table::
   :header-rows: 1

   * - Operation
     - Purpose
   * - ``nvvm.inline_ptx(asm)``
     - Emit raw inline PTX.
   * - ``nvvm.vote_sync``, ``nvvm.match_sync``,
       ``nvvm.shfl_sync``, ``nvvm.shfl``
     - Warp-level synchronisation primitives.
   * - ``nvvm.bar_warp_sync``
     - Warp barrier.
   * - ``nvvm.barrier``
     - Block-level barrier (alternative to ``gpu.barrier``).
   * - ``nvvm.read_ptx_sreg_warpsize``,
       ``nvvm.read_ptx_sreg_laneid``
     - Read PTX special registers.
   * - ``nvvm.nanosleep``
     - Thread sleep.
   * - ``nvvm.breakpoint``
     - Device-side breakpoint.

The intrinsic wrappers in :py:mod:`numba_cuda_mlir.lowering.mlir.nvvm`
provide higher-level Python entry points to a handful of these (e.g. a
``@intrinsic``-decorated ``breakpoint`` function).

gpu dialect
-----------

`Upstream reference <https://mlir.llvm.org/docs/Dialects/GPU/>`__

.. list-table::
   :header-rows: 1

   * - Operation
     - Purpose
   * - ``gpu.thread_id(gpu.Dimension.x|y|z)``
     - Thread index within block.
   * - ``gpu.block_id(...)``
     - Block index within grid.
   * - ``gpu.block_dim(...)``
     - Block dimensions.
   * - ``gpu.grid_dim(...)``
     - Grid dimensions.
   * - ``gpu.barrier()``
     - Block-level synchronisation.
   * - ``gpu.printf(format, *args)``
     - Formatted device-side output (uses the CUDA runtime ``printf``).
   * - ``gpu.address_space``
     - Memory-space enum attribute (``Private``, ``Workgroup``, ``Global``).

func dialect
------------

`Upstream reference <https://mlir.llvm.org/docs/Dialects/Func/>`__

.. list-table::
   :header-rows: 1

   * - Operation
     - Purpose
   * - ``func.func``
     - Function definition (op view).
   * - ``func.call(result=[ret_ty], callee=name, operands_=[...])``
     - Call a declared function by name.
   * - ``func.return_(values)``
     - Return from a function.

External callees are declared once per GPU module with
:py:func:`~numba_cuda_mlir.lowering_utilities.get_or_insert_function`.

scf dialect
-----------

`Upstream reference <https://mlir.llvm.org/docs/Dialects/SCFDialect/>`__

.. list-table::
   :header-rows: 1

   * - Operation
     - Purpose
   * - ``scf.for(lower, upper, step)``
     - Counted loop.
   * - ``scf.if(cond)``
     - Conditional.
   * - ``scf.forall(...)``
     - Parallel loop.
   * - ``scf.while(...)``
     - General while loop.
   * - ``scf.index_switch(value, cases)``
     - Multi-way branch on an ``index`` value.
   * - ``scf.yield_(values)``
     - Terminate a region.

Prefer the Python helpers in :py:mod:`numba_cuda_mlir.mlir.dialect_exts.scf`
— particularly ``if_ctx_manager`` and ``else_ctx_manager`` for ``scf.if`` —
over the raw bindings.

vector dialect
--------------

`Upstream reference <https://mlir.llvm.org/docs/Dialects/Vector/>`__

.. list-table::
   :header-rows: 1

   * - Operation
     - Purpose
   * - ``vector.load(vt, memref, indices)`` / ``vector.store(value, memref, indices)``
     - Vector-width load/store.
   * - ``vector.transfer_read(...)`` / ``vector.transfer_write(...)``
     - Strided / masked vector transfers.
   * - ``vector.from_elements(...)``
     - Build a vector from individual scalar lane values.
   * - ``vector.extract(vector, indices)``
     - Extract a lane (or sub-vector).

tensor dialect
--------------

`Upstream reference <https://mlir.llvm.org/docs/Dialects/TensorOps/>`__

.. list-table::
   :header-rows: 1

   * - Operation
     - Purpose
   * - ``tensor.empty(sizes, element_type)``
     - Allocate a value-semantic tensor with no defined contents.
   * - ``tensor.splat(value, element_type=...)``
     - Tensor with all elements set to a scalar value.
   * - ``tensor.extract(tensor, indices)``
     - Pull a scalar out of a tensor.
   * - ``tensor.dim(tensor, index)``
     - Runtime dimension extent.
   * - ``tensor.generate``
     - Build a tensor by evaluating a region per element.
   * - ``tensor.collapse_shape``
     - Static rank reduction.
   * - ``tensor.bitcast``
     - Reinterpret element type.

linalg dialect
--------------

`Upstream reference <https://mlir.llvm.org/docs/Dialects/Linalg/>`__

.. list-table::
   :header-rows: 1

   * - Operation
     - Purpose
   * - ``linalg.map(result, inputs, init)``
     - Elementwise map; pass a decorator-style block describing the
       per-element body.
   * - ``linalg.ReduceOp(result, inputs, inits, dimensions)``
     - Reduction along given dimensions; populate the ``combiner`` region
       with the reduction body.

Other dialects
--------------

Less commonly used dialects that nevertheless appear in
Numba-CUDA-MLIR lowerings:

.. list-table::
   :header-rows: 1

   * - Dialect
     - Selected ops
   * - ``cf`` (`upstream <https://mlir.llvm.org/docs/Dialects/ControlFlow/>`__)
     - ``cf.assert_``, ``cf.br``, ``cf.cond_br``
   * - ``complex`` (`upstream <https://mlir.llvm.org/docs/Dialects/ComplexOps/>`__)
     - ``complex.create``, ``complex.re``, ``complex.im``, complex arithmetic
   * - ``index`` (`upstream <https://mlir.llvm.org/docs/Dialects/IndexOps/>`__)
     - ``index.cmp``, ``index.add``, ``index.constant``
   * - ``shape`` (`upstream <https://mlir.llvm.org/docs/Dialects/ShapeDialect/>`__)
     - ``shape.shape_of``, ``shape.get_extent``
   * - ``bufferization`` (`upstream <https://mlir.llvm.org/docs/Bufferization/>`__)
     - Tensor → memref bufferisation ops
   * - ``nvgpu`` (`upstream <https://mlir.llvm.org/docs/Dialects/NVGPU/>`__)
     - Higher-level NVIDIA GPU intrinsics complementing ``nvvm``
   * - ``amdgpu``
     - AMD GPU intrinsics (rarely used in Numba-CUDA-MLIR)
   * - ``emitc``
     - Used by some external transforms
   * - ``builtin``
     - The MLIR built-in dialect — module / function / unrealized cast ops

Each of these is imported the same way: ``from
numba_cuda_mlir._mlir.dialects import <name>``.

Useful classes in ``_mlir.ir``
------------------------------

.. list-table::
   :header-rows: 1

   * - Class / function
     - Purpose
   * - ``Context``, ``Location``
     - MLIR context and source location; needed only when manipulating IR
       outside of a lowering callback.
   * - ``Module``
     - Top-level MLIR module.
   * - ``Block``, ``Region``, ``Operation``, ``OpView``
     - Core IR data structures. ``OpView`` is the Python wrapper around an
       ``Operation`` exposed by a dialect.
   * - ``InsertionPoint``
     - Context manager for controlling where new ops are inserted.
   * - ``Value``, ``Type``, ``Attribute``
     - Generic SSA value, type, and attribute classes.
   * - ``IntegerType``, ``FloatType``, ``IndexType``, ``MemRefType``,
       ``RankedTensorType``, ``VectorType``
     - Concrete type classes.
   * - ``DenseI64ArrayAttr``, ``DenseElementsAttr``, ``IntegerAttr``,
       ``StringAttr``, ``ArrayAttr``
     - Common attribute classes.
   * - ``get_parent_of_type(value, type_class)``,
       ``get_ops_of_type(parent, type_class)``
     - Helpers for walking the IR.
