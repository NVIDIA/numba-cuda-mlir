..
   SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
   SPDX-License-Identifier: BSD-2-Clause

.. _extension-api:

Extension API
=============

Numba-CUDA-MLIR can be extended to support new types, functions, and methods so
that they are usable from JIT-compiled device code. The Extension API has two
tiers:

* The **High-level API** enables support for new Python callables, methods, and
  attributes by writing pure Python `implementation functions` that are
  themselves JIT-compiled. This API should be preferred wherever possible.
* The **Low-level API** exposes type inference and code generation machinery
  directly. It should be used for extensions that cannot be expressed with the
  high-level API - for example, when implementing a new type, when MLIR or PTX
  needs to be emitted directly, or when implementing implicit conversions
  between types.

The High-level API is closely modelled on `Numba's High-level Extension API
<https://numba.readthedocs.io/en/stable/extending/high-level.html>`_. Its
decorators, such as :py:func:`~numba_cuda_mlir.extending.overload` and
:py:func:`~numba_cuda_mlir.extending.overload_method`, behave in a similar way
to their Numba counterparts.

The Low-level Typing API is also similar to Numba's, but the Lowering API
differs: instead of emitting LLVM IR through ``llvmlite``, lowering functions
emit MLIR through the bindings accessed via :py:mod:`numba_cuda_mlir._mlir`.

An understanding of how type inference works in Numba-CUDA-MLIR is crucial to
effective use of the extension APIs. For a reference description of type
inference, see Numba's `NBEP 5: Type Inference
<https://numba.readthedocs.io/en/stable/proposals/type-inference.html>`_
document.

The pages below cover each part of the API in turn. A separate dialect primer
and a tabular operation listing are provided so that authors of low-level
extensions can find what they need quickly.

.. toctree::
   :maxdepth: 2

   high-level.rst
   low-level.rst
   mlir-dialects.rst
   mlir-ops.rst
