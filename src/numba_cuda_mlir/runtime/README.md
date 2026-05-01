# Runtime

Quick-and-dirty way to add an intrinsic into the `numba_cuda_mlir.cuda.intrin` module.
Just drop an MLIR file into this directory, and its contents will be imported into
an ExternMLIRLibrary object, and its functions will be converted to ExternMLIRLibraryFunction
objects and exposed under the `numba_cuda_mlir.cuda.intrin` module.

Make sure to add the `attributes {always_inline}` and mark them `private` so they
don't needlessly clutter the user's program.
