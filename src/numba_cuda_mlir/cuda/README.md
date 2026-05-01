# NUMBA_CUDA_MLIR Declarations

These are the modules that users will be using.
They often don't do anything (look at the `Stub` class in numba-cuda)
but they give the users something to call, the type declarations something to type,
and the lowering something to lower.

We could define numba intrinsics here and let the users call them directly,
but intrinsics take a typing context as their first argument but the user
never sees it, and users calling intrinsics directly does not allow for flexible
overloading.
