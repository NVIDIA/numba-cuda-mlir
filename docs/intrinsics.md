# Typing and Lowering

To use something in a numba-cuda-mlir kernel, we need to tell the type system how to type
it and how to lower it to MLIR.
Numba provides high-level and low-level interfaces for extending the type system and lowering,
and we _strongly_ prefer the low-level interface because:

- The high-level interface requires the compiler to type and lower an entirely new function.
    This is very slow compared to the alternatives.
- The high-level interface lends itself to implicit typing, since we just write Python
    code or intrinsics and the type system infers the types for us.
- The low-level interface allows us to explicitly type and lower something.

The high-level interface involves writing an _overload_ like so:

```python
# Lower
def math_ceil_cg(mlir_lower, target, args, kwargs): ...

@intrinsic
def math_ceil_intrinsic(typingctx, x):
    return x(x), math_ceil_cg

# Let Numba implicitly type math.ceil
@overload(math.ceil)
def math_ceil_ol(x):
    def ol(typingctx, x):
        return math_ceil_intrinsic
    return ol
```

The time spent typing and lowering `math_ceil_ol` is very expensive, we already know how we want
the function to be typed, we want the function to be unconditionally inlined anyways,
and we need to provide a lowering anyways. Instead of the above, we can just write:

```python
# Type math.ceil in numba_cuda_mlir/typing/math.py
class MathCeilTemplate(AbstractTemplate):
    key = math.ceil
    def generic(self, args, kws):
        assert len(args) == 1 and len(kws) == 0
        return signature(args[0], args[0])

# Lower math.ceil in numba_cuda_mlir/lowering/math.py
@lower(math.ceil, types.Number)
def math_ceil_cg(mlir_lower, target, args, kwargs): ...
```

This way, the type system can simply retrieve the type from the template and
lower the function with far fewer steps.
