# Inline Assembly

The CUDA C++ programming guide uses this example for inline assembly:
```c
  int x, y;
  asm(".reg .u32 t1;\n\t"              // temp reg t1
      " mul.lo.u32 t1, %1, %1;\n\t"    // t1 = x * x
      " mul.lo.u32 %0, t1, %1;"        // y = t1 * x
      : "=r"(y) : "r" (x));
```

To keep the interface as close to CUDA C++ as possible, we use the following syntax:
```python
@jit()
def kernel():
    x = np.int32(1)
    res = ptx.inline_ptx(
        """.reg .u32 t1;
            mul.lo.u32 t1, %1, %1;
            mul.lo.u32 %0, t1, %1;
            """, "=r", np.int32, "r", x)
    printf("res: %d\n", res)
```

Notable differences:
- Instead of `"string-constraint"(asm-parameter)`, we first pass the string constraint
    and then the asm parameter.
- Instead of passing return pointers, we pass the _type_ of the return _values_
    which are then returned (as a scalar for one return value, as a tuple for multiple return values).

Note also that the MLIR interface for inline assembly is quite different from the CUDA C++ interface.
We attempt to match the CUDA C++ interface, but we may expose unstable MLIR concepts
so ninja users have the flexibility they need.

[The MLIR interface for inline assembly](https://mlir.llvm.org/docs/Dialects/NVVMDialect/#nvvminline_ptx-nvvminlineptxop)
might be quite foreign to most users:
```
operation ::= `nvvm.inline_ptx` $ptxCode
              ( `ro` `(` $readOnlyArgs^ `:` type($readOnlyArgs) `)` )?
              ( `rw` `(` $readWriteArgs^ `:` type($readWriteArgs) `)` )?
              (`,` `predicate` `=` $predicate^)?
              attr-dict
              ( `->` type($writeOnlyArgs)^ )?
```

Instead of formatting arguments with `%0`, `%1`, etc, they would need to map
the read-only, read-write, and write-only arguments to `$r0`, `$rw0`, `$w0`, etc.
To provide the CUDA C++-like interface, we map the arguments to the C++ interface
to their MLIR equivalents, and then we replace the `%0`, `%1`, etc with `$rw2`, etc,
based on the access and type constraint strings.
