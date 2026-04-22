# CUSIMT Type Declarations

This is where we take all the functions defined in `cusimt.cuda`
and tell Numba how to type them.
If you are comparing this with numba, these are equivalent to the `*decl` modules,
like `cudadecl.py`.

TODO: this module should only be for _declarations_ of the types 
of functions and modules in cusimt.cuda, but not definitions of new types.
These should be moved into cusimt.decls.
