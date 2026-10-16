# Packaging the MLIR build

`ci/build-llvm-modern.sh` builds LLVM+MLIR from unmodified upstream sources and
has always produced an install tree that only this repository consumed. These
two distributions publish that tree instead, so that something built elsewhere
can link the same MLIR.

| Distribution | Contents | Varies with |
|---|---|---|
| `cuda-mlir-core` | the Python bindings and `libMLIRPythonCAPI.so`, importable as `cudamlir` | platform, Python version |
| `cuda-mlir-core-dev` | headers, CMake config, `mlir-tblgen`, link archives | platform |

They are split because only the first contains Python extension modules. The
second is what a dialect build needs, and the only reason a dialect build would
otherwise have to compile LLVM itself.

Neither is named `mlir`. This is one particular build of MLIR — a chosen
commit, with RTTI and assertion settings that consumers have to match — and not
general MLIR, so it should not be mistakable for an official upstream package.
The names say nothing about which targets are enabled, because that is a
property of the build that may change and not of the thing being depended on.

## Why this is published rather than kept private

MLIR identifies a type by the address of a per-type static, so two copies of
MLIR in one process are two disjoint type universes: a dialect registered
through one is simply absent from the other, `isa<>` across the boundary
returns false, and nothing crashes. A dialect built against its own copy of
MLIR cannot register into a context created by these bindings. The only way for
an independently built dialect to work is for both sides to resolve to the same
MLIR, which means that MLIR has to be a thing you can depend on.

## Versioning

The version carries the LLVM commit in its local segment
(`23.0.0.dev0+<short>`) because what consumers have to agree on is a particular
build, not a release number: MLIR's C++ API has no cross-version ABI contract,
and two builds of the same version still will not interoperate if they differ
on RTTI or assertions. Both distributions are pinned exactly by consumers, so a
mismatch surfaces when dependencies are resolved rather than as silence at
runtime.

## Building

```bash
export LLVM_MODERN_INSTALL=/path/to/llvm-modern-install
bash ci/build-llvm-modern.sh
python -m build --wheel packaging/cuda-mlir-core
python -m build --wheel packaging/cuda-mlir-core-dev
```
