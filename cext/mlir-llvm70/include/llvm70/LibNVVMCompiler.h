/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
//===- LibNVVMCompiler.h - libnvvm dlopen wrapper ---------------*- C++ -*-===//
//
// Wraps libnvvm.so (dlopen'd at runtime) to compile NVVM bitcode → PTX.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM70_LIBNVVMCOMPILER_H
#define LLVM70_LIBNVVMCOMPILER_H

#include "llvm70/CAPILoader.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <memory>
#include <string>

namespace llvm70 {

// Opaque handle matching nvvm.h
using nvvmProgram = struct _nvvmProgram *;

// nvvmResult values
enum nvvmResult {
  NVVM_SUCCESS = 0,
  NVVM_ERROR_OUT_OF_MEMORY = 1,
  NVVM_ERROR_PROGRAM_CREATION_FAILURE = 2,
  NVVM_ERROR_IR_VERSION_MISMATCH = 3,
  NVVM_ERROR_INVALID_INPUT = 4,
  NVVM_ERROR_INVALID_PROGRAM = 5,
  NVVM_ERROR_INVALID_IR = 6,
  NVVM_ERROR_INVALID_OPTION = 7,
  NVVM_ERROR_NO_MODULE_IN_PROGRAM = 8,
  NVVM_ERROR_COMPILATION = 9,
};

class LibNVVMCompiler {
public:
  static llvm::Expected<std::unique_ptr<LibNVVMCompiler>>
  create(llvm::StringRef libnvvmPath);

  ~LibNVVMCompiler();
  LibNVVMCompiler(const LibNVVMCompiler &) = delete;

  /// Compile bitcode buffer(s) to PTX (or LTOIR when \p genLTO is true) for
  /// the given SM architecture.  \p arch is e.g. "compute_80".  \p modules is
  /// a list of (buffer, size) pairs — each may be LLVM bitcode or text IR.
  llvm::Expected<std::string>
  compile(llvm::StringRef arch,
          llvm::ArrayRef<std::pair<const char *, size_t>> modules,
          unsigned optLevel = 2, bool genLTO = false);

  /// Get the program log (warnings, errors) from libnvvm.
  std::string getLog(nvvmProgram prog);

  /// Query the NVVM IR and debug-metadata versions supported by the
  /// loaded libnvvm (nvvmIRVersion).
  llvm::Error getIRVersion(int &irMajor, int &irMinor, int &dbgMajor,
                           int &dbgMinor);

private:
  LibNVVMCompiler() = default;
  llvm::Error resolveSymbols();

  std::unique_ptr<CAPILoader> loader;

  // Function pointers from libnvvm
  nvvmResult (*fnCreateProgram)(nvvmProgram *) = nullptr;
  nvvmResult (*fnDestroyProgram)(nvvmProgram *) = nullptr;
  nvvmResult (*fnAddModuleToProgram)(nvvmProgram, const char *, size_t,
                                    const char *) = nullptr;
  nvvmResult (*fnLazyAddModuleToProgram)(nvvmProgram, const char *, size_t,
                                        const char *) = nullptr;
  nvvmResult (*fnCompileProgram)(nvvmProgram, int, const char **) = nullptr;
  nvvmResult (*fnGetCompiledResultSize)(nvvmProgram, size_t *) = nullptr;
  nvvmResult (*fnGetCompiledResult)(nvvmProgram, char *) = nullptr;
  nvvmResult (*fnGetProgramLogSize)(nvvmProgram, size_t *) = nullptr;
  nvvmResult (*fnGetProgramLog)(nvvmProgram, char *) = nullptr;
  nvvmResult (*fnIRVersion)(int *, int *, int *, int *) = nullptr;
};

} // namespace llvm70

#endif // LLVM70_LIBNVVMCOMPILER_H
