/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
//===- LibNVVMCompiler.cpp - libnvvm dlopen wrapper -------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "llvm70/LibNVVMCompiler.h"

using namespace llvm70;

llvm::Expected<std::unique_ptr<LibNVVMCompiler>>
LibNVVMCompiler::create(llvm::StringRef libnvvmPath) {
  auto loaderOrErr = CAPILoader::create(libnvvmPath);
  if (!loaderOrErr)
    return loaderOrErr.takeError();

  auto c = std::unique_ptr<LibNVVMCompiler>(new LibNVVMCompiler());
  c->loader = std::move(*loaderOrErr);

  if (auto err = c->resolveSymbols())
    return std::move(err);

  return std::move(c);
}

LibNVVMCompiler::~LibNVVMCompiler() = default;

#define RESOLVE(FIELD, SYM)                                                    \
  do {                                                                         \
    auto r = loader->resolve<decltype(FIELD)>(SYM);                            \
    if (!r)                                                                    \
      return r.takeError();                                                    \
    FIELD = *r;                                                                \
  } while (0)

llvm::Error LibNVVMCompiler::resolveSymbols() {
  RESOLVE(fnCreateProgram, "nvvmCreateProgram");
  RESOLVE(fnDestroyProgram, "nvvmDestroyProgram");
  RESOLVE(fnAddModuleToProgram, "nvvmAddModuleToProgram");
  RESOLVE(fnLazyAddModuleToProgram, "nvvmLazyAddModuleToProgram");
  RESOLVE(fnCompileProgram, "nvvmCompileProgram");
  RESOLVE(fnGetCompiledResultSize, "nvvmGetCompiledResultSize");
  RESOLVE(fnGetCompiledResult, "nvvmGetCompiledResult");
  RESOLVE(fnGetProgramLogSize, "nvvmGetProgramLogSize");
  RESOLVE(fnGetProgramLog, "nvvmGetProgramLog");
  return llvm::Error::success();
}

#undef RESOLVE

llvm::Expected<std::string> LibNVVMCompiler::compile(
    llvm::StringRef arch,
    llvm::ArrayRef<std::pair<const char *, size_t>> modules,
    unsigned optLevel, bool genLTO) {
  nvvmProgram prog = nullptr;
  nvvmResult rc = fnCreateProgram(&prog);
  if (rc != NVVM_SUCCESS)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "nvvmCreateProgram failed (%d)", (int)rc);

  for (auto [i, mod] : llvm::enumerate(modules)) {
    std::string modName = "module_" + std::to_string(i);
    auto addModule = i == 0 ? fnAddModuleToProgram : fnLazyAddModuleToProgram;
    rc = addModule(prog, mod.first, mod.second, modName.c_str());
    if (rc != NVVM_SUCCESS) {
      std::string log = getLog(prog);
      fnDestroyProgram(&prog);
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "%s failed (%d): %s",
          i == 0 ? "nvvmAddModuleToProgram" : "nvvmLazyAddModuleToProgram",
          (int)rc, log.c_str());
    }
  }

  std::string archOpt = "-arch=" + arch.str();
  std::string optOpt = "-opt=" + std::to_string(optLevel);
  llvm::SmallVector<const char *, 8> opts;
  opts.push_back(archOpt.c_str());
  opts.push_back(optOpt.c_str());
  if (genLTO)
    opts.push_back("-gen-lto");
  rc = fnCompileProgram(prog, opts.size(), opts.data());
  if (rc != NVVM_SUCCESS) {
    std::string log = getLog(prog);
    fnDestroyProgram(&prog);
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "nvvmCompileProgram failed (%d): %s",
                                   (int)rc, log.c_str());
  }

  size_t ptxSize = 0;
  rc = fnGetCompiledResultSize(prog, &ptxSize);
  if (rc != NVVM_SUCCESS) {
    fnDestroyProgram(&prog);
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "nvvmGetCompiledResultSize failed (%d)",
                                   (int)rc);
  }
  std::string ptx(ptxSize, '\0');
  rc = fnGetCompiledResult(prog, ptx.data());
  if (rc != NVVM_SUCCESS) {
    fnDestroyProgram(&prog);
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "nvvmGetCompiledResult failed (%d)",
                                   (int)rc);
  }
  if (!genLTO && ptxSize > 0 && ptx.back() == '\0')
    ptx.pop_back();

  fnDestroyProgram(&prog);
  return ptx;
}

std::string LibNVVMCompiler::getLog(nvvmProgram prog) {
  std::string log;
  size_t logSize = 0;
  fnGetProgramLogSize(prog, &logSize);
  if (logSize > 1) {
    log.resize(logSize);
    fnGetProgramLog(prog, log.data());
  }
  return log;
}
