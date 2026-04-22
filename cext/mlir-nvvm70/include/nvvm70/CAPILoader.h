/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
//===- CAPILoader.h - Generic dlopen/dlsym utility --------------*- C++ -*-===//
//
// Thin RAII wrapper around dlopen/dlsym for loading shared libraries and
// resolving C API function pointers at runtime.  Used to isolate the old
// LLVM 7 C API and libnvvm from the modern LLVM linked by MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef NVVM70_CAPILOADER_H
#define NVVM70_CAPILOADER_H

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#include <dlfcn.h>
#include <string>

namespace nvvm70 {

class CAPILoader {
public:
  CAPILoader() = default;

  static llvm::Expected<std::unique_ptr<CAPILoader>>
  create(llvm::StringRef libPath) {
    void *handle =
        dlopen(libPath.str().c_str(), RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!handle)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "failed to dlopen '%s': %s", libPath.str().c_str(), dlerror());
    auto loader = std::make_unique<CAPILoader>();
    loader->handle = handle;
    loader->path = libPath.str();
    return std::move(loader);
  }

  ~CAPILoader() {
    if (handle)
      dlclose(handle);
  }

  CAPILoader(const CAPILoader &) = delete;
  CAPILoader &operator=(const CAPILoader &) = delete;
  CAPILoader(CAPILoader &&other) noexcept
      : handle(other.handle), path(std::move(other.path)) {
    other.handle = nullptr;
  }

  template <typename FnPtr>
  llvm::Expected<FnPtr> resolve(const char *symbol) {
    dlerror(); // clear
    void *sym = dlsym(handle, symbol);
    const char *err = dlerror();
    if (err)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "failed to resolve '%s' in '%s': %s", symbol, path.c_str(), err);
    return reinterpret_cast<FnPtr>(sym);
  }

  template <typename FnPtr>
  FnPtr resolveRequired(const char *symbol) {
    auto result = resolve<FnPtr>(symbol);
    if (!result) {
      llvm::errs() << "FATAL: " << llvm::toString(result.takeError()) << "\n";
      abort();
    }
    return *result;
  }

  void *getHandle() const { return handle; }
  const std::string &getPath() const { return path; }

private:
  void *handle = nullptr;
  std::string path;
};

} // namespace nvvm70

#endif // NVVM70_CAPILOADER_H
