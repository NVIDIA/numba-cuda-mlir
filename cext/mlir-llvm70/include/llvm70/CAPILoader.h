/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
//===- CAPILoader.h - Generic dynamic loading utility -----------*- C++ -*-===//
//
// Thin RAII wrapper around platform dynamic loading APIs and
// resolving C API function pointers at runtime.  Used to isolate the old
// LLVM 7 C API and libnvvm from the modern LLVM linked by MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM70_CAPILOADER_H
#define LLVM70_CAPILOADER_H

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <string>

namespace llvm70 {

class CAPILoader {
public:
  CAPILoader() = default;

  static llvm::Expected<std::unique_ptr<CAPILoader>>
  create(llvm::StringRef libPath) {
#ifdef _WIN32
    HMODULE handle = nullptr;
    std::string path = libPath.str();
    bool ownsHandle = true;
    if (path.empty()) {
      GetModuleHandleExA(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS,
                         reinterpret_cast<LPCSTR>(&CAPILoader::create), &handle);
      path = "<current module>";
    } else {
      handle = LoadLibraryA(path.c_str());
    }
    if (!handle)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "failed to LoadLibrary '%s': error %lu", path.c_str(), GetLastError());
#else
    void *handle =
        dlopen(libPath.str().c_str(), RTLD_LAZY | RTLD_LOCAL | RTLD_DEEPBIND);
    if (!handle)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "failed to dlopen '%s': %s", libPath.str().c_str(), dlerror());
#endif
    auto loader = std::make_unique<CAPILoader>();
    loader->handle = reinterpret_cast<void *>(handle);
    loader->ownsHandle = ownsHandle;
    loader->path = libPath.str();
#ifdef _WIN32
    if (loader->path.empty())
      loader->path = "<current module>";
#endif
    return std::move(loader);
  }

  ~CAPILoader() {
    if (handle) {
#ifdef _WIN32
      if (ownsHandle)
        FreeLibrary(reinterpret_cast<HMODULE>(handle));
#else
      dlclose(handle);
#endif
    }
  }

  CAPILoader(const CAPILoader &) = delete;
  CAPILoader &operator=(const CAPILoader &) = delete;
  CAPILoader(CAPILoader &&other) noexcept
      : handle(other.handle), ownsHandle(other.ownsHandle),
        path(std::move(other.path)) {
    other.handle = nullptr;
    other.ownsHandle = false;
  }

  template <typename FnPtr>
  llvm::Expected<FnPtr> resolve(const char *symbol) {
#ifdef _WIN32
    FARPROC sym = GetProcAddress(reinterpret_cast<HMODULE>(handle), symbol);
    if (!sym)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "failed to resolve '%s' in '%s': error %lu", symbol, path.c_str(),
          GetLastError());
    return reinterpret_cast<FnPtr>(sym);
#else
    dlerror(); // clear
    void *sym = dlsym(handle, symbol);
    const char *err = dlerror();
    if (err)
      return llvm::createStringError(
          llvm::inconvertibleErrorCode(),
          "failed to resolve '%s' in '%s': %s", symbol, path.c_str(), err);
    return reinterpret_cast<FnPtr>(sym);
#endif
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
  bool ownsHandle = true;
  std::string path;
};

} // namespace llvm70

#endif // LLVM70_CAPILOADER_H
