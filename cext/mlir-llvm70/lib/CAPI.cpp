/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include "llvm70/LLVM70Target.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/Operation.h"
#include "llvm/Support/ErrorHandling.h"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

static constexpr const char *kDefaultDataLayout =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32"
    "-i64:64:64-f32:32:32-f64:64:64"
    "-v16:16:16-v32:32:32-v64:64:64-v128:128:128"
    "-n16:32:64";

static void fatalErrorHandler(void *, const char *reason, bool) {
  throw std::runtime_error(reason ? reason : "unknown fatal error");
}

extern "C" {

/// Translate a gpu.module Operation* directly to PTX.
/// The caller passes a raw mlir::Operation* (as void*) that must be a
/// gpu::GPUModuleOp. All paths (libllvm, libnvvm, libdevice) are passed
/// explicitly — no text parsing, no MLIRContext creation.
int llvm70_translate_gpu_module_from_op(
    void *raw_op,
    const char *chip, const char *data_layout,
    const char *libllvm, const char *libnvvm, const char *libdevice,
    int gen_lto, int opt_level, int gen_lineinfo,
    char **out, size_t *out_len, char **err_out) {

  *out = nullptr;
  *out_len = 0;
  *err_out = nullptr;

  if (!raw_op) {
    *err_out = strdup("null Operation*");
    return 1;
  }

  auto *op = static_cast<mlir::Operation *>(raw_op);
  auto gpuMod = mlir::dyn_cast<mlir::gpu::GPUModuleOp>(op);
  if (!gpuMod) {
    *err_out = strdup("Operation is not a gpu.module");
    return 1;
  }

  llvm70::LLVM70Options opts;
  opts.chip = chip ? chip : "sm_80";
  opts.dataLayout = (data_layout && data_layout[0]) ? data_layout
                                                    : kDefaultDataLayout;
  if (libllvm && libllvm[0]) opts.libLLVMPath = libllvm;
  if (libnvvm && libnvvm[0]) opts.libnvvmPath = libnvvm;
  if (libdevice && libdevice[0]) opts.linkLibs.push_back(libdevice);
  opts.genLTO = gen_lto != 0;
  opts.optLevel = (opt_level >= 0 && opt_level <= 3) ? opt_level : 2;
  opts.debugLevel = gen_lineinfo;  // 0=none, 1=lineinfo, 2=full debug

  // Intercept llvm::report_fatal_error so it throws an exception instead of
  // aborting the host process (e.g. bf16 rejection, unsupported types).
  // Using throw/catch ensures proper C++ stack unwinding, destructor calls.
  llvm::install_fatal_error_handler(fatalErrorHandler, nullptr);

  try {
    auto ptxOrErr = llvm70::translateToPTX(gpuMod, opts);
    llvm::remove_fatal_error_handler();

    if (!ptxOrErr) {
      std::string msg = llvm::toString(ptxOrErr.takeError());
      *err_out = strdup(msg.c_str());
      return 1;
    }

    const std::string &ptx = *ptxOrErr;
    *out = static_cast<char *>(malloc(ptx.size()));
    if (!*out) {
      *err_out = strdup("malloc failed");
      return 1;
    }
    memcpy(*out, ptx.data(), ptx.size());
    *out_len = ptx.size();
    return 0;
  } catch (const std::exception &e) {
    llvm::remove_fatal_error_handler();
    *err_out = strdup(e.what());
    return 1;
  } catch (...) {
    llvm::remove_fatal_error_handler();
    *err_out = strdup("unknown exception");
    return 1;
  }
}

void llvm70_free(void *ptr) { free(ptr); }

} // extern "C"
