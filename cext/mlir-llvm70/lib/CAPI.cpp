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
#include <string>

static constexpr const char *kDefaultDataLayout =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32"
    "-i64:64:64-f32:32:32-f64:64:64"
    "-v16:16:16-v32:32:32-v64:64:64-v128:128:128"
    "-n16:32:64";

static void fatalErrorHandler(void *, const char *reason, bool) {
  throw std::runtime_error(reason ? reason : "unknown fatal error");
}

static char *copyCString(const char *message) {
#ifdef _WIN32
  return _strdup(message);
#else
  return strdup(message);
#endif
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
    int nvvm_ir_major, int nvvm_ir_minor, int nvvm_debug_major,
    int nvvm_debug_minor,
    char **out, size_t *out_len, char **err_out,
    // Deliberately last: a caller built against this signature can invoke a
    // library that predates it (the older callee ignores the trailing
    // argument), and the output pointers never shift. The reverse pairing,
    // an older caller against this library, remains undefined, as with any
    // C signature change.
    const char *nvvm_options) {

  *out = nullptr;
  *out_len = 0;
  *err_out = nullptr;

  if (!raw_op) {
    *err_out = copyCString("null Operation*");
    return 1;
  }

  auto *op = static_cast<mlir::Operation *>(raw_op);
  auto gpuMod = mlir::dyn_cast<mlir::gpu::GPUModuleOp>(op);
  if (!gpuMod) {
    std::string msg = "Operation is not a gpu.module";
    std::string opName = op->getName().getStringRef().str();
    if (!opName.empty()) {
      msg += " (operation name: " + opName + ")";
      if (opName == mlir::gpu::GPUModuleOp::getOperationName()) {
        msg += "; operation name matches gpu.module, which suggests an MLIR "
               "registered-operation TypeID mismatch between MLIRToLLVM70 "
               "and MLIRPythonCAPI";
      }
    }
    *err_out = copyCString(msg.c_str());
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
  opts.nvvmIRMajor = nvvm_ir_major;
  opts.nvvmIRMinor = nvvm_ir_minor;
  opts.nvvmDebugMajor = nvvm_debug_major;
  opts.nvvmDebugMinor = nvvm_debug_minor;

  // Space-separated extra nvvmCompileProgram options (e.g. "-prec-div=0").
  if (nvvm_options && nvvm_options[0]) {
    const char *p = nvvm_options;
    while (*p) {
      while (*p == ' ')
        ++p;
      const char *start = p;
      while (*p && *p != ' ')
        ++p;
      if (p > start)
        opts.nvvmOptions.emplace_back(start, p - start);
    }
  }

  // Intercept llvm::report_fatal_error so it throws an exception instead of
  // aborting the host process (e.g. bf16 rejection, unsupported types).
  // Using throw/catch ensures proper C++ stack unwinding, destructor calls.
  llvm::install_fatal_error_handler(fatalErrorHandler, nullptr);

  try {
    auto ptxOrErr = llvm70::translateToPTX(gpuMod, opts);
    llvm::remove_fatal_error_handler();

    if (!ptxOrErr) {
      std::string msg = llvm::toString(ptxOrErr.takeError());
      *err_out = copyCString(msg.c_str());
      return 1;
    }

    const std::string &ptx = *ptxOrErr;
    *out = static_cast<char *>(malloc(ptx.size()));
    if (!*out) {
      *err_out = copyCString("malloc failed");
      return 1;
    }
    memcpy(*out, ptx.data(), ptx.size());
    *out_len = ptx.size();
    return 0;
  } catch (const std::exception &e) {
    llvm::remove_fatal_error_handler();
    *err_out = copyCString(e.what());
    return 1;
  } catch (...) {
    llvm::remove_fatal_error_handler();
    *err_out = copyCString("unknown exception");
    return 1;
  }
}

void llvm70_free(void *ptr) { free(ptr); }

} // extern "C"
