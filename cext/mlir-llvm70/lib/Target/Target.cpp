/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
//===- Target.cpp - LLVM70 target compilation --------------------*- C++ -*-===//
//
// Implements the TargetAttrInterface for #nvvm_llvm70.target.  serializeToObject()
// bypasses the normal ModuleToObject pipeline and instead:
//   1. Walks the MLIR gpu.module directly with MLIRToLLVM70
//   2. Builds LLVM 7 IR via the LLVM 7.0.1 C API (dlopen'd)
//   3. Compiles to PTX via libnvvm (dlopen'd)
//
//===----------------------------------------------------------------------===//

#include "llvm70/Target/Target.h"
#include "llvm70/Dialect/LLVM70.h"
#include "llvm70/LLVM70Target.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"

using namespace mlir;
using namespace mlir::llvm70;

static constexpr const char *kDefaultDataLayout =
    "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32"
    "-i64:64:64-f32:32:32-f64:64:64"
    "-v16:16:16-v32:32:32-v64:64:64-v128:128:128"
    "-n16:32:64";

namespace {
class LLVM70TargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<LLVM70TargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;

  Attribute createObject(Attribute attribute, Operation *module,
                         const gpu::SerializedObject &object,
                         const gpu::TargetOptions &options) const;
};
} // namespace

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

void ::llvm70::registerLLVM70TargetInterfaceExternalModels(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, mlir::llvm70::LLVM70Dialect *dialect) {
        LLVM70TargetAttr::attachInterface<LLVM70TargetAttrImpl>(*ctx);
      });
}

void ::llvm70::registerLLVM70TargetInterfaceExternalModels(MLIRContext &context) {
  DialectRegistry registry;
  ::llvm70::registerLLVM70TargetInterfaceExternalModels(registry);
  context.appendDialectRegistry(registry);
}

//===----------------------------------------------------------------------===//
// Helper: resolve paths from attribute, env, or TargetOptions
//===----------------------------------------------------------------------===//

static std::string resolveLibLLVMPath(LLVM70TargetAttr target) {
  StringRef path = target.getLibllvm();
  if (!path.empty())
    return path.str();
  if (const char *env = std::getenv("LIBLLVM7"))
    return env;
  return {};
}

static std::string resolveLibNVVMPath(LLVM70TargetAttr target,
                                      StringRef toolkitPath) {
  StringRef path = target.getLibnvvm();
  if (!path.empty())
    return path.str();
  if (const char *env = std::getenv("LLVM70_LIBNVVM"))
    return env;
  if (!toolkitPath.empty()) {
    llvm::SmallString<256> p(toolkitPath);
    llvm::sys::path::append(p, "nvvm", "lib64", "libnvvm.so");
    return std::string(p);
  }
  return {};
}

//===----------------------------------------------------------------------===//
// serializeToObject — the core: MLIR → old LLVM IR → PTX
//===----------------------------------------------------------------------===//

std::optional<SmallVector<char, 0>> LLVM70TargetAttrImpl::serializeToObject(
    Attribute attribute, Operation *module,
    const gpu::TargetOptions &options) const {
  assert(module && "The module must be non null.");
  if (!module)
    return std::nullopt;

  auto gpuMod = dyn_cast<gpu::GPUModuleOp>(module);
  if (!gpuMod) {
    module->emitError("Module must be a GPU module.");
    return std::nullopt;
  }

  auto target = cast<LLVM70TargetAttr>(attribute);

  // Resolve paths.
  std::string libllvmPath = resolveLibLLVMPath(target);
  if (libllvmPath.empty()) {
    module->emitError()
        << "LLVM70: libLLVM path not set. Use #nvvm_llvm70.target<libllvm=...>, "
           "set LIBLLVM7 env, or pass via target options.";
    return std::nullopt;
  }

  std::string libnvvmPath =
      resolveLibNVVMPath(target, options.getToolkitPath());
  if (libnvvmPath.empty()) {
    module->emitError()
        << "LLVM70: libnvvm path not set. Use #nvvm_llvm70.target<libnvvm=...>, "
           "set LLVM70_LIBNVVM env, or set CUDA_ROOT.";
    return std::nullopt;
  }

  // Build LLVM70Options.
  ::llvm70::LLVM70Options opts;
  opts.libLLVMPath = libllvmPath;
  opts.libnvvmPath = libnvvmPath;
  opts.chip = target.getChip().str();
  opts.triple = target.getTriple().str();
  opts.optLevel = target.getO();
  opts.dataLayout = target.getDataLayout().empty()
                        ? kDefaultDataLayout
                        : target.getDataLayout().str();

  // Collect link libraries from the attribute.
  if (ArrayAttr linkAttr = target.getLink()) {
    for (Attribute a : linkAttr) {
      if (auto strAttr = dyn_cast<StringAttr>(a))
        opts.linkLibs.push_back(strAttr.getValue().str());
    }
  }

  // Append libdevice from toolkit path if not explicitly provided.
  if (opts.linkLibs.empty() && !options.getToolkitPath().empty()) {
    llvm::SmallString<256> libdevicePath(options.getToolkitPath());
    llvm::sys::path::append(libdevicePath, "nvvm", "libdevice",
                            "libdevice.10.bc");
    if (llvm::sys::fs::exists(libdevicePath))
      opts.linkLibs.push_back(std::string(libdevicePath));
  }

  // Check if any LLVM IR dump was requested via cmdOptions.
  StringRef cmdOpts = options.getCmdOptions();
  constexpr StringLiteral kIRDumpFlag("--llvm70-ir-dump=");
  constexpr StringLiteral kIRStderrFlag("--llvm70-ir-stderr");
  constexpr StringLiteral kChipFlag("--llvm70-chip=");
  bool dumpIRToFile = cmdOpts.contains(kIRDumpFlag);
  bool dumpIRToStderr = cmdOpts.contains(kIRStderrFlag);

  // --chip override from cmdOptions or LLVM70_CHIP env var.
  if (cmdOpts.contains(kChipFlag)) {
    StringRef tail =
        cmdOpts.substr(cmdOpts.find(kChipFlag) + kChipFlag.size());
    opts.chip = tail.split(' ').first.str();
  } else if (const char *envChip = std::getenv("LLVM70_CHIP")) {
    if (envChip[0] != '\0')
      opts.chip = envChip;
  }

  // Validate SM range: supports sm_75 through sm_9x.
  {
    unsigned smVersion = 0;
    StringRef chip(opts.chip);
    if (chip.starts_with("sm_") &&
        !chip.drop_front(3).getAsInteger(10, smVersion)) {
      if (smVersion < 75) {
        module->emitError()
            << "LLVM70 does not support " << opts.chip
            << ". Minimum supported architecture is sm_75.";
        return std::nullopt;
      }
      if (smVersion >= 100) {
        module->emitError()
            << "LLVM70 does not support " << opts.chip
            << " (sm_100+). Use the standard NVVM pipeline for "
               "Blackwell and later architectures.";
        return std::nullopt;
      }
    }
  }

  if (dumpIRToFile || dumpIRToStderr) {
    auto irOrErr = ::llvm70::translateToNVVMIR(gpuMod, opts);
    if (!irOrErr) {
      module->emitError() << "LLVM70 IR generation failed: "
                          << llvm::toString(irOrErr.takeError());
      return std::nullopt;
    }
    if (dumpIRToFile) {
      StringRef tail = cmdOpts.substr(cmdOpts.find(kIRDumpFlag) +
                                      kIRDumpFlag.size());
      std::string irDumpPath = tail.split(' ').first.str();
      std::error_code ec;
      llvm::raw_fd_ostream stream(irDumpPath, ec, llvm::sys::fs::OF_Text);
      if (ec)
        module->emitError() << "Cannot open NVVM IR dump file '" << irDumpPath
                            << "': " << ec.message();
      else
        stream << *irOrErr;
    }
    if (dumpIRToStderr)
      llvm::errs() << *irOrErr;
  }

  // Translate MLIR → PTX via the old LLVM C API + libnvvm.
  auto ptxOrErr = ::llvm70::translateToPTX(gpuMod, opts);
  if (!ptxOrErr) {
    module->emitError() << "LLVM70 translation failed: "
                        << llvm::toString(ptxOrErr.takeError());
    return std::nullopt;
  }

  // Invoke the ISA callback (if set) to let the caller dump/inspect PTX.
  if (auto isaCallback = options.getISACallback())
    isaCallback(*ptxOrErr);

  StringRef ptx = *ptxOrErr;
  return SmallVector<char, 0>(ptx.begin(), ptx.end());
}

//===----------------------------------------------------------------------===//
// createObject — wrap the serialized PTX into a gpu.object attribute
//===----------------------------------------------------------------------===//

Attribute LLVM70TargetAttrImpl::createObject(
    Attribute attribute, Operation *module,
    const gpu::SerializedObject &object,
    const gpu::TargetOptions &options) const {
  gpu::CompilationTarget format = options.getCompilationTarget();
  Builder builder(attribute.getContext());
  return builder.getAttr<gpu::ObjectAttr>(
      attribute, format,
      builder.getStringAttr(StringRef(object.getObject().data(),
                                      object.getObject().size())),
      /*properties=*/nullptr, /*kernels=*/nullptr);
}
