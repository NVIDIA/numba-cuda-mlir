/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
//===- nvvm70-translate.cpp - MLIR → PTX via NVVM70 target --------*- C++ -*-===//
//
// Reads an MLIR file containing a gpu.module with a #nvvm70.target attribute,
// serializes GPU modules to PTX via the old LLVM C API and libnvvm, and either
// prints the resulting MLIR with gpu.binary operations, or (with
// --mlir-to-llvm-ir) translates the whole module to host LLVM IR with
// embedded PTX.
//
// Usage:
//   nvvm70-translate input.mlir [-o output.mlir]
//       [--llvm-dump-path=ir.ll] [--ptx-dump-path=kernel.ptx]
//       [--dump-llvm] [--dump-ptx]
//       [--mlir-to-llvm-ir]
//
// Library paths are resolved from the #nvvm70.target attribute (libllvm, libnvvm
// parameters) or from LIBLLVM7 / NVVM70_LIBNVVM environment variables.
//
//===----------------------------------------------------------------------===//

#include "nvvm70/Dialect/NVVM70.h"
#include "nvvm70/Target/Target.h"

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace mlir;

static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input .mlir file>"),
                                          cl::Required);
static cl::opt<std::string> outputFilename("o", cl::desc("Output file"),
                                           cl::value_desc("filename"),
                                           cl::init("-"));
static cl::opt<std::string>
    ptxDumpPath("ptx-dump-path",
                cl::desc("Dump generated PTX to this file path"),
                cl::value_desc("path"), cl::init(""));
static cl::opt<std::string>
    llvmDumpPath("llvm-dump-path",
                 cl::desc("Dump generated NVVM70 LLVM IR to this file path"),
                 cl::value_desc("path"), cl::init(""));
static cl::opt<bool> dumpLLVM("dump-llvm",
                              cl::desc("Print NVVM70 LLVM IR to stderr"),
                              cl::init(false));
static cl::opt<bool> dumpPTX("dump-ptx",
                             cl::desc("Print generated PTX to stderr"),
                             cl::init(false));
static cl::opt<bool>
    mlirToLLVMIR("mlir-to-llvm-ir",
                 cl::desc("Translate to host LLVM IR (with embedded PTX) "
                          "instead of outputting MLIR"),
                 cl::init(false));
static cl::opt<std::string>
    chipOverride("chip",
                 cl::desc("Override the SM architecture (e.g. sm_80)"),
                 cl::value_desc("sm_XX"), cl::init(""));

int main(int argc, char **argv) {
  InitLLVM initLLVM(argc, argv);
  cl::ParseCommandLineOptions(argc, argv, "NVVM70 MLIR-to-PTX translator\n");

  // Set up MLIR context with required dialects.
  MLIRContext context;
  DialectRegistry registry;
  registry.insert<gpu::GPUDialect, LLVM::LLVMDialect, NVVM::NVVMDialect,
                  mlir::nvvm70::NVVM70Dialect>();
  ::nvvm70::registerNVVM70TargetInterfaceExternalModels(registry);
  registerAllToLLVMIRTranslations(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();

  // Parse the input MLIR file.
  auto srcMgr = std::make_shared<llvm::SourceMgr>();
  std::string errMsg;
  auto inputFile = openInputFile(inputFilename, &errMsg);
  if (!inputFile) {
    llvm::errs() << "error: " << errMsg << "\n";
    return 1;
  }
  srcMgr->AddNewSourceBuffer(std::move(inputFile), SMLoc());

  auto module = parseSourceFile<ModuleOp>(*srcMgr, &context);
  if (!module) {
    llvm::errs() << "error: failed to parse MLIR input\n";
    return 1;
  }

  // Build TargetOptions with callbacks for dumping IR/PTX.
  std::string cmdOpts;
  if (!llvmDumpPath.empty())
    cmdOpts += " --nvvm70-ir-dump=" + llvmDumpPath;
  if (dumpLLVM)
    cmdOpts += " --nvvm70-ir-stderr";
  if (!chipOverride.empty())
    cmdOpts += " --nvvm70-chip=" + chipOverride;

  auto ptxCallback = [&](StringRef ptx) {
    if (!ptxDumpPath.empty()) {
      std::error_code ec;
      llvm::raw_fd_ostream stream(ptxDumpPath, ec, llvm::sys::fs::OF_Text);
      if (ec)
        llvm::errs() << "error: cannot open PTX dump file: " << ec.message()
                     << "\n";
      else
        stream << ptx;
    }
    if (dumpPTX)
      llvm::errs() << ptx;
  };

  llvm::function_ref<void(StringRef)> isaCallback = {};
  if (!ptxDumpPath.empty() || dumpPTX)
    isaCallback = ptxCallback;

  gpu::TargetOptions targetOptions(
      /*toolkitPath=*/{}, /*librariesToLink=*/{}, cmdOpts,
      /*elfSection=*/{}, gpu::TargetOptions::getDefaultCompilationTarget(),
      /*getSymbolTableCallback=*/{},
      /*initialLlvmIRCallback=*/{}, /*linkedLlvmIRCallback=*/{},
      /*optimizedLlvmIRCallback=*/{}, isaCallback);

  if (failed(gpu::transformGpuModulesToBinaries(*module,
                                                /*handler=*/nullptr,
                                                targetOptions))) {
    llvm::errs() << "error: gpu-module-to-binary failed\n";
    return 1;
  }

  // Output the result.
  auto outFile = openOutputFile(outputFilename, &errMsg);
  if (!outFile) {
    llvm::errs() << "error: " << errMsg << "\n";
    return 1;
  }

  if (mlirToLLVMIR) {
    llvm::LLVMContext llvmContext;
    auto llvmModule =
        translateModuleToLLVMIR(*module, llvmContext, "nvvm70_module");
    if (!llvmModule) {
      llvm::errs() << "error: failed to translate MLIR to LLVM IR\n";
      return 1;
    }
    llvmModule->print(outFile->os(), /*AssemblyAnnotationWriter=*/nullptr);
  } else {
    module->print(outFile->os());
  }

  outFile->keep();
  return 0;
}
