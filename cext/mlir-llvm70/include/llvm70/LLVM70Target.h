/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
//===- LLVM70Target.h - MLIR → old LLVM IR → PTX via C API -------*- C++ -*-===//
//
// Walks a gpu.module containing LLVM dialect ops, builds LLVM 7 IR
// through the old LLVM C API (via LLVM70IRBuilder), and compiles to PTX through
// libnvvm (via LibNVVMCompiler).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM70_LLVM70TARGET_H
#define LLVM70_LLVM70TARGET_H

#include "llvm70/LibNVVMCompiler.h"
#include "llvm70/LLVM70IRBuilder.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <llvm-c/DebugInfo.h>
#include <string>

namespace mlir {
class ModuleOp;
namespace gpu {
class GPUModuleOp;
} // namespace gpu
} // namespace mlir

namespace llvm70 {

struct LLVM70Options {
  std::string libLLVMPath;  // Path to libLLVM-7.so
  std::string libnvvmPath;  // Path to libnvvm.so
  std::string chip;         // e.g. "sm_80"
  std::string dataLayout;
  std::string triple = "nvptx64-nvidia-cuda";
  unsigned optLevel = 2;    // libnvvm optimization level (0–3)
  bool genLTO = false;      // Pass -gen-lto to produce LTOIR instead of PTX
  // Debug/lineinfo level: 0=none, 1=lineinfo (.file/.loc only), 2=full debug
  int debugLevel = 1;
  /// Extra .bc files to link (libdevice, runtime BCs)
  llvm::SmallVector<std::string> linkLibs;
};

/// Translate a gpu.module (containing LLVM dialect ops) to PTX.
llvm::Expected<std::string> translateToPTX(mlir::gpu::GPUModuleOp gpuMod,
                                           const LLVM70Options &opts);

/// Lower-level: translate a gpu.module to old LLVM IR text (for debugging).
llvm::Expected<std::string> translateToNVVMIR(mlir::gpu::GPUModuleOp gpuMod,
                                              const LLVM70Options &opts);

//===----------------------------------------------------------------------===//
// MLIR → NVVM 7 IR translator
//===----------------------------------------------------------------------===//
class MLIRToLLVM70 {
public:
  MLIRToLLVM70(LLVM70IRBuilder &builder) : b(builder) {}

  /// Translate all ops in the given gpu.module.
  /// debugLevel: 0=none, 1=lineinfo, 2=full debug
  llvm::Error translate(mlir::gpu::GPUModuleOp gpuMod, int debugLevel = 1);

private:
  LLVM70IRBuilder &b;

  // MLIR Value → old LLVMValueRef
  llvm::DenseMap<mlir::Value, LLVMValueRef> valueMap;
  // MLIR Block → old LLVMBasicBlockRef
  llvm::DenseMap<mlir::Block *, LLVMBasicBlockRef> blockMap;

  // Forwarder blocks created by switch translation.
  // Key: MLIR destination block. Value: list of (LLVM trampoline BB, operands).
  using ForwarderList =
      llvm::SmallVector<std::pair<LLVMBasicBlockRef, mlir::OperandRange>>;
  llvm::DenseMap<mlir::Block *, ForwarderList> switchForwarders;

  // Debug info state
  LLVMMetadataRef diCompileUnit = nullptr;
  LLVMMetadataRef diSubroutineType = nullptr;
  llvm::StringMap<LLVMMetadataRef> diFileCache;
  LLVMMetadataRef currentSubprogram = nullptr;

  LLVMMetadataRef getOrCreateDIFile(llvm::StringRef filename);
  void setDebugLocFromOp(mlir::Operation *op);
  std::tuple<llvm::StringRef, unsigned, unsigned>
  extractFileLineCol(mlir::Location loc);

  // Map an MLIR value to its old-LLVM counterpart.
  void mapValue(mlir::Value v, LLVMValueRef lv) { valueMap[v] = lv; }
  LLVMValueRef lookupValue(mlir::Value v);

  // Type conversion: MLIR type → LLVM 7 type.
  // For ptr types the element type must be recovered from context.
  LLVMTypeRef convertType(mlir::Type ty);

  // Op handlers
  llvm::Error translateGlobalOp(mlir::Operation *op);
  llvm::Error translateFuncOp(mlir::Operation *op);
  llvm::Error translateBlock(mlir::Block &block);
  llvm::Error translateOp(mlir::Operation *op);

  // Individual op translators (return Error on unsupported)
  llvm::Error translateReturnOp(mlir::Operation *op);
  llvm::Error translateBrOp(mlir::Operation *op);
  llvm::Error translateCondBrOp(mlir::Operation *op);
  llvm::Error translateCallOp(mlir::Operation *op);
  llvm::Error translateConstantOp(mlir::Operation *op);
  llvm::Error translateUndefOp(mlir::Operation *op);
  llvm::Error translatePoisonOp(mlir::Operation *op);
  llvm::Error translateZeroOp(mlir::Operation *op);
  llvm::Error translateAddressOfOp(mlir::Operation *op);
  llvm::Error translateArithOp(mlir::Operation *op);
  llvm::Error translateICmpOp(mlir::Operation *op);
  llvm::Error translateFCmpOp(mlir::Operation *op);
  llvm::Error translateLoadOp(mlir::Operation *op);
  llvm::Error translateStoreOp(mlir::Operation *op);
  llvm::Error translateAllocaOp(mlir::Operation *op);
  llvm::Error translateGEPOp(mlir::Operation *op);
  llvm::Error translateCastOp(mlir::Operation *op);
  llvm::Error translateSelectOp(mlir::Operation *op);
  llvm::Error translateExtractValueOp(mlir::Operation *op);
  llvm::Error translateInsertValueOp(mlir::Operation *op);
  llvm::Error translateExtractElementOp(mlir::Operation *op);
  llvm::Error translateInsertElementOp(mlir::Operation *op);
  llvm::Error translateAtomicRMWOp(mlir::Operation *op);
  llvm::Error translateFloatAtomicCASLoop(mlir::Operation *op);
  llvm::Error translateAtomicCmpXchgOp(mlir::Operation *op);
  llvm::Error translateInlineAsmOp(mlir::Operation *op);
  llvm::Error translateFrexpOp(mlir::Operation *op);
  llvm::Error translateLdexpOp(mlir::Operation *op);
  llvm::Error translateSimpleIntIntrinsic(mlir::Operation *op,
                                          llvm::StringRef intrBase);
  llvm::Error translateSwitchOp(mlir::Operation *op);
  llvm::Error translateUnaryFloatIntrinsic(mlir::Operation *op,
                                           llvm::StringRef intrBase);
  llvm::Error translateBinaryIntIntrinsic(mlir::Operation *op,
                                          LLVMIntPredicate pred);
  llvm::Error translateAbsOp(mlir::Operation *op);
  llvm::Error translateBinaryFloatIntrinsic(mlir::Operation *op,
                                            llvm::StringRef intrBase);
  llvm::Error translateMinimumMaximumOp(mlir::Operation *op,
                                        llvm::StringRef minnumMaxnum);
  llvm::Error translateMemsetOp(mlir::Operation *op);
  llvm::Error translateMemcpyOp(mlir::Operation *op);
  llvm::Error translateLifetimeOp(mlir::Operation *op, bool isStart);
  llvm::Error translateClusterArriveOp(mlir::Operation *op, bool isRelaxed);
  llvm::Error translateClusterWaitOp(mlir::Operation *op);
  llvm::Error translateBarrierOp(mlir::Operation *op);
  llvm::Error translateCttzCtlzOp(mlir::Operation *op, llvm::StringRef intrBase,
                                  mlir::Value in, mlir::Value res,
                                  bool isZeroPoison);
  llvm::Error translateNVVMOp(mlir::Operation *op);
  llvm::Error translateSregOp(mlir::Operation *op);
  llvm::Error translateBarrier0Op(mlir::Operation *op);
  llvm::Error translateFmaOp(mlir::Operation *op);
  llvm::Error translateElectSyncOp(mlir::Operation *op);
  llvm::Error translateMatchSyncOp(mlir::Operation *op);
  llvm::Error translateNanosleepOp(mlir::Operation *op);
  llvm::Error translateSyncWarpOp(mlir::Operation *op);
  llvm::Error translateVoteSyncOp(mlir::Operation *op);
  llvm::Error translateShflOp(mlir::Operation *op);
  llvm::Error translateReduxSyncOp(mlir::Operation *op);
  llvm::Error translateMembarOp(mlir::Operation *op);
  llvm::Error translatePhiOps(mlir::Block &block);
  llvm::Error translateDbgDeclareOp(mlir::Operation *op);
  llvm::Error translateDbgValueOp(mlir::Operation *op);
  llvm::Error emitDbgIntrinsic(mlir::Operation *op, LLVMValueRef val,
                               mlir::LLVM::DILocalVariableAttr varInfo,
                               bool isDeclare);

  LLVMMetadataRef getOrCreateDIType(mlir::LLVM::DITypeAttr typeAttr);

  void emitKernelMetadata(LLVMValueRef fn, mlir::Operation *funcOp);

  // bf16 promotion helpers (LLVM 7 has no bfloat type; we use i16 <-> f32).
  LLVMValueRef bf16ToF32(LLVMValueRef i16Val);
  LLVMValueRef f32ToBf16(LLVMValueRef f32Val);
};

} // namespace llvm70

#endif // LLVM70_LLVM70TARGET_H
