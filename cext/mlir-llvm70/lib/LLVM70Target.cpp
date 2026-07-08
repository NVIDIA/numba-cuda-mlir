/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
//===- LLVM70Target.cpp - MLIR → old LLVM IR → PTX via C API -----*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "llvm70/LLVM70Target.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/FormatVariadic.h"
#include <cstring>
#include <fstream>

#define DEBUG_TYPE "llvm70-target"

using namespace llvm70;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Public entry points
//===----------------------------------------------------------------------===//

llvm::Expected<std::string> llvm70::translateToNVVMIR(gpu::GPUModuleOp gpuMod,
                                                     const LLVM70Options &opts) {
  auto builderOrErr = LLVM70IRBuilder::create(opts.libLLVMPath);
  if (!builderOrErr)
    return builderOrErr.takeError();
  auto &builder = *builderOrErr;

  builder->setTarget(opts.triple.c_str());
  if (!opts.dataLayout.empty())
    builder->setDataLayout(opts.dataLayout.c_str());

  MLIRToLLVM70 translator(*builder);
  if (auto err = translator.translate(gpuMod, opts.debugLevel, opts.genLTO, &opts))
    return std::move(err);

  return builder->printModuleToString();
}

llvm::Expected<std::string> llvm70::translateToPTX(gpu::GPUModuleOp gpuMod,
                                                  const LLVM70Options &opts) {
  auto builderOrErr = LLVM70IRBuilder::create(opts.libLLVMPath);
  if (!builderOrErr)
    return builderOrErr.takeError();
  auto &builder = *builderOrErr;

  builder->setTarget(opts.triple.c_str());
  if (!opts.dataLayout.empty())
    builder->setDataLayout(opts.dataLayout.c_str());

  MLIRToLLVM70 translator(*builder);
  if (auto err = translator.translate(gpuMod, opts.debugLevel, opts.genLTO, &opts))
    return std::move(err);

  LLVM_DEBUG({
    llvm::dbgs() << "=== Generated NVVM IR ===\n"
                 << builder->printModuleToString() << "\n";
  });

  // Serialize to bitcode
  LLVMMemoryBufferRef buf = builder->writeBitcodeToMemoryBuffer();
  const char *bcData = builder->getBufferStart(buf);
  size_t bcSize = builder->getBufferSize(buf);

  // Collect modules: our bitcode + any link libraries
  llvm::SmallVector<std::pair<const char *, size_t>> modules;
  modules.push_back({bcData, bcSize});

  // Read link libraries (libdevice, runtime BCs)
  llvm::SmallVector<std::string> libBuffers;
  for (const auto &libPath : opts.linkLibs) {
    std::ifstream file(libPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      builder->disposeMemoryBuffer(buf);
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "cannot open link library: %s",
                                     libPath.c_str());
    }
    size_t sz = file.tellg();
    file.seekg(0);
    libBuffers.emplace_back(sz, '\0');
    file.read(libBuffers.back().data(), sz);
    modules.push_back({libBuffers.back().data(), sz});
  }

  // Compile via libnvvm
  auto compilerOrErr = LibNVVMCompiler::create(opts.libnvvmPath);
  if (!compilerOrErr) {
    builder->disposeMemoryBuffer(buf);
    return compilerOrErr.takeError();
  }

  std::string computeArch =
      "compute_" + opts.chip.substr(opts.chip.find_first_of("0123456789"));
  auto ptxOrErr =
      (*compilerOrErr)->compile(computeArch, modules, opts.optLevel, opts.genLTO);

  builder->disposeMemoryBuffer(buf);
  return ptxOrErr;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static uint64_t toUInt64(const llvm::APInt &val) {
  if (val.getActiveBits() > 64)
    llvm::report_fatal_error("llvm70: integer constant exceeds 64 bits",
        /*GenCrashDiag=*/false);
  return val.getZExtValue();
}

static bool isBF16(Type ty) { return isa<BFloat16Type>(ty); }

// Intrinsic suffix for a float type; bf16 promotes to f32 for computation.
static const char *floatSuffix(Type ty) {
  if (isBF16(ty) || ty.isF32()) return "f32";
  if (ty.isF16()) return "f16";
  if (ty.isF64()) return "f64";
  return nullptr;
}

static LLVMValueRef constBF16(LLVM70IRBuilder &b, const llvm::APFloat &v) {
  return b.constInt(b.i16Ty(), v.bitcastToAPInt().getZExtValue(), false);
}

// bf16 (stored as i16) → f32: zero-extend to i32, shift left 16, bitcast.
LLVMValueRef MLIRToLLVM70::bf16ToF32(LLVMValueRef i16Val) {
  LLVMValueRef i32Val = b.buildZExt(i16Val, b.i32Ty(), "");
  LLVMValueRef shifted = b.buildShl(i32Val, b.constInt(b.i32Ty(), 16, false), "");
  return b.buildBitCast(shifted, b.floatTy(), "");
}

// f32 → bf16 (i16) with round-to-nearest-even.
LLVMValueRef MLIRToLLVM70::f32ToBf16(LLVMValueRef f32Val) {
  LLVMValueRef bits = b.buildBitCast(f32Val, b.i32Ty(), "");
  LLVMValueRef lsb = b.buildLShr(bits, b.constInt(b.i32Ty(), 16, false), "");
  lsb = b.buildAnd(lsb, b.constInt(b.i32Ty(), 1, false), "");
  LLVMValueRef rounding = b.buildAdd(lsb, b.constInt(b.i32Ty(), 0x7FFF, false), "");
  bits = b.buildAdd(bits, rounding, "");
  LLVMValueRef hi16 = b.buildLShr(bits, b.constInt(b.i32Ty(), 16, false), "");
  return b.buildTrunc(hi16, b.i16Ty(), "");
}

//===----------------------------------------------------------------------===//
// Type conversion
//===----------------------------------------------------------------------===//

LLVMTypeRef MLIRToLLVM70::convertType(Type ty) {
  return llvm::TypeSwitch<Type, LLVMTypeRef>(ty)
      .Case<IntegerType>([&](auto intTy) { return b.intTy(intTy.getWidth()); })
      .Case<BFloat16Type>([&](auto) { return b.i16Ty(); })
      .Case<Float16Type>([&](auto) { return b.halfTy(); })
      .Case<Float32Type>([&](auto) { return b.floatTy(); })
      .Case<Float64Type>([&](auto) { return b.doubleTy(); })
      .Case<IndexType>([&](auto) { return b.i64Ty(); })
      .Case<LLVM::LLVMVoidType>([&](auto) { return b.voidTy(); })
      .Case<LLVM::LLVMPointerType>([&](auto ptrTy) {
        return b.ptrTy(b.i8Ty(), ptrTy.getAddressSpace());
      })
      .Case<LLVM::LLVMArrayType>([&](auto arrTy) {
        return b.arrayTy(convertType(arrTy.getElementType()),
                         arrTy.getNumElements());
      })
      .Case<LLVM::LLVMStructType>([&](auto structTy) {
        llvm::SmallVector<LLVMTypeRef> elems;
        for (Type e : structTy.getBody())
          elems.push_back(convertType(e));
        return b.structTy(elems.data(), elems.size(), structTy.isPacked());
      })
      .Case<LLVM::LLVMFunctionType>([&](auto funcTy) {
        LLVMTypeRef retTy = convertType(funcTy.getReturnType());
        llvm::SmallVector<LLVMTypeRef> params;
        for (Type p : funcTy.getParams())
          params.push_back(convertType(p));
        return b.funcTy(retTy, params.data(), params.size(),
                        funcTy.isVarArg());
      })
      .Case<VectorType>([&](auto vecTy) -> LLVMTypeRef {
        assert(!vecTy.isScalable() && "scalable vectors not supported");
        return b.vectorTy(convertType(vecTy.getElementType()),
                          vecTy.getNumElements());
      })
      .Default([&](Type t) -> LLVMTypeRef {
        std::string msg;
        llvm::raw_string_ostream os(msg);
        os << "llvm70: unsupported type: " << t;
        llvm::report_fatal_error(llvm::StringRef(msg),
                                 /*GenCrashDiag=*/false);
      });
}

//===----------------------------------------------------------------------===//
// Value lookup
//===----------------------------------------------------------------------===//

LLVMValueRef MLIRToLLVM70::lookupValue(Value v) {
  auto it = valueMap.find(v);
  if (it != valueMap.end())
    return it->second;
  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "llvm70: unmapped MLIR value: " << v
     << "\nThis is a translator bug — the value was used before being defined.";
  llvm::report_fatal_error(llvm::StringRef(msg), /*GenCrashDiag=*/false);
}

//===----------------------------------------------------------------------===//
// Debug info helpers
//===----------------------------------------------------------------------===//

std::tuple<llvm::StringRef, unsigned, unsigned>
MLIRToLLVM70::extractFileLineCol(Location loc) {
  if (auto flc = dyn_cast<FileLineColLoc>(loc))
    return {flc.getFilename(), flc.getLine(), flc.getColumn()};
  if (auto fused = dyn_cast<FusedLoc>(loc)) {
    for (auto inner : fused.getLocations()) {
      auto [f, l, c] = extractFileLineCol(inner);
      if (!f.empty())
        return {f, l, c};
    }
  }
  if (auto name = dyn_cast<NameLoc>(loc))
    return extractFileLineCol(name.getChildLoc());
  if (auto callSite = dyn_cast<CallSiteLoc>(loc))
    return extractFileLineCol(callSite.getCallee());
  return {"", 0, 0};
}

LLVMMetadataRef MLIRToLLVM70::getOrCreateDIFile(llvm::StringRef filename) {
  auto it = diFileCache.find(filename);
  if (it != diFileCache.end())
    return it->second;

  // Split filename into directory and basename.
  auto [dir, base] = filename.rsplit('/');
  if (base.empty()) {
    base = dir;
    dir = ".";
  }
  LLVMMetadataRef file =
      b.createDIFile(base.data(), base.size(), dir.data(), dir.size());
  diFileCache[filename] = file;
  return file;
}

void MLIRToLLVM70::setDebugLocFromOp(Operation *op) {
  if (!diCompileUnit)
    return;

  auto [filename, line, col] = extractFileLineCol(op->getLoc());
  if (filename.empty() || line == 0) {
    b.clearDebugLocation();
    return;
  }

  LLVMMetadataRef scope = currentSubprogram ? currentSubprogram : diCompileUnit;
  b.setDebugLocation(line, col, scope);
}

//===----------------------------------------------------------------------===//
// Top-level translation
//===----------------------------------------------------------------------===//

llvm::Error MLIRToLLVM70::translate(gpu::GPUModuleOp gpuMod, int debugLevel,
                                    bool omitDebugInfoVersionFlag,
                                    const LLVM70Options *opts) {
  bool needFullDebug = (debugLevel >= 2);
  if (debugLevel > 0) {
    // Upgrade to FullDebug when the IR contains debug variable intrinsics
    // that require DILocalVariable metadata (e.g. from the standalone tool
    // where debugLevel defaults to 1).
    if (!needFullDebug) {
      gpuMod.walk([&](Operation *op) -> WalkResult {
        if (isa<LLVM::DbgDeclareOp, LLVM::DbgValueOp>(op)) {
          needFullDebug = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
    }
    b.initDebugInfo();
    constexpr llvm::StringLiteral moduleName("llvm70_module");
    constexpr llvm::StringLiteral moduleDir(".");
    LLVMMetadataRef mainFile =
        b.createDIFile(moduleName.data(), moduleName.size(),
                       moduleDir.data(), moduleDir.size());
    diCompileUnit = b.createDICompileUnit(mainFile, needFullDebug);
    diSubroutineType = b.createDISubroutineType(mainFile);
  }

  // First pass: declare all functions and globals (forward references)
  for (auto &op : *gpuMod.getBody()) {
    if (isa<LLVM::GlobalOp>(&op)) {
      if (auto err = translateGlobalOp(&op))
        return err;
    } else if (isa<LLVM::LLVMFuncOp>(&op)) {
      auto funcOp = cast<LLVM::LLVMFuncOp>(&op);
      auto fty = funcOp.getFunctionType();
      LLVMTypeRef retTy = convertType(fty.getReturnType());
      llvm::SmallVector<LLVMTypeRef> paramTys;
      for (Type p : fty.getParams())
        paramTys.push_back(convertType(p));
      LLVMTypeRef llvmFty =
          b.funcTy(retTy, paramTys.data(), paramTys.size(), fty.isVarArg());
      b.addFunction(funcOp.getName().str().c_str(), llvmFty);
    }
  }

  // Second pass: translate function bodies
  for (auto &op : *gpuMod.getBody()) {
    if (isa<LLVM::LLVMFuncOp>(&op)) {
      if (auto err = translateFuncOp(&op))
        return err;
    }
  }

  // Emit @llvm.used so libnvvm preserves kernel entry points in LTOIR.
  {
    llvm::SmallVector<LLVMValueRef> kernelFns;
    for (auto &op : *gpuMod.getBody()) {
      if (auto funcOp = dyn_cast<LLVM::LLVMFuncOp>(&op)) {
        if (funcOp->hasAttr("gpu.kernel")) {
          LLVMValueRef fn = b.getNamedFunction(funcOp.getName().str().c_str());
          kernelFns.push_back(b.constBitCast(fn, b.ptrTy(b.i8Ty(), 0)));
        }
      }
    }
    if (!kernelFns.empty()) {
      LLVMTypeRef i8PtrTy = b.ptrTy(b.i8Ty(), 0);
      LLVMTypeRef arrTy = b.arrayTy(i8PtrTy, kernelFns.size());
      LLVMValueRef arr = b.constArray(i8PtrTy, kernelFns.data(),
                                      kernelFns.size());
      LLVMValueRef gv = b.addGlobal(arrTy, "llvm.used");
      b.setLinkage(gv, LLVMAppendingLinkage);
      b.setSection(gv, "llvm.metadata");
      b.setInitializer(gv, arr);
    }
  }

  if (diCompileUnit) {
    b.finalizeDebugInfo();
    // libnvvm accepts the lineinfo/debug metadata when producing LLVM70 LTOIR,
    // but nvJitLink cannot combine that LTOIR with NVRTC LTOIR if this module
    // flag is present. Keep the metadata and omit only this flag for LTOIR.
    if (!omitDebugInfoVersionFlag) {
      LLVMValueRef flagVals[3] = {b.constInt(b.i32Ty(), 2, false),
                                  b.mdString("Debug Info Version", 18),
                                  b.constInt(b.i32Ty(), 3, false)};
      b.addNamedMetadataOperand("llvm.module.flags", b.mdNode(flagVals, 3));
    }
  }

  int irMajor = opts ? opts->nvvmIRMajor : 2;
  int irMinor = opts ? opts->nvvmIRMinor : 0;
  int debugMajor = opts ? opts->nvvmDebugMajor : 0;
  int debugMinor = opts ? opts->nvvmDebugMinor : 0;
  LLVMValueRef verVals[4] = {
      b.constInt(b.i32Ty(), irMajor, false),
      b.constInt(b.i32Ty(), irMinor, false),
      b.constInt(b.i32Ty(), debugMajor, false),
      b.constInt(b.i32Ty(), debugMinor, false)};
  LLVMValueRef verNode = b.mdNode(verVals, 4);
  b.addNamedMetadataOperand("nvvmir.version", verNode);

  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Globals
//===----------------------------------------------------------------------===//

llvm::Error MLIRToLLVM70::translateGlobalOp(Operation *op) {
  auto globalOp = cast<LLVM::GlobalOp>(op);
  LLVMTypeRef ty = convertType(globalOp.getType());
  unsigned addrSpace = globalOp.getAddrSpace();

  LLVMValueRef gv;
  if (addrSpace != 0)
    gv = b.addGlobalInAddressSpace(ty, globalOp.getName().str().c_str(),
                                   addrSpace);
  else
    gv = b.addGlobal(ty, globalOp.getName().str().c_str());

  if (globalOp.getLinkage() == LLVM::Linkage::Internal)
    b.setLinkage(gv, LLVMInternalLinkage);
  else if (globalOp.getLinkage() == LLVM::Linkage::Private)
    b.setLinkage(gv, LLVMPrivateLinkage);
  else if (globalOp.getLinkage() == LLVM::Linkage::Common)
    b.setLinkage(gv, LLVMCommonLinkage);

  if (globalOp.getAlignment().has_value())
    b.setGlobalAlignment(gv, *globalOp.getAlignment());

  if (auto initAttr = globalOp.getValueOrNull()) {
    if (auto strAttr = dyn_cast<StringAttr>(initAttr)) {
      llvm::StringRef data = strAttr.getValue();
      llvm::SmallVector<LLVMValueRef> bytes;
      for (char c : data)
        bytes.push_back(b.constInt(b.i8Ty(), static_cast<uint8_t>(c), false));
      b.setInitializer(gv, b.constArray(b.i8Ty(), bytes.data(), bytes.size()));
    } else if (auto intAttr = dyn_cast<IntegerAttr>(initAttr)) {
      b.setInitializer(gv, b.constInt(convertType(intAttr.getType()),
                                       toUInt64(intAttr.getValue()),
                                       intAttr.getValue().isNegative()));
    } else {
      b.setInitializer(gv, b.constNull(ty));
    }
  } else if (!globalOp.getInitializerRegion().empty()) {
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "global '%s' has a body-region initializer; "
        "a constant-folding pass should have collapsed this to an attribute first",
        globalOp.getName().str().c_str());
  } else if (globalOp.getLinkage() != LLVM::Linkage::External) {
    b.setInitializer(gv, b.constNull(ty));
  }

  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

llvm::Error MLIRToLLVM70::translateFuncOp(Operation *op) {
  auto funcOp = cast<LLVM::LLVMFuncOp>(op);
  if (funcOp.isExternal())
    return llvm::Error::success();

  LLVMValueRef fn = b.getNamedFunction(funcOp.getName().str().c_str());
  if (!fn)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "function not found: %s",
                                   funcOp.getName().str().c_str());
  switchForwarders.clear();

  // Check for gpu.kernel attribute → emit !nvvm.annotations
  if (funcOp->hasAttr("gpu.kernel"))
    emitKernelMetadata(fn, op);

  // Attach debug info subprogram to this function.
  if (diCompileUnit) {
    auto [filename, line, col] = extractFileLineCol(funcOp.getLoc());
    LLVMMetadataRef diFile =
        filename.empty() ? getOrCreateDIFile("llvm70_module")
                         : getOrCreateDIFile(filename);
    std::string name = funcOp.getName().str();
    currentSubprogram =
        b.createDIFunction(diFile, name.c_str(), name.size(), diFile,
                           line ? line : 1, diSubroutineType);
    b.setSubprogram(fn, currentSubprogram);
  }

  // Create basic blocks (forward pass for branch targets)
  for (Block &block : funcOp.getBody()) {
    auto *bb = b.appendBB(fn, "");
    blockMap[&block] = bb;
  }

  // Map function arguments
  Block &entryBlock = funcOp.getBody().front();
  for (auto [i, arg] : llvm::enumerate(entryBlock.getArguments()))
    mapValue(arg, b.getParam(fn, i));

  // Translate each block
  for (Block &block : funcOp.getBody()) {
    if (auto err = translateBlock(block))
      return err;
  }

  // Wire up phi nodes (block arguments → phi incoming values)
  for (Block &block : funcOp.getBody()) {
    if (auto err = translatePhiOps(block))
      return err;
  }

  currentSubprogram = nullptr;
  b.clearDebugLocation();

  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Block translation
//===----------------------------------------------------------------------===//

llvm::Error MLIRToLLVM70::translateBlock(Block &block) {
  b.positionAtEnd(blockMap[&block]);

  // Create phi nodes for block arguments (except entry block, already mapped)
  if (!block.isEntryBlock()) {
    for (auto arg : block.getArguments()) {
      LLVMTypeRef ty = convertType(arg.getType());
      LLVMValueRef phi = b.buildPhi(ty, "");
      mapValue(arg, phi);
    }
  }

  for (Operation &op : block) {
    setDebugLocFromOp(&op);
    if (auto err = translateOp(&op))
      return err;
  }

  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Phi wiring
//===----------------------------------------------------------------------===//

llvm::Error MLIRToLLVM70::translatePhiOps(Block &block) {
  if (block.isEntryBlock())
    return llvm::Error::success();

  for (auto [argIdx, arg] : llvm::enumerate(block.getArguments())) {
    LLVMValueRef phi = lookupValue(arg);

    llvm::SmallVector<LLVMValueRef> inVals;
    llvm::SmallVector<LLVMBasicBlockRef> inBlocks;

    llvm::SmallPtrSet<Block *, 4> seen;
    for (Block *pred : block.getPredecessors()) {
      if (!seen.insert(pred).second)
        continue;
      Operation *term = pred->getTerminator();
      if (auto brOp = dyn_cast<LLVM::BrOp>(term)) {
        inVals.push_back(lookupValue(brOp.getDestOperands()[argIdx]));
        inBlocks.push_back(blockMap[pred]);
      } else if (auto condBr = dyn_cast<LLVM::CondBrOp>(term)) {
        // Skip if both branches target the same block — handled by trampolines
        if (condBr.getTrueDest() == condBr.getFalseDest())
          continue;
        if (condBr.getTrueDest() == &block) {
          inVals.push_back(lookupValue(condBr.getTrueDestOperands()[argIdx]));
          inBlocks.push_back(blockMap[pred]);
        }
        if (condBr.getFalseDest() == &block) {
          inVals.push_back(lookupValue(condBr.getFalseDestOperands()[argIdx]));
          inBlocks.push_back(blockMap[pred]);
        }
      }
    }

    auto it = switchForwarders.find(&block);
    if (it != switchForwarders.end()) {
      for (auto &[trampBB, ops] : it->second) {
        inVals.push_back(lookupValue(ops[argIdx]));
        inBlocks.push_back(trampBB);
      }
    }

    if (!inVals.empty())
      b.addIncoming(phi, inVals.data(), inBlocks.data(), inVals.size());
  }
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Op dispatch
//===----------------------------------------------------------------------===//

llvm::Error MLIRToLLVM70::translateOp(Operation *op) {
  return llvm::TypeSwitch<Operation *, llvm::Error>(op)
      // Control flow
      .Case<LLVM::ReturnOp>([&](auto) { return this->translateReturnOp(op); })
      .Case<LLVM::BrOp>([&](auto) { return this->translateBrOp(op); })
      .Case<LLVM::CondBrOp>([&](auto) { return this->translateCondBrOp(op); })
      .Case<LLVM::SwitchOp>([&](auto) { return this->translateSwitchOp(op); })
      .Case<LLVM::CallOp>([&](auto) { return this->translateCallOp(op); })
      // Constants
      .Case<LLVM::ConstantOp>(
          [&](auto) { return this->translateConstantOp(op); })
      .Case<LLVM::UndefOp>([&](auto) { return this->translateUndefOp(op); })
      .Case<LLVM::PoisonOp>([&](auto) { return this->translatePoisonOp(op); })
      .Case<LLVM::ZeroOp>([&](auto) { return this->translateZeroOp(op); })
      .Case<LLVM::AddressOfOp>(
          [&](auto) { return this->translateAddressOfOp(op); })
      // Memory
      .Case<LLVM::LoadOp>([&](auto) { return this->translateLoadOp(op); })
      .Case<LLVM::StoreOp>([&](auto) { return this->translateStoreOp(op); })
      .Case<LLVM::AllocaOp>([&](auto) { return this->translateAllocaOp(op); })
      .Case<LLVM::GEPOp>([&](auto) { return this->translateGEPOp(op); })
      // Comparison
      .Case<LLVM::ICmpOp>([&](auto) { return this->translateICmpOp(op); })
      .Case<LLVM::FCmpOp>([&](auto) { return this->translateFCmpOp(op); })
      .Case<LLVM::SelectOp>([&](auto) { return this->translateSelectOp(op); })
      // Aggregate
      .Case<LLVM::ExtractValueOp>(
          [&](auto) { return this->translateExtractValueOp(op); })
      .Case<LLVM::InsertValueOp>(
          [&](auto) { return this->translateInsertValueOp(op); })
      // Vector
      .Case<LLVM::ExtractElementOp>(
          [&](auto) { return this->translateExtractElementOp(op); })
      .Case<LLVM::InsertElementOp>(
          [&](auto) { return this->translateInsertElementOp(op); })
      // Atomics
      .Case<LLVM::AtomicRMWOp>(
          [&](auto) { return this->translateAtomicRMWOp(op); })
      .Case<LLVM::AtomicCmpXchgOp>(
          [&](auto) { return this->translateAtomicCmpXchgOp(op); })
      // Debug intrinsics
      .Case<LLVM::DbgDeclareOp>(
          [&](auto) { return this->translateDbgDeclareOp(op); })
      .Case<LLVM::DbgValueOp>(
          [&](auto) { return this->translateDbgValueOp(op); })
      .Case<LLVM::DbgLabelOp>( // No LLVM 7 *C* API for labels
          [&](auto) { return llvm::Error::success(); })
      // Inline asm
      .Case<LLVM::InlineAsmOp>(
          [&](auto) { return this->translateInlineAsmOp(op); })
      // LLVM intrinsics lowered to libdevice calls
      .Case<LLVM::FractionExpOp>(
          [&](auto) { return this->translateFrexpOp(op); })
      .Case<LLVM::LoadExpOp>(
          [&](auto) { return this->translateLdexpOp(op); })
      .Case<LLVM::CtPopOp>(
          [&](auto) { return this->translateSimpleIntIntrinsic(
                           op, "llvm.ctpop"); })
      .Case<LLVM::BitReverseOp>(
          [&](auto) { return this->translateSimpleIntIntrinsic(
                           op, "llvm.bitreverse"); })
      .Case<LLVM::FTruncOp>([&](auto) {
        return this->translateUnaryFloatIntrinsic(op, "llvm.trunc");
      })
      .Case<LLVM::SMinOp>([&](auto) {
        return this->translateBinaryIntIntrinsic(op, LLVMIntSLT);
      })
      .Case<LLVM::SMaxOp>([&](auto) {
        return this->translateBinaryIntIntrinsic(op, LLVMIntSGT);
      })
      .Case<LLVM::UMinOp>([&](auto) {
        return this->translateBinaryIntIntrinsic(op, LLVMIntULT);
      })
      .Case<LLVM::UMaxOp>([&](auto) {
        return this->translateBinaryIntIntrinsic(op, LLVMIntUGT);
      })
      .Case<LLVM::MinimumOp>([&](auto) {
        return this->translateMinimumMaximumOp(op, "llvm.minnum");
      })
      .Case<LLVM::MaximumOp>([&](auto) {
        return this->translateMinimumMaximumOp(op, "llvm.maxnum");
      })
      .Case<LLVM::MinNumOp>([&](auto) {
        return this->translateBinaryFloatIntrinsic(op, "llvm.minnum");
      })
      .Case<LLVM::MaxNumOp>([&](auto) {
        return this->translateBinaryFloatIntrinsic(op, "llvm.maxnum");
      })
      .Case<LLVM::MemsetOp>(
          [&](auto) { return this->translateMemsetOp(op); })
      .Case<LLVM::MemcpyOp>(
          [&](auto) { return this->translateMemcpyOp(op); })
      .Case<LLVM::LifetimeStartOp>(
          [&](auto) { return this->translateLifetimeOp(op, /*isStart=*/true); })
      .Case<LLVM::LifetimeEndOp>(
          [&](auto) { return this->translateLifetimeOp(op, /*isStart=*/false); })
      .Case<LLVM::AbsOp>([&](auto) {
        return this->translateAbsOp(op);
      })
      .Case<LLVM::CountTrailingZerosOp>([&](auto) {
        auto o = cast<LLVM::CountTrailingZerosOp>(op);
        return this->translateCttzCtlzOp(op, "llvm.cttz", o.getIn(),
                                         o.getRes(), o.getIsZeroPoison());
      })
      .Case<LLVM::CountLeadingZerosOp>([&](auto) {
        auto o = cast<LLVM::CountLeadingZerosOp>(op);
        return this->translateCttzCtlzOp(op, "llvm.ctlz", o.getIn(),
                                         o.getRes(), o.getIsZeroPoison());
      })
      // Arithmetic / logical
      .Case<LLVM::AddOp, LLVM::SubOp, LLVM::MulOp, LLVM::SDivOp, LLVM::UDivOp,
            LLVM::SRemOp, LLVM::URemOp, LLVM::FAddOp, LLVM::FSubOp,
            LLVM::FMulOp, LLVM::FDivOp, LLVM::FRemOp, LLVM::AndOp,
            LLVM::OrOp, LLVM::XOrOp, LLVM::ShlOp, LLVM::LShrOp,
            LLVM::AShrOp, LLVM::FNegOp>(
          [&](auto) { return this->translateArithOp(op); })
      // Casts
      .Case<LLVM::BitcastOp, LLVM::AddrSpaceCastOp, LLVM::IntToPtrOp,
            LLVM::PtrToIntOp, LLVM::TruncOp, LLVM::ZExtOp, LLVM::SExtOp,
            LLVM::FPTruncOp, LLVM::FPExtOp, LLVM::FPToSIOp, LLVM::FPToUIOp,
            LLVM::SIToFPOp, LLVM::UIToFPOp>(
          [&](auto) { return this->translateCastOp(op); })
      // Default: check NVVM dialect, arith survivors, or error
      .Default([&](Operation *o) -> llvm::Error {
        if (o->getDialect()->getNamespace() == "nvvm")
          return this->translateNVVMOp(o);
        auto opName = o->getName().getStringRef();
        if (opName == "arith.truncf" || opName == "arith.extf")
          return this->translateCastOp(o);
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(), "unsupported op: %s",
            opName.str().c_str());
      });
}

//===----------------------------------------------------------------------===//
// Individual op translators
//===----------------------------------------------------------------------===//

llvm::Error MLIRToLLVM70::translateReturnOp(Operation *op) {
  auto retOp = cast<LLVM::ReturnOp>(op);
  if (retOp.getNumOperands() == 0)
    b.buildRetVoid();
  else
    b.buildRet(lookupValue(retOp.getOperand(0)));
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateBrOp(Operation *op) {
  auto brOp = cast<LLVM::BrOp>(op);
  b.buildBr(blockMap[brOp.getDest()]);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateCondBrOp(Operation *op) {
  auto condBr = cast<LLVM::CondBrOp>(op);
  Block *trueDest = condBr.getTrueDest();
  Block *falseDest = condBr.getFalseDest();

  LLVMBasicBlockRef trueBB = blockMap[trueDest];
  LLVMBasicBlockRef falseBB = blockMap[falseDest];

  // When both branches target the same block with arguments, create
  // trampoline blocks so each edge gets a unique LLVM predecessor for PHI.
  if (trueDest == falseDest && trueDest->getNumArguments() > 0) {
    auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
    LLVMValueRef fn = b.getNamedFunction(funcOp.getName().str().c_str());
    LLVMBasicBlockRef savedBB = b.getInsertBlock();

    LLVMBasicBlockRef destBB = trueBB;
    auto mkTramp = [&](OperandRange ops) {
      LLVMBasicBlockRef t = b.appendBB(fn, "");
      b.positionAtEnd(t);
      b.buildBr(destBB);
      switchForwarders[trueDest].push_back({t, ops});
      return t;
    };

    trueBB = mkTramp(condBr.getTrueDestOperands());
    falseBB = mkTramp(condBr.getFalseDestOperands());
    b.positionAtEnd(savedBB);
    b.buildCondBr(lookupValue(condBr.getCondition()), trueBB, falseBB);
  } else {
    b.buildCondBr(lookupValue(condBr.getCondition()), trueBB, falseBB);
  }
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateSwitchOp(Operation *op) {
  auto switchOp = cast<LLVM::SwitchOp>(op);
  LLVMValueRef val = lookupValue(switchOp.getValue());

  auto caseValues = switchOp.getCaseValues();
  auto caseDestinations = switchOp.getCaseDestinations();
  unsigned numCases = caseDestinations.size();

  // Trampoline blocks for edges with block arguments.
  auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
  LLVMValueRef fn = b.getNamedFunction(funcOp.getName().str().c_str());

  auto getEdgeBB = [&](Block *dest, OperandRange ops) -> LLVMBasicBlockRef {
    if (dest->getNumArguments() == 0)
      return blockMap[dest];
    LLVMBasicBlockRef tramp = b.appendBB(fn, "");
    LLVMBasicBlockRef savedBB = b.getInsertBlock();
    b.positionAtEnd(tramp);
    b.buildBr(blockMap[dest]);
    b.positionAtEnd(savedBB);
    switchForwarders[dest].push_back({tramp, ops});
    return tramp;
  };

  LLVMBasicBlockRef defaultBB =
      getEdgeBB(switchOp.getDefaultDestination(),
                switchOp.getDefaultOperands());
  LLVMValueRef sw = b.buildSwitch(val, defaultBB, numCases);

  if (caseValues) {
    auto values = caseValues->getValues<llvm::APInt>();
    auto caseOperands = switchOp.getCaseOperands();
    unsigned idx = 0;
    for (const auto &caseVal : values) {
      LLVMValueRef onVal =
          b.constInt(convertType(switchOp.getValue().getType()),
                     toUInt64(caseVal), false);
      b.addCase(sw, onVal, getEdgeBB(caseDestinations[idx], caseOperands[idx]));
      ++idx;
    }
  }

  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateCallOp(Operation *op) {
  auto callOp = cast<LLVM::CallOp>(op);

  LLVMValueRef callee = nullptr;
  bool isIndirect = false;

  if (auto sym = callOp.getCalleeAttr()) {
    callee = b.getNamedFunction(sym.getValue().str().c_str());
    if (!callee)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "unresolved callee in llvm.call: %s",
                                     sym.getValue().str().c_str());
  } else {
    // Indirect call: first callee_operand is the function pointer.
    auto allOperands = callOp.getCalleeOperands();
    if (allOperands.empty())
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "indirect llvm.call with no callee operand");
    callee = lookupValue(allOperands[0]);
    isIndirect = true;
  }

  // Bitcast pointer arguments to match the callee's declared parameter types.
  // All MLIR !llvm.ptr become i8* in declarations, but downstream ops (alloca,
  // GEP) may have produced pointers of different types (e.g. i32*, {..}*).
  llvm::SmallVector<LLVMValueRef> args;
  for (auto [idx, mlirArg] : llvm::enumerate(callOp.getArgOperands())) {
    LLVMValueRef v = lookupValue(mlirArg);
    if (auto ptrTy = dyn_cast<LLVM::LLVMPointerType>(mlirArg.getType())) {
      LLVMTypeRef expectedTy = b.ptrTy(b.i8Ty(), ptrTy.getAddressSpace());
      v = b.buildBitCast(v, expectedTy, "");
    }
    args.push_back(v);
  }

  if (isIndirect) {
    // For indirect calls, bitcast the callee to the expected function pointer
    // type so LLVMBuildCall can infer the signature.
    llvm::SmallVector<LLVMTypeRef> paramTys;
    for (auto mlirArg : callOp.getArgOperands())
      paramTys.push_back(convertType(mlirArg.getType()));

    LLVMTypeRef retTy = (callOp.getNumResults() > 0)
                             ? convertType(callOp->getResult(0).getType())
                             : b.voidTy();
    LLVMTypeRef fnTy = b.funcTy(retTy, paramTys.data(), paramTys.size(), false);
    LLVMTypeRef fnPtrTy = b.ptrTy(fnTy, 0);
    callee = b.buildBitCast(callee, fnPtrTy, "");
  }

  LLVMValueRef result = b.buildCall(callee, args.data(), args.size(), "");

  if (callOp.getNumResults() > 0)
    mapValue(callOp->getResult(0), result);

  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateConstantOp(Operation *op) {
  auto constOp = cast<LLVM::ConstantOp>(op);
  Value result = constOp.getResult();
  Attribute value = constOp.getValue();

  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    LLVMTypeRef ty = convertType(result.getType());
    mapValue(result, b.constInt(ty, toUInt64(intAttr.getValue()),
                                intAttr.getValue().isNegative()));
    return llvm::Error::success();
  }

  if (auto floatAttr = dyn_cast<FloatAttr>(value)) {
    mapValue(result, isBF16(result.getType())
        ? constBF16(b, floatAttr.getValue())
        : b.constReal(convertType(result.getType()),
                      floatAttr.getValueAsDouble()));
    return llvm::Error::success();
  }

  if (auto arrayAttr = dyn_cast<ArrayAttr>(value)) {
    SmallVector<LLVMValueRef> elems;
    for (Attribute elemAttr : arrayAttr) {
      if (auto intAttr = dyn_cast<IntegerAttr>(elemAttr)) {
        LLVMTypeRef ty = convertType(intAttr.getType());
        elems.push_back(b.constInt(ty, toUInt64(intAttr.getValue()),
                                   intAttr.getValue().isNegative()));
      } else if (auto floatAttr = dyn_cast<FloatAttr>(elemAttr)) {
        elems.push_back(isBF16(floatAttr.getType())
            ? constBF16(b, floatAttr.getValue())
            : b.constReal(convertType(floatAttr.getType()),
                          floatAttr.getValueAsDouble()));
      } else {
        std::string msg;
        llvm::raw_string_ostream os(msg);
        os << "unsupported element in constant array attribute: ";
        elemAttr.print(os);
        return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                       msg.c_str());
      }
    }
    bool packed = false;
    if (auto structTy = dyn_cast<LLVM::LLVMStructType>(result.getType()))
      packed = structTy.isPacked();
    mapValue(result,
             b.constStruct(elems.data(), elems.size(), packed));
    return llvm::Error::success();
  }

  if (auto denseAttr = dyn_cast<DenseElementsAttr>(value)) {
    auto vecTy = dyn_cast<VectorType>(result.getType());
    if (!vecTy)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "DenseElementsAttr on non-vector type");
    Type elemTy = vecTy.getElementType();
    LLVMTypeRef elemLLVMTy = convertType(elemTy);
    SmallVector<LLVMValueRef> elems;
    if (elemTy.isIntOrIndex()) {
      for (auto val : denseAttr.getValues<APInt>())
        elems.push_back(b.constInt(elemLLVMTy, toUInt64(val),
                                   val.isNegative()));
    } else {
      // Reconstruct floating-point elements from their raw bit patterns instead
      // of using DenseElementsAttr::getValues<APFloat>(). When this bridge and
      // the MLIR Python bindings embed distinct LLVM copies (e.g. on Windows,
      // where duplicate symbols are not interposed as they are with ELF),
      // APFloat's fltSemantics singletons live at different addresses across the
      // module boundary and the float element iterator silently yields zero.
      // Raw byte access carries the correct IEEE encoding, so we read the bits
      // and materialize the constant via an integer bitcast. bf16 is stored as
      // i16, so it needs no bitcast.
      unsigned bitWidth = elemTy.getIntOrFloatBitWidth();
      unsigned byteWidth = bitWidth / 8;
      ArrayRef<char> raw = denseAttr.getRawData();
      bool splat = denseAttr.isSplat();
      LLVMTypeRef intElemTy = b.intTy(bitWidth);
      for (int64_t i = 0, e = vecTy.getNumElements(); i < e; ++i) {
        uint64_t bits = 0;
        std::memcpy(&bits, raw.data() + (splat ? 0 : i * byteWidth), byteWidth);
        LLVMValueRef intConst = b.constInt(intElemTy, bits, /*signExt=*/false);
        elems.push_back(isBF16(elemTy)
                            ? intConst
                            : b.constBitCast(intConst, elemLLVMTy));
      }
    }
    mapValue(result, b.constVector(elems.data(), elems.size()));
    return llvm::Error::success();
  }

  std::string msg;
  llvm::raw_string_ostream os(msg);
  os << "unsupported constant attribute: ";
  value.print(os);
  os << " of type ";
  result.getType().print(os);
  return llvm::createStringError(llvm::inconvertibleErrorCode(), msg.c_str());
}

llvm::Error MLIRToLLVM70::translateUndefOp(Operation *op) {
  auto undefOp = cast<LLVM::UndefOp>(op);
  mapValue(undefOp.getResult(), b.getUndef(convertType(undefOp.getType())));
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateZeroOp(Operation *op) {
  auto zeroOp = cast<LLVM::ZeroOp>(op);
  mapValue(zeroOp.getResult(), b.constNull(convertType(zeroOp.getType())));
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translatePoisonOp(Operation *op) {
  auto poisonOp = cast<LLVM::PoisonOp>(op);
  // LLVM 7 has no poison values; undef is the closest equivalent.
  mapValue(poisonOp.getResult(), b.getUndef(convertType(poisonOp.getType())));
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateAddressOfOp(Operation *op) {
  auto addrOf = cast<LLVM::AddressOfOp>(op);
  std::string name = addrOf.getGlobalName().str();
  LLVMValueRef gv = b.getNamedGlobal(name.c_str());
  if (!gv)
    gv = b.getNamedFunction(name.c_str());
  if (!gv)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "addressof: '%s' not found", name.c_str());
  if (auto ptrTy =
          dyn_cast<LLVM::LLVMPointerType>(addrOf.getResult().getType()))
    gv = b.constBitCast(gv, b.ptrTy(b.i8Ty(), ptrTy.getAddressSpace()));
  mapValue(addrOf.getResult(), gv);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateArithOp(Operation *op) {
  LLVMValueRef result = nullptr;

  // Unary
  if (auto fneg = dyn_cast<LLVM::FNegOp>(op)) {
    LLVMValueRef operand = lookupValue(fneg.getOperand());
    bool bf = isBF16(fneg.getOperand().getType());
    if (bf)
      operand = bf16ToF32(operand);
    result = b.buildFNeg(operand, "");
    if (bf)
      result = f32ToBf16(result);
    mapValue(fneg.getResult(), result);
    return llvm::Error::success();
  }

  // Binary ops — operand(0) and operand(1)
  LLVMValueRef lhs = lookupValue(op->getOperand(0));
  LLVMValueRef rhs = lookupValue(op->getOperand(1));

  bool bf = isBF16(op->getOperand(0).getType());
  if (bf) {
    lhs = bf16ToF32(lhs);
    rhs = bf16ToF32(rhs);
  }

  if (isa<LLVM::AddOp>(op))
    result = b.buildAdd(lhs, rhs, "");
  else if (isa<LLVM::SubOp>(op))
    result = b.buildSub(lhs, rhs, "");
  else if (isa<LLVM::MulOp>(op))
    result = b.buildMul(lhs, rhs, "");
  else if (isa<LLVM::SDivOp>(op))
    result = b.buildSDiv(lhs, rhs, "");
  else if (isa<LLVM::UDivOp>(op))
    result = b.buildUDiv(lhs, rhs, "");
  else if (isa<LLVM::SRemOp>(op))
    result = b.buildSRem(lhs, rhs, "");
  else if (isa<LLVM::URemOp>(op))
    result = b.buildURem(lhs, rhs, "");
  else if (isa<LLVM::FAddOp>(op))
    result = b.buildFAdd(lhs, rhs, "");
  else if (isa<LLVM::FSubOp>(op))
    result = b.buildFSub(lhs, rhs, "");
  else if (isa<LLVM::FMulOp>(op))
    result = b.buildFMul(lhs, rhs, "");
  else if (isa<LLVM::FDivOp>(op))
    result = b.buildFDiv(lhs, rhs, "");
  else if (isa<LLVM::FRemOp>(op))
    result = b.buildFRem(lhs, rhs, "");
  else if (isa<LLVM::AndOp>(op))
    result = b.buildAnd(lhs, rhs, "");
  else if (isa<LLVM::OrOp>(op))
    result = b.buildOr(lhs, rhs, "");
  else if (isa<LLVM::XOrOp>(op))
    result = b.buildXor(lhs, rhs, "");
  else if (isa<LLVM::ShlOp>(op))
    result = b.buildShl(lhs, rhs, "");
  else if (isa<LLVM::LShrOp>(op))
    result = b.buildLShr(lhs, rhs, "");
  else if (isa<LLVM::AShrOp>(op))
    result = b.buildAShr(lhs, rhs, "");
  else
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unhandled arith op: %s",
                                   op->getName().getStringRef().str().c_str());

  if (bf)
    result = f32ToBf16(result);

  mapValue(op->getResult(0), result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateICmpOp(Operation *op) {
  auto icmpOp = cast<LLVM::ICmpOp>(op);
  auto pred = icmpOp.getPredicate();

  LLVMIntPredicate lp = LLVMIntEQ;
  switch (pred) {
  case LLVM::ICmpPredicate::eq:
    lp = LLVMIntEQ;
    break;
  case LLVM::ICmpPredicate::ne:
    lp = LLVMIntNE;
    break;
  case LLVM::ICmpPredicate::ugt:
    lp = LLVMIntUGT;
    break;
  case LLVM::ICmpPredicate::uge:
    lp = LLVMIntUGE;
    break;
  case LLVM::ICmpPredicate::ult:
    lp = LLVMIntULT;
    break;
  case LLVM::ICmpPredicate::ule:
    lp = LLVMIntULE;
    break;
  case LLVM::ICmpPredicate::sgt:
    lp = LLVMIntSGT;
    break;
  case LLVM::ICmpPredicate::sge:
    lp = LLVMIntSGE;
    break;
  case LLVM::ICmpPredicate::slt:
    lp = LLVMIntSLT;
    break;
  case LLVM::ICmpPredicate::sle:
    lp = LLVMIntSLE;
    break;
  }

  auto result = b.buildICmp(lp, lookupValue(icmpOp.getLhs()),
                            lookupValue(icmpOp.getRhs()), "");
  mapValue(icmpOp.getResult(), result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateFCmpOp(Operation *op) {
  auto fcmpOp = cast<LLVM::FCmpOp>(op);
  auto pred = fcmpOp.getPredicate();

  LLVMRealPredicate lp = LLVMRealPredicateFalse;
  switch (pred) {
  case LLVM::FCmpPredicate::_false:
    lp = LLVMRealPredicateFalse;
    break;
  case LLVM::FCmpPredicate::oeq:
    lp = LLVMRealOEQ;
    break;
  case LLVM::FCmpPredicate::ogt:
    lp = LLVMRealOGT;
    break;
  case LLVM::FCmpPredicate::oge:
    lp = LLVMRealOGE;
    break;
  case LLVM::FCmpPredicate::olt:
    lp = LLVMRealOLT;
    break;
  case LLVM::FCmpPredicate::ole:
    lp = LLVMRealOLE;
    break;
  case LLVM::FCmpPredicate::one:
    lp = LLVMRealONE;
    break;
  case LLVM::FCmpPredicate::ord:
    lp = LLVMRealORD;
    break;
  case LLVM::FCmpPredicate::uno:
    lp = LLVMRealUNO;
    break;
  case LLVM::FCmpPredicate::ueq:
    lp = LLVMRealUEQ;
    break;
  case LLVM::FCmpPredicate::ugt:
    lp = LLVMRealUGT;
    break;
  case LLVM::FCmpPredicate::uge:
    lp = LLVMRealUGE;
    break;
  case LLVM::FCmpPredicate::ult:
    lp = LLVMRealULT;
    break;
  case LLVM::FCmpPredicate::ule:
    lp = LLVMRealULE;
    break;
  case LLVM::FCmpPredicate::une:
    lp = LLVMRealUNE;
    break;
  case LLVM::FCmpPredicate::_true:
    lp = LLVMRealPredicateTrue;
    break;
  }

  LLVMValueRef lhsVal = lookupValue(fcmpOp.getLhs());
  LLVMValueRef rhsVal = lookupValue(fcmpOp.getRhs());
  if (isBF16(fcmpOp.getLhs().getType())) {
    lhsVal = bf16ToF32(lhsVal);
    rhsVal = bf16ToF32(rhsVal);
  }
  auto result = b.buildFCmp(lp, lhsVal, rhsVal, "");
  mapValue(fcmpOp.getResult(), result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateLoadOp(Operation *op) {
  auto loadOp = cast<LLVM::LoadOp>(op);
  LLVMValueRef ptr = lookupValue(loadOp.getAddr());

  // Reconstruct pointer type: bitcast i8* → elemTy* so the LLVM 7
  // load infers the correct result type.
  LLVMTypeRef elemTy = convertType(loadOp.getResult().getType());
  unsigned as =
      cast<LLVM::LLVMPointerType>(loadOp.getAddr().getType()).getAddressSpace();
  ptr = b.buildBitCast(ptr, b.ptrTy(elemTy, as), "");

  LLVMValueRef val = b.buildLoad(ptr, "");
  mapValue(loadOp.getResult(), val);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateStoreOp(Operation *op) {
  auto storeOp = cast<LLVM::StoreOp>(op);
  LLVMValueRef val = lookupValue(storeOp.getValue());
  LLVMValueRef ptr = lookupValue(storeOp.getAddr());

  // Reconstruct pointer type: bitcast i8* → elemTy* to match stored value
  // type.
  LLVMTypeRef elemTy = convertType(storeOp.getValue().getType());
  unsigned as = cast<LLVM::LLVMPointerType>(storeOp.getAddr().getType())
                    .getAddressSpace();
  ptr = b.buildBitCast(ptr, b.ptrTy(elemTy, as), "");

  b.buildStore(val, ptr);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateAllocaOp(Operation *op) {
  auto allocaOp = cast<LLVM::AllocaOp>(op);
  LLVMTypeRef elemTy = convertType(allocaOp.getElemType());
  LLVMValueRef sizeVal = lookupValue(allocaOp.getArraySize());
  LLVMValueRef val = b.buildArrayAlloca(elemTy, sizeVal, "");
  if (auto align = allocaOp.getAlignment())
    b.setGlobalAlignment(val, *align);
  mapValue(allocaOp.getResult(), val);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateGEPOp(Operation *op) {
  auto gepOp = cast<LLVM::GEPOp>(op);
  LLVMValueRef base = lookupValue(gepOp.getBase());

  // Reconstruct pointer type: bitcast base from i8* → elemTy* so that the
  // old LLVM GEP computes offsets with the correct element size.
  LLVMTypeRef elemTy = convertType(gepOp.getElemType());
  unsigned as =
      cast<LLVM::LLVMPointerType>(gepOp.getBase().getType()).getAddressSpace();
  base = b.buildBitCast(base, b.ptrTy(elemTy, as), "");

  llvm::SmallVector<LLVMValueRef> indices;
  auto rawConstantIndices = gepOp.getRawConstantIndices();
  auto dynamicIndices = gepOp.getDynamicIndices();
  unsigned dynIdx = 0;
  for (int32_t raw : rawConstantIndices) {
    if (raw == LLVM::GEPOp::kDynamicIndex)
      indices.push_back(lookupValue(dynamicIndices[dynIdx++]));
    else
      indices.push_back(b.constInt(b.i32Ty(), raw, /*signExt=*/true));
  }

  bool isInbounds = static_cast<uint32_t>(gepOp.getNoWrapFlags()) &
                    static_cast<uint32_t>(LLVM::GEPNoWrapFlags::inboundsFlag);
  LLVMValueRef result;
  if (isInbounds)
    result = b.buildInBoundsGEP(base, indices.data(), indices.size(), "");
  else
    result = b.buildGEP(base, indices.data(), indices.size(), "");

  mapValue(gepOp.getResult(), result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateCastOp(Operation *op) {
  LLVMValueRef src = lookupValue(op->getOperand(0));
  Type srcMlir = op->getOperand(0).getType();
  Type dstMlir = op->getResult(0).getType();
  bool srcBF = isBF16(srcMlir), dstBF = isBF16(dstMlir);
  LLVMTypeRef destTy = convertType(dstMlir);
  LLVMValueRef result = nullptr;

  if (isa<LLVM::BitcastOp>(op))
    result = (srcBF || dstBF) ? src : b.buildBitCast(src, destTy, "");
  else if (isa<LLVM::AddrSpaceCastOp>(op))
    result = b.buildAddrSpaceCast(src, destTy, "");
  else if (isa<LLVM::IntToPtrOp>(op))
    result = b.buildIntToPtr(src, destTy, "");
  else if (isa<LLVM::PtrToIntOp>(op))
    result = b.buildPtrToInt(src, destTy, "");
  else if (isa<LLVM::TruncOp>(op))
    result = b.buildTrunc(src, destTy, "");
  else if (isa<LLVM::ZExtOp>(op))
    result = b.buildZExt(src, destTy, "");
  else if (isa<LLVM::SExtOp>(op))
    result = b.buildSExt(src, destTy, "");
  else if (isa<LLVM::FPTruncOp>(op)) {
    if (dstBF)
      result = f32ToBf16(srcMlir.isF64() ? b.buildFPTrunc(src, b.floatTy(), "")
                                         : src);
    else
      result = b.buildFPTrunc(srcBF ? bf16ToF32(src) : src, destTy, "");
  } else if (isa<LLVM::FPExtOp>(op)) {
    if (srcBF) {
      LLVMValueRef f32 = bf16ToF32(src);
      result = dstMlir.isF64() ? b.buildFPExt(f32, destTy, "") : f32;
    } else
      result = dstBF ? f32ToBf16(b.buildFPExt(src, b.floatTy(), ""))
                     : b.buildFPExt(src, destTy, "");
  } else if (isa<LLVM::FPToSIOp>(op))
    result = b.buildFPToSI(srcBF ? bf16ToF32(src) : src, destTy, "");
  else if (isa<LLVM::FPToUIOp>(op))
    result = b.buildFPToUI(srcBF ? bf16ToF32(src) : src, destTy, "");
  else if (isa<LLVM::SIToFPOp>(op))
    result = dstBF ? f32ToBf16(b.buildSIToFP(src, b.floatTy(), ""))
                   : b.buildSIToFP(src, destTy, "");
  else if (isa<LLVM::UIToFPOp>(op))
    result = dstBF ? f32ToBf16(b.buildUIToFP(src, b.floatTy(), ""))
                   : b.buildUIToFP(src, destTy, "");

  if (result) {
    mapValue(op->getResult(0), result);
    return llvm::Error::success();
  }
  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unhandled cast: %s",
                                 op->getName().getStringRef().str().c_str());
}

llvm::Error MLIRToLLVM70::translateSelectOp(Operation *op) {
  auto selectOp = cast<LLVM::SelectOp>(op);
  auto result = b.buildSelect(lookupValue(selectOp.getCondition()),
                              lookupValue(selectOp.getTrueValue()),
                              lookupValue(selectOp.getFalseValue()), "");
  mapValue(selectOp.getResult(), result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateExtractValueOp(Operation *op) {
  auto evOp = cast<LLVM::ExtractValueOp>(op);
  LLVMValueRef agg = lookupValue(evOp.getContainer());
  for (int64_t idx : evOp.getPosition())
    agg = b.buildExtractValue(agg, static_cast<unsigned>(idx), "");
  mapValue(evOp.getResult(), agg);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateInsertValueOp(Operation *op) {
  auto ivOp = cast<LLVM::InsertValueOp>(op);
  LLVMValueRef agg = lookupValue(ivOp.getContainer());
  LLVMValueRef val = lookupValue(ivOp.getValue());
  auto positions = ivOp.getPosition();
  if (positions.size() == 1) {
    auto result = b.buildInsertValue(agg, val, positions[0], "");
    mapValue(ivOp.getResult(), result);
  } else {
    // Multi-level: extract the nested aggregate, insert into it, then
    // insert it back into the parent at each level.
    llvm::SmallVector<LLVMValueRef> chain;
    chain.push_back(agg);
    for (size_t i = 0; i + 1 < positions.size(); ++i)
      chain.push_back(b.buildExtractValue(chain.back(), positions[i], ""));
    LLVMValueRef inner =
        b.buildInsertValue(chain.back(), val, positions.back(), "");
    for (int i = static_cast<int>(positions.size()) - 2; i >= 0; --i)
      inner = b.buildInsertValue(chain[i], inner, positions[i], "");
    mapValue(ivOp.getResult(), inner);
  }
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateExtractElementOp(Operation *op) {
  auto eeOp = cast<LLVM::ExtractElementOp>(op);
  LLVMValueRef vec = lookupValue(eeOp.getVector());
  LLVMValueRef idx = lookupValue(eeOp.getPosition());
  mapValue(eeOp.getResult(), b.buildExtractElement(vec, idx, ""));
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateInsertElementOp(Operation *op) {
  auto ieOp = cast<LLVM::InsertElementOp>(op);
  LLVMValueRef vec = lookupValue(ieOp.getVector());
  LLVMValueRef val = lookupValue(ieOp.getValue());
  LLVMValueRef idx = lookupValue(ieOp.getPosition());
  mapValue(ieOp.getResult(), b.buildInsertElement(vec, val, idx, ""));
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Inline asm
//===----------------------------------------------------------------------===//

llvm::Error MLIRToLLVM70::translateInlineAsmOp(Operation *op) {
  auto asmOp = cast<LLVM::InlineAsmOp>(op);

  // Build the function type: (operand types...) -> result type
  LLVMTypeRef retTy;
  if (asmOp.getNumResults() == 0)
    retTy = b.voidTy();
  else
    retTy = convertType(asmOp->getResult(0).getType());

  llvm::SmallVector<LLVMTypeRef> paramTys;
  for (Value operand : asmOp.getOperands())
    paramTys.push_back(convertType(operand.getType()));

  LLVMTypeRef fnTy =
      b.funcTy(retTy, paramTys.data(), paramTys.size(), /*varArg=*/false);

  // Create the inline asm value (LLVM 7 API: LLVMConstInlineAsm).
  LLVMValueRef asmVal =
      b.constInlineAsm(fnTy, asmOp.getAsmString().str().c_str(),
                       asmOp.getConstraints().str().c_str(),
                       asmOp.getHasSideEffects(), asmOp.getIsAlignStack());

  // Collect operands, bitcasting pointers to match the function type
  // (convertType maps all MLIR !llvm.ptr to i8*, but GEP results are typed).
  llvm::SmallVector<LLVMValueRef> args;
  for (auto [i, operand] : llvm::enumerate(asmOp.getOperands())) {
    LLVMValueRef val = lookupValue(operand);
    if (isa<LLVM::LLVMPointerType>(operand.getType()))
      val = b.buildBitCast(val, paramTys[i], "");
    args.push_back(val);
  }

  LLVMValueRef result = b.buildCall(asmVal, args.data(), args.size(), "");

  if (asmOp.getNumResults() > 0)
    mapValue(asmOp->getResult(0), result);

  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateFrexpOp(Operation *op) {
  auto frexpOp = cast<LLVM::FractionExpOp>(op);
  Type valTy = frexpOp.getVal().getType();

  // llvm.frexp doesn't exist in LLVM 7. Lower to libdevice __nv_frexp[f].
  // Signature: double __nv_frexp(double, i32*)  /  float __nv_frexpf(float, i32*)
  const char *funcName;
  if (valTy.isF64())
    funcName = "__nv_frexp";
  else if (valTy.isF32())
    funcName = "__nv_frexpf";
  else
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported frexp type");

  LLVMTypeRef llvmValTy = convertType(valTy);
  LLVMTypeRef i32PtrTy = b.ptrTy(b.i32Ty(), 0);
  LLVMTypeRef paramTys[2] = {llvmValTy, i32PtrTy};
  LLVMTypeRef fnTy = b.funcTy(llvmValTy, paramTys, 2, false);

  LLVMValueRef fn = b.getNamedFunction(funcName);
  if (!fn)
    fn = b.addFunction(funcName, fnTy);

  // Alloca for the exponent output.
  LLVMValueRef expAlloca = b.buildAlloca(b.i32Ty(), "");
  LLVMValueRef args[2] = {lookupValue(frexpOp.getVal()), expAlloca};
  LLVMValueRef mantissa = b.buildCall(fn, args, 2, "");
  LLVMValueRef exponent = b.buildLoad(expAlloca, "");

  // Build the result struct {valTy, i32}.
  LLVMTypeRef elemTys[2] = {llvmValTy, b.i32Ty()};
  LLVMTypeRef structTy = b.structTy(elemTys, 2, false);
  LLVMValueRef result = b.getUndef(structTy);
  result = b.buildInsertValue(result, mantissa, 0, "");
  result = b.buildInsertValue(result, exponent, 1, "");
  mapValue(frexpOp.getRes(), result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateLdexpOp(Operation *op) {
  auto ldexpOp = cast<LLVM::LoadExpOp>(op);
  Type valTy = ldexpOp.getVal().getType();

  // llvm.ldexp doesn't exist in LLVM 7. Lower to libdevice __nv_ldexp[f].
  // Signature: double __nv_ldexp(double, i32)  /  float __nv_ldexpf(float, i32)
  const char *funcName;
  if (valTy.isF64())
    funcName = "__nv_ldexp";
  else if (valTy.isF32())
    funcName = "__nv_ldexpf";
  else
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported ldexp type");

  LLVMTypeRef llvmValTy = convertType(valTy);
  LLVMTypeRef paramTys[2] = {llvmValTy, b.i32Ty()};
  LLVMTypeRef fnTy = b.funcTy(llvmValTy, paramTys, 2, false);

  LLVMValueRef fn = b.getNamedFunction(funcName);
  if (!fn)
    fn = b.addFunction(funcName, fnTy);

  LLVMValueRef args[2] = {lookupValue(ldexpOp.getVal()),
                           lookupValue(ldexpOp.getPower())};
  LLVMValueRef result = b.buildCall(fn, args, 2, "");
  mapValue(ldexpOp.getRes(), result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateSimpleIntIntrinsic(Operation *op,
                                                      llvm::StringRef intrBase) {
  Value in = op->getOperand(0);
  Value res = op->getResult(0);
  Type valTy = in.getType();
  LLVMTypeRef llvmTy = convertType(valTy);

  unsigned bitWidth = cast<IntegerType>(valTy).getWidth();
  std::string intrName = (intrBase + ".i" + llvm::Twine(bitWidth)).str();

  LLVMTypeRef fnTy = b.funcTy(llvmTy, &llvmTy, 1, false);
  LLVMValueRef fn = b.getNamedFunction(intrName.c_str());
  if (!fn)
    fn = b.addFunction(intrName.c_str(), fnTy);

  LLVMValueRef arg = lookupValue(in);
  LLVMValueRef result = b.buildCall(fn, &arg, 1, "");
  mapValue(res, result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateUnaryFloatIntrinsic(Operation *op,
                                                       llvm::StringRef intrBase) {
  Value in = op->getOperand(0);
  Value res = op->getResult(0);
  Type valTy = in.getType();
  bool bf = isBF16(valTy);
  const char *suffix = floatSuffix(valTy);
  if (!suffix)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported float type for intrinsic");

  LLVMTypeRef llvmTy = bf ? b.floatTy() : convertType(valTy);
  std::string intrName = (intrBase + "." + suffix).str();
  LLVMTypeRef fnTy = b.funcTy(llvmTy, &llvmTy, 1, false);
  LLVMValueRef fn = b.getNamedFunction(intrName.c_str());
  if (!fn)
    fn = b.addFunction(intrName.c_str(), fnTy);

  LLVMValueRef arg = lookupValue(in);
  if (bf) arg = bf16ToF32(arg);
  LLVMValueRef result = b.buildCall(fn, &arg, 1, "");
  if (bf) result = f32ToBf16(result);
  mapValue(res, result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateBinaryIntIntrinsic(Operation *op,
                                                      LLVMIntPredicate pred) {
  LLVMValueRef l = lookupValue(op->getOperand(0));
  LLVMValueRef r = lookupValue(op->getOperand(1));
  LLVMValueRef cmp = b.buildICmp(pred, l, r, "");
  mapValue(op->getResult(0), b.buildSelect(cmp, l, r, ""));
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateAbsOp(Operation *op) {
  // llvm.abs doesn't exist in LLVM 7; lower to: select(x < 0, -x, x)
  auto absOp = cast<LLVM::AbsOp>(op);
  LLVMValueRef x = lookupValue(absOp.getIn());
  LLVMTypeRef ty = convertType(absOp.getRes().getType());
  LLVMValueRef zero = b.constInt(ty, 0, false);
  LLVMValueRef neg = b.buildSub(zero, x, "");
  LLVMValueRef cmp = b.buildICmp(LLVMIntSLT, x, zero, "");
  mapValue(absOp.getRes(), b.buildSelect(cmp, neg, x, ""));
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateBinaryFloatIntrinsic(Operation *op,
                                                        llvm::StringRef intrBase) {
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  Value res = op->getResult(0);
  Type valTy = lhs.getType();
  bool bf = isBF16(valTy);
  const char *suffix = floatSuffix(valTy);
  if (!suffix)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported float type for intrinsic");

  LLVMTypeRef llvmTy = bf ? b.floatTy() : convertType(valTy);
  std::string intrName = (intrBase + "." + suffix).str();
  LLVMTypeRef paramTys[2] = {llvmTy, llvmTy};
  LLVMTypeRef fnTy = b.funcTy(llvmTy, paramTys, 2, false);
  LLVMValueRef fn = b.getNamedFunction(intrName.c_str());
  if (!fn)
    fn = b.addFunction(intrName.c_str(), fnTy);

  LLVMValueRef args[2] = {lookupValue(lhs), lookupValue(rhs)};
  if (bf) { args[0] = bf16ToF32(args[0]); args[1] = bf16ToF32(args[1]); }
  LLVMValueRef result = b.buildCall(fn, args, 2, "");
  if (bf) result = f32ToBf16(result);
  mapValue(res, result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateMinimumMaximumOp(Operation *op,
                                                    llvm::StringRef minnumMaxnum) {
  Value lhs = op->getOperand(0);
  Value rhs = op->getOperand(1);
  Type valTy = lhs.getType();
  bool bf = isBF16(valTy);
  const char *suffix = floatSuffix(valTy);
  if (!suffix)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported float type for minimum/maximum");

  LLVMTypeRef llvmTy = bf ? b.floatTy() : convertType(valTy);
  std::string intrName = (minnumMaxnum + "." + suffix).str();
  LLVMTypeRef paramTys[2] = {llvmTy, llvmTy};
  LLVMTypeRef fnTy = b.funcTy(llvmTy, paramTys, 2, false);
  LLVMValueRef fn = b.getNamedFunction(intrName.c_str());
  if (!fn)
    fn = b.addFunction(intrName.c_str(), fnTy);

  LLVMValueRef a = lookupValue(lhs);
  LLVMValueRef bv = lookupValue(rhs);
  if (bf) { a = bf16ToF32(a); bv = bf16ToF32(bv); }
  LLVMValueRef args[2] = {a, bv};
  LLVMValueRef mmnResult = b.buildCall(fn, args, 2, "");
  LLVMValueRef isNaN = b.buildFCmp(LLVMRealUNO, a, bv, "");
  LLVMValueRef nanVal = b.buildFAdd(a, bv, "");
  LLVMValueRef result = b.buildSelect(isNaN, nanVal, mmnResult, "");
  if (bf) result = f32ToBf16(result);
  mapValue(op->getResult(0), result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateMemsetOp(Operation *op) {
  auto memsetOp = cast<LLVM::MemsetOp>(op);

  // llvm.memset.p<as>i8.i64(i8*, i8, i64, i1 isvolatile) -> void
  unsigned as =
      cast<LLVM::LLVMPointerType>(memsetOp.getDst().getType()).getAddressSpace();
  std::string intrName =
      "llvm.memset.p" + std::to_string(as) + "i8.i64";

  LLVMTypeRef ptrTy = b.ptrTy(b.i8Ty(), as);
  LLVMTypeRef paramTys[4] = {ptrTy, b.i8Ty(), b.i64Ty(), b.i1Ty()};
  LLVMTypeRef fnTy = b.funcTy(b.voidTy(), paramTys, 4, false);
  LLVMValueRef fn = b.getNamedFunction(intrName.c_str());
  if (!fn)
    fn = b.addFunction(intrName.c_str(), fnTy);

  LLVMValueRef dst = lookupValue(memsetOp.getDst());
  dst = b.buildBitCast(dst, ptrTy, "");

  LLVMValueRef args[4] = {
      dst, lookupValue(memsetOp.getVal()), lookupValue(memsetOp.getLen()),
      b.constInt(b.i1Ty(), memsetOp.getIsVolatile() ? 1 : 0, false)};
  b.buildCall(fn, args, 4, "");
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateMemcpyOp(Operation *op) {
  auto memcpyOp = cast<LLVM::MemcpyOp>(op);

  unsigned dstAS =
      cast<LLVM::LLVMPointerType>(memcpyOp.getDst().getType()).getAddressSpace();
  unsigned srcAS =
      cast<LLVM::LLVMPointerType>(memcpyOp.getSrc().getType()).getAddressSpace();

  // llvm.memcpy.p<dstAS>i8.p<srcAS>i8.i64(i8*, i8*, i64, i1) -> void
  std::string intrName = "llvm.memcpy.p" + std::to_string(dstAS) + "i8.p" +
                          std::to_string(srcAS) + "i8.i64";

  LLVMTypeRef dstPtrTy = b.ptrTy(b.i8Ty(), dstAS);
  LLVMTypeRef srcPtrTy = b.ptrTy(b.i8Ty(), srcAS);
  LLVMTypeRef paramTys[4] = {dstPtrTy, srcPtrTy, b.i64Ty(), b.i1Ty()};
  LLVMTypeRef fnTy = b.funcTy(b.voidTy(), paramTys, 4, false);
  LLVMValueRef fn = b.getNamedFunction(intrName.c_str());
  if (!fn)
    fn = b.addFunction(intrName.c_str(), fnTy);

  LLVMValueRef dst = lookupValue(memcpyOp.getDst());
  dst = b.buildBitCast(dst, dstPtrTy, "");
  LLVMValueRef src = lookupValue(memcpyOp.getSrc());
  src = b.buildBitCast(src, srcPtrTy, "");

  LLVMValueRef args[4] = {
      dst, src, lookupValue(memcpyOp.getLen()),
      b.constInt(b.i1Ty(), memcpyOp.getIsVolatile() ? 1 : 0, false)};
  b.buildCall(fn, args, 4, "");
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateLifetimeOp(Operation *op, bool isStart) {
  Value ptr = op->getOperand(0);
  unsigned as =
      cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace();
  std::string intrName = std::string("llvm.lifetime.") +
                          (isStart ? "start" : "end") + ".p" +
                          std::to_string(as) + "i8";

  LLVMTypeRef ptrTy = b.ptrTy(b.i8Ty(), as);
  LLVMTypeRef paramTys[2] = {b.i64Ty(), ptrTy};
  LLVMTypeRef fnTy = b.funcTy(b.voidTy(), paramTys, 2, false);
  LLVMValueRef fn = b.getNamedFunction(intrName.c_str());
  if (!fn)
    fn = b.addFunction(intrName.c_str(), fnTy);

  LLVMValueRef ptrVal = lookupValue(ptr);
  ptrVal = b.buildBitCast(ptrVal, ptrTy, "");

  // Size = -1 means unknown.
  LLVMValueRef args[2] = {b.constInt(b.i64Ty(), (uint64_t)-1, true), ptrVal};
  b.buildCall(fn, args, 2, "");
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateClusterArriveOp(Operation *op,
                                                   bool isRelaxed) {
  bool aligned = false;
  if (auto attr = op->getAttrOfType<UnitAttr>("aligned"))
    aligned = true;

  std::string ptx = "barrier.cluster.arrive";
  if (isRelaxed)
    ptx += ".relaxed";
  if (aligned)
    ptx += ".aligned";
  ptx += ";";

  LLVMTypeRef fnTy = b.funcTy(b.voidTy(), nullptr, 0, false);
  LLVMValueRef asmVal =
      b.constInlineAsm(fnTy, ptx.c_str(), "", true, false);
  b.buildCall(asmVal, nullptr, 0, "");
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateClusterWaitOp(Operation *op) {
  bool aligned = false;
  if (op->getAttrOfType<UnitAttr>("aligned"))
    aligned = true;

  std::string ptx = "barrier.cluster.wait";
  if (aligned)
    ptx += ".aligned";
  ptx += ";";

  LLVMTypeRef fnTy = b.funcTy(b.voidTy(), nullptr, 0, false);
  LLVMValueRef asmVal =
      b.constInlineAsm(fnTy, ptx.c_str(), "", true, false);
  b.buildCall(asmVal, nullptr, 0, "");
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateBarrierOp(Operation *op) {
  auto barrierOp = cast<NVVM::BarrierOp>(op);
  Value barrierId = barrierOp.getBarrierId();
  Value numThreads = barrierOp.getNumberOfThreads();
  Value predicate = barrierOp.getReductionPredicate();
  auto reductionOp = barrierOp.getReductionOp();

  if (reductionOp) {
    if (!predicate)
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "barrier reduction requires a predicate");

    llvm::StringRef opStr;
    switch (*reductionOp) {
    case NVVM::BarrierReduction::POPC: opStr = "popc"; break;
    case NVVM::BarrierReduction::AND:  opStr = "and";  break;
    case NVVM::BarrierReduction::OR:   opStr = "or";   break;
    }

    std::string intrName = ("llvm.nvvm.barrier0." + opStr).str();
    LLVMTypeRef paramTy = b.i32Ty();
    LLVMTypeRef fnTy = b.funcTy(b.i32Ty(), &paramTy, 1, false);
    LLVMValueRef fn = b.getNamedFunction(intrName.c_str());
    if (!fn)
      fn = b.addFunction(intrName.c_str(), fnTy);
    LLVMValueRef predVal = lookupValue(predicate);
    LLVMValueRef result = b.buildCall(fn, &predVal, 1, "");
    mapValue(barrierOp.getRes(), result);
  } else if (numThreads) {
    // bar.sync barId, numThreads;
    std::string ptx = "bar.sync $0, $1;";
    LLVMTypeRef paramTys[2] = {b.i32Ty(), b.i32Ty()};
    LLVMTypeRef fnTy = b.funcTy(b.voidTy(), paramTys, 2, false);
    LLVMValueRef asmVal =
        b.constInlineAsm(fnTy, ptx.c_str(), "r,r", true, false);
    LLVMValueRef args[2] = {lookupValue(barrierId), lookupValue(numThreads)};
    b.buildCall(asmVal, args, 2, "");
  } else if (barrierId) {
    // bar.sync barId;
    std::string ptx = "bar.sync $0;";
    LLVMTypeRef paramTy = b.i32Ty();
    LLVMTypeRef fnTy = b.funcTy(b.voidTy(), &paramTy, 1, false);
    LLVMValueRef asmVal =
        b.constInlineAsm(fnTy, ptx.c_str(), "r", true, false);
    LLVMValueRef arg = lookupValue(barrierId);
    b.buildCall(asmVal, &arg, 1, "");
  } else {
    // bar.sync 0; (equivalent to barrier0)
    LLVMTypeRef fnTy = b.funcTy(b.voidTy(), nullptr, 0, false);
    LLVMValueRef asmVal =
        b.constInlineAsm(fnTy, "bar.sync 0;", "", true, false);
    b.buildCall(asmVal, nullptr, 0, "");
  }

  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateCttzCtlzOp(Operation *op,
                                              llvm::StringRef intrBase,
                                              Value in, Value res,
                                              bool isZeroPoison) {
  Type valTy = in.getType();
  LLVMTypeRef llvmTy = convertType(valTy);

  unsigned bitWidth = cast<IntegerType>(valTy).getWidth();
  std::string intrName =
      (intrBase + ".i" + llvm::Twine(bitWidth)).str();

  // llvm.ct{tz,lz}.iN takes (iN, i1 is_zero_poison) -> iN
  LLVMTypeRef paramTys[2] = {llvmTy, b.i1Ty()};
  LLVMTypeRef fnTy = b.funcTy(llvmTy, paramTys, 2, false);
  LLVMValueRef fn = b.getNamedFunction(intrName.c_str());
  if (!fn)
    fn = b.addFunction(intrName.c_str(), fnTy);

  LLVMValueRef args[2] = {lookupValue(in),
                           b.constInt(b.i1Ty(), isZeroPoison ? 1 : 0, false)};
  LLVMValueRef result = b.buildCall(fn, args, 2, "");
  mapValue(res, result);
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Atomics
//===----------------------------------------------------------------------===//

static LLVMAtomicOrdering convertAtomicOrdering(LLVM::AtomicOrdering ordering) {
  switch (ordering) {
  case LLVM::AtomicOrdering::not_atomic:
    return LLVMAtomicOrderingNotAtomic;
  case LLVM::AtomicOrdering::unordered:
    return LLVMAtomicOrderingUnordered;
  case LLVM::AtomicOrdering::monotonic:
    return LLVMAtomicOrderingMonotonic;
  case LLVM::AtomicOrdering::acquire:
    return LLVMAtomicOrderingAcquire;
  case LLVM::AtomicOrdering::release:
    return LLVMAtomicOrderingRelease;
  case LLVM::AtomicOrdering::acq_rel:
    return LLVMAtomicOrderingAcquireRelease;
  case LLVM::AtomicOrdering::seq_cst:
    return LLVMAtomicOrderingSequentiallyConsistent;
  }
  return LLVMAtomicOrderingMonotonic;
}

static llvm::Expected<LLVMAtomicRMWBinOp>
convertAtomicBinOp(LLVM::AtomicBinOp op) {
  switch (op) {
  case LLVM::AtomicBinOp::xchg:
    return LLVMAtomicRMWBinOpXchg;
  case LLVM::AtomicBinOp::add:
    return LLVMAtomicRMWBinOpAdd;
  case LLVM::AtomicBinOp::sub:
    return LLVMAtomicRMWBinOpSub;
  case LLVM::AtomicBinOp::_and:
    return LLVMAtomicRMWBinOpAnd;
  case LLVM::AtomicBinOp::nand:
    return LLVMAtomicRMWBinOpNand;
  case LLVM::AtomicBinOp::_or:
    return LLVMAtomicRMWBinOpOr;
  case LLVM::AtomicBinOp::_xor:
    return LLVMAtomicRMWBinOpXor;
  case LLVM::AtomicBinOp::max:
    return LLVMAtomicRMWBinOpMax;
  case LLVM::AtomicBinOp::min:
    return LLVMAtomicRMWBinOpMin;
  case LLVM::AtomicBinOp::umax:
    return LLVMAtomicRMWBinOpUMax;
  case LLVM::AtomicBinOp::umin:
    return LLVMAtomicRMWBinOpUMin;
  default:
    return llvm::createStringError(
        llvm::inconvertibleErrorCode(),
        "unsupported atomicrmw bin_op for LLVM 7: %s",
        LLVM::stringifyAtomicBinOp(op).str().c_str());
  }
}

llvm::Error MLIRToLLVM70::translateFloatAtomicCASLoop(Operation *op) {
  auto rmwOp = cast<LLVM::AtomicRMWOp>(op);
  auto binOp = rmwOp.getBinOp();
  Type mlirValTy = rmwOp.getVal().getType();

  LLVMTypeRef floatTy = convertType(mlirValTy);
  LLVMTypeRef intTy;
  if (mlirValTy.isF32())
    intTy = b.i32Ty();
  else if (mlirValTy.isF64())
    intTy = b.i64Ty();
  else
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported type for float atomicrmw");

  // All variants use maxnum/minnum (NaN-ignoring) semantics in the CAS loop.
  // Hardware atomicrmw fmaximum/fminimum also lowers to atom.max/atom.min
  // (maxnum/minnum), so we match that behavior here.
  LLVMRealPredicate cmpPred;
  switch (binOp) {
  case LLVM::AtomicBinOp::fmax:
  case LLVM::AtomicBinOp::fmaximum:
    cmpPred = LLVMRealOLT;
    break;
  case LLVM::AtomicBinOp::fmin:
  case LLVM::AtomicBinOp::fminimum:
    cmpPred = LLVMRealOGT;
    break;
  default:
    llvm_unreachable("unexpected binop for CAS float atomicrmw");
  }

  auto funcOp = op->getParentOfType<LLVM::LLVMFuncOp>();
  LLVMValueRef fn = b.getNamedFunction(funcOp.getName().str().c_str());

  LLVMValueRef ptr = lookupValue(rmwOp.getPtr());
  LLVMValueRef val = lookupValue(rmwOp.getVal());

  unsigned as =
      cast<LLVM::LLVMPointerType>(rmwOp.getPtr().getType()).getAddressSpace();
  LLVMValueRef typedPtr = b.buildBitCast(ptr, b.ptrTy(floatTy, as), "");

  LLVMBasicBlockRef entryBB = b.getInsertBlock();
  LLVMBasicBlockRef loopBB = b.appendBB(fn, "cas.loop");
  LLVMBasicBlockRef attemptBB = b.appendBB(fn, "cas.attempt");
  LLVMBasicBlockRef doneBB = b.appendBB(fn, "cas.done");

  // --- entry (current block) ---
  LLVMValueRef ptrval = b.buildLoad(typedPtr, "");

  // Early-exit when val is NaN: maxnum(old, NaN) = old — no change needed.
  LLVMValueRef valIsNaN = b.buildFCmp(LLVMRealUNO, val, val, "");
  b.buildCondBr(valIsNaN, doneBB, loopBB);

  // --- loop ---
  b.positionAtEnd(loopBB);
  LLVMValueRef dold = b.buildPhi(floatTy, "");
  LLVMValueRef cmp = b.buildFCmp(cmpPred, dold, val, "");

  // Swap if (dold < val) OR (dold is NaN).
  // When *ptr is NaN, maxnum(NaN, val) = val — must swap val in.
  LLVMValueRef doldIsNaN = b.buildFCmp(LLVMRealUNO, dold, dold, "");
  LLVMValueRef shouldSwap = b.buildOr(cmp, doldIsNaN, "");
  b.buildCondBr(shouldSwap, attemptBB, doneBB);

  // --- attempt (CAS) ---
  b.positionAtEnd(attemptBB);
  LLVMValueRef iptr = b.buildBitCast(typedPtr, b.ptrTy(intTy, as), "");
  LLVMValueRef oldInt = b.buildBitCast(dold, intTy, "");
  LLVMValueRef newInt = b.buildBitCast(val, intTy, "");
  LLVMAtomicOrdering successOrd = convertAtomicOrdering(rmwOp.getOrdering());
  LLVMAtomicOrdering failureOrd = successOrd;
  if (failureOrd == LLVMAtomicOrderingRelease)
    failureOrd = LLVMAtomicOrderingMonotonic;
  else if (failureOrd == LLVMAtomicOrderingAcquireRelease)
    failureOrd = LLVMAtomicOrderingAcquire;
  LLVMValueRef pair = b.buildAtomicCmpXchg(iptr, oldInt, newInt, successOrd,
                                            failureOrd,
                                            /*singleThread=*/false);
  LLVMValueRef casInt = b.buildExtractValue(pair, 0, "");
  LLVMValueRef casOk = b.buildExtractValue(pair, 1, "");
  LLVMValueRef dcas = b.buildBitCast(casInt, floatTy, "");
  b.buildCondBr(casOk, doneBB, loopBB);

  // Wire the phi: dold comes from [ptrval, entry] or [dcas, attempt(fail)]
  LLVMValueRef phiVals[2] = {ptrval, dcas};
  LLVMBasicBlockRef phiBlocks[2] = {entryBB, attemptBB};
  b.addIncoming(dold, phiVals, phiBlocks, 2);

  // --- done ---
  // The result of atomicrmw is the old value in memory.
  // Predecessors: entryBB (NaN early exit), loopBB (no swap), attemptBB (CAS ok).
  b.positionAtEnd(doneBB);
  LLVMValueRef resultPhi = b.buildPhi(floatTy, "");
  LLVMValueRef resultVals[3] = {ptrval, dold, dold};
  LLVMBasicBlockRef resultBlocks[3] = {entryBB, loopBB, attemptBB};
  b.addIncoming(resultPhi, resultVals, resultBlocks, 3);
  mapValue(rmwOp.getResult(), resultPhi);

  // Update blockMap: subsequent ops (including the block terminator) will be
  // emitted in doneBB, so phi wiring in successors must reference doneBB.
  blockMap[op->getBlock()] = doneBB;

  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateAtomicRMWOp(Operation *op) {
  auto rmwOp = cast<LLVM::AtomicRMWOp>(op);

  auto binOp = rmwOp.getBinOp();

  // Float min/max don't exist in the LLVM 7 C API and have limited native PTX
  // support (f64 min/max is SM 90+ only, fminimum/fmaximum have no PTX atomic).
  // Emit a compare-and-swap loop following numba-cuda's pattern.
  if (binOp == LLVM::AtomicBinOp::fmax || binOp == LLVM::AtomicBinOp::fmin ||
      binOp == LLVM::AtomicBinOp::fmaximum ||
      binOp == LLVM::AtomicBinOp::fminimum)
    return translateFloatAtomicCASLoop(op);

  // Float add: use llvm.nvvm.atomic.load.add.f{32,64} intrinsic.
  if (binOp == LLVM::AtomicBinOp::fadd) {
    Type valTy = rmwOp.getVal().getType();
    unsigned as =
        cast<LLVM::LLVMPointerType>(rmwOp.getPtr().getType()).getAddressSpace();

    const char *tySuffix = nullptr;
    LLVMTypeRef floatTy = nullptr;
    if (valTy.isF32()) {
      tySuffix = "f32";
      floatTy = b.floatTy();
    } else if (valTy.isF64()) {
      tySuffix = "f64";
      floatTy = b.doubleTy();
    } else {
      return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                     "unsupported type for float atomicrmw");
    }

    // llvm.nvvm.atomic.load.add.f32.p1f32(f32 addrspace(1)*, f32) -> f32
    std::string intrName = llvm::formatv(
        "llvm.nvvm.atomic.load.add.{0}.p{1}{0}", tySuffix, as);
    LLVMTypeRef ptrTy = b.ptrTy(floatTy, as);
    LLVMTypeRef paramTys[2] = {ptrTy, floatTy};
    LLVMTypeRef fnTy = b.funcTy(floatTy, paramTys, 2, false);
    LLVMValueRef fn = b.getNamedFunction(intrName.c_str());
    if (!fn)
      fn = b.addFunction(intrName.c_str(), fnTy);

    LLVMValueRef ptr = lookupValue(rmwOp.getPtr());
    ptr = b.buildBitCast(ptr, ptrTy, "");
    LLVMValueRef val = lookupValue(rmwOp.getVal());
    LLVMValueRef args[2] = {ptr, val};
    LLVMValueRef result = b.buildCall(fn, args, 2, "");
    mapValue(rmwOp.getResult(), result);
    return llvm::Error::success();
  }

  auto binOpOrErr = convertAtomicBinOp(rmwOp.getBinOp());
  if (!binOpOrErr)
    return binOpOrErr.takeError();

  LLVMValueRef ptr = lookupValue(rmwOp.getPtr());
  LLVMValueRef val = lookupValue(rmwOp.getVal());

  // Bitcast the opaque ptr to a pointer matching the value type.
  LLVMTypeRef valTy = convertType(rmwOp.getVal().getType());
  unsigned as =
      cast<LLVM::LLVMPointerType>(rmwOp.getPtr().getType()).getAddressSpace();
  ptr = b.buildBitCast(ptr, b.ptrTy(valTy, as), "");

  LLVMAtomicOrdering ordering = convertAtomicOrdering(rmwOp.getOrdering());
  LLVMValueRef result =
      b.buildAtomicRMW(*binOpOrErr, ptr, val, ordering, /*singleThread=*/false);

  mapValue(rmwOp.getResult(), result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateAtomicCmpXchgOp(Operation *op) {
  auto cmpxchgOp = cast<LLVM::AtomicCmpXchgOp>(op);

  LLVMValueRef ptr = lookupValue(cmpxchgOp.getPtr());
  LLVMValueRef cmp = lookupValue(cmpxchgOp.getCmp());
  LLVMValueRef newVal = lookupValue(cmpxchgOp.getVal());

  // Bitcast to the expected pointer type.
  LLVMTypeRef cmpTy = convertType(cmpxchgOp.getCmp().getType());
  unsigned as = cast<LLVM::LLVMPointerType>(cmpxchgOp.getPtr().getType())
                    .getAddressSpace();
  ptr = b.buildBitCast(ptr, b.ptrTy(cmpTy, as), "");

  LLVMAtomicOrdering successOrd =
      convertAtomicOrdering(cmpxchgOp.getSuccessOrdering());
  LLVMAtomicOrdering failureOrd =
      convertAtomicOrdering(cmpxchgOp.getFailureOrdering());

  LLVMValueRef result = b.buildAtomicCmpXchg(
      ptr, cmp, newVal, successOrd, failureOrd, /*singleThread=*/false);

  // cmpxchg in LLVM returns {T, i1}. Map the aggregate result directly.
  mapValue(cmpxchgOp.getResult(), result);
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// NVVM dialect ops → inline PTX
//===----------------------------------------------------------------------===//

llvm::Error MLIRToLLVM70::translateSregOp(Operation *op) {
  // Lower all special-register reads to inline PTX directly, bypassing
  // libnvvm intrinsics which aren't reliably recognized across versions.
  using SregMap =
      std::pair<llvm::function_ref<bool(Operation *)>, const char *>;
  const SregMap sregOps[] = {
      {[](Operation *o) { return isa<NVVM::ThreadIdXOp>(o); },
       "mov.u32 $0, %tid.x;"},
      {[](Operation *o) { return isa<NVVM::ThreadIdYOp>(o); },
       "mov.u32 $0, %tid.y;"},
      {[](Operation *o) { return isa<NVVM::ThreadIdZOp>(o); },
       "mov.u32 $0, %tid.z;"},
      {[](Operation *o) { return isa<NVVM::BlockDimXOp>(o); },
       "mov.u32 $0, %ntid.x;"},
      {[](Operation *o) { return isa<NVVM::BlockDimYOp>(o); },
       "mov.u32 $0, %ntid.y;"},
      {[](Operation *o) { return isa<NVVM::BlockDimZOp>(o); },
       "mov.u32 $0, %ntid.z;"},
      {[](Operation *o) { return isa<NVVM::BlockIdXOp>(o); },
       "mov.u32 $0, %ctaid.x;"},
      {[](Operation *o) { return isa<NVVM::BlockIdYOp>(o); },
       "mov.u32 $0, %ctaid.y;"},
      {[](Operation *o) { return isa<NVVM::BlockIdZOp>(o); },
       "mov.u32 $0, %ctaid.z;"},
      {[](Operation *o) { return isa<NVVM::GridDimXOp>(o); },
       "mov.u32 $0, %nctaid.x;"},
      {[](Operation *o) { return isa<NVVM::GridDimYOp>(o); },
       "mov.u32 $0, %nctaid.y;"},
      {[](Operation *o) { return isa<NVVM::GridDimZOp>(o); },
       "mov.u32 $0, %nctaid.z;"},
      {[](Operation *o) { return isa<NVVM::WarpIdOp>(o); },
       "mov.u32 $0, %warpid;"},
      {[](Operation *o) { return isa<NVVM::LaneIdOp>(o); },
       "mov.u32 $0, %laneid;"},
      {[](Operation *o) { return isa<NVVM::WarpSizeOp>(o); },
       "mov.u32 $0, WARP_SZ;"},
      {[](Operation *o) { return isa<NVVM::SmIdOp>(o); },
       "mov.u32 $0, %smid;"},
      {[](Operation *o) { return isa<NVVM::WarpDimOp>(o); },
       "mov.u32 $0, %nwarpid;"},
      {[](Operation *o) { return isa<NVVM::ClusterIdXOp>(o); },
       "mov.u32 $0, %clusterid.x;"},
      {[](Operation *o) { return isa<NVVM::ClusterIdYOp>(o); },
       "mov.u32 $0, %clusterid.y;"},
      {[](Operation *o) { return isa<NVVM::ClusterIdZOp>(o); },
       "mov.u32 $0, %clusterid.z;"},
      {[](Operation *o) { return isa<NVVM::ClusterDimXOp>(o); },
       "mov.u32 $0, %nclusterid.x;"},
      {[](Operation *o) { return isa<NVVM::ClusterDimYOp>(o); },
       "mov.u32 $0, %nclusterid.y;"},
      {[](Operation *o) { return isa<NVVM::ClusterDimZOp>(o); },
       "mov.u32 $0, %nclusterid.z;"},
      {[](Operation *o) { return isa<NVVM::ClusterDimBlocksXOp>(o); },
       "mov.u32 $0, %cluster_nctaid.x;"},
      {[](Operation *o) { return isa<NVVM::ClusterDimBlocksYOp>(o); },
       "mov.u32 $0, %cluster_nctaid.y;"},
      {[](Operation *o) { return isa<NVVM::ClusterDimBlocksZOp>(o); },
       "mov.u32 $0, %cluster_nctaid.z;"},
  };

  for (const auto &[pred, asmStr] : sregOps) {
    if (pred(op)) {
      LLVMTypeRef fnTy = b.funcTy(b.i32Ty(), nullptr, 0, false);
      LLVMValueRef asmVal = b.constInlineAsm(fnTy, asmStr, "=r", false, false);
      LLVMValueRef result = b.buildCall(asmVal, nullptr, 0, "");
      mapValue(op->getResult(0), result);
      return llvm::Error::success();
    }
  }

  return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                 "unsupported sreg op: %s",
                                 op->getName().getStringRef().str().c_str());
}

llvm::Error MLIRToLLVM70::translateFmaOp(Operation *op) {
  auto fmaOp = cast<NVVM::FmaOp>(op);
  Type valTy = fmaOp.getA().getType();
  bool bf = isBF16(valTy);
  const char *suffix = floatSuffix(valTy);
  if (!suffix)
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "unsupported fma type");

  LLVMTypeRef llvmTy = bf ? b.floatTy() : convertType(valTy);
  std::string intrName = ("llvm.fma." + llvm::StringRef(suffix)).str();
  LLVMTypeRef paramTys[3] = {llvmTy, llvmTy, llvmTy};
  LLVMTypeRef fnTy = b.funcTy(llvmTy, paramTys, 3, false);

  LLVMValueRef fn = b.getNamedFunction(intrName.c_str());
  if (!fn)
    fn = b.addFunction(intrName.c_str(), fnTy);

  LLVMValueRef args[3] = {lookupValue(fmaOp.getA()), lookupValue(fmaOp.getB()),
                          lookupValue(fmaOp.getC())};
  if (bf) { args[0] = bf16ToF32(args[0]); args[1] = bf16ToF32(args[1]); args[2] = bf16ToF32(args[2]); }
  LLVMValueRef result = b.buildCall(fn, args, 3, "");
  if (bf) result = f32ToBf16(result);
  mapValue(fmaOp.getRes(), result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateElectSyncOp(Operation *op) {
  auto electOp = cast<NVVM::ElectSyncOp>(op);

  // Use inline PTX: elect.sync returns a predicate.
  // We use i32 output (0 or 1) to avoid bitcode incompatibility, then
  // truncate to i1.
  LLVMTypeRef paramTy = b.i32Ty();
  LLVMTypeRef fnTy = b.funcTy(b.i32Ty(), &paramTy, 1, false);
  LLVMValueRef asmVal = b.constInlineAsm(
      fnTy,
      "{\n"
      "  .reg .pred %p;\n"
      "  elect.sync _|%p, $1;\n"
      "  selp.u32 $0, 1, 0, %p;\n"
      "}",
      "=r,r", true, false);

  LLVMValueRef mask;
  if (electOp.getMembermask())
    mask = lookupValue(electOp.getMembermask());
  else
    mask = b.constInt(b.i32Ty(), 0xFFFFFFFF, false);

  LLVMValueRef i32Result = b.buildCall(asmVal, &mask, 1, "");
  LLVMValueRef pred = b.buildTrunc(i32Result, b.i1Ty(), "");
  mapValue(electOp.getPred(), pred);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateMatchSyncOp(Operation *op) {
  auto matchOp = cast<NVVM::MatchSyncOp>(op);
  Type valTy = matchOp.getVal().getType();
  bool is64 = valTy.isInteger(64);
  llvm::StringRef tySuffix = is64 ? "i64" : "i32";

  bool isAll = matchOp.getKind() == NVVM::MatchSyncKind::all;
  llvm::StringRef kindStr = isAll ? "all" : "any";

  std::string intrName =
      ("llvm.nvvm.match." + kindStr + ".sync." + tySuffix).str();

  LLVMTypeRef valLLTy = convertType(valTy);
  LLVMTypeRef paramTys[2] = {b.i32Ty(), valLLTy};
  LLVMTypeRef retTy;
  if (isAll) {
    LLVMTypeRef elems[2] = {b.i32Ty(), b.i1Ty()};
    retTy = b.structTy(elems, 2, false);
  } else {
    retTy = b.i32Ty();
  }
  LLVMTypeRef fnTy = b.funcTy(retTy, paramTys, 2, false);

  LLVMValueRef fn = b.getNamedFunction(intrName.c_str());
  if (!fn)
    fn = b.addFunction(intrName.c_str(), fnTy);

  LLVMValueRef args[2] = {lookupValue(matchOp.getThreadMask()),
                          lookupValue(matchOp.getVal())};
  LLVMValueRef result = b.buildCall(fn, args, 2, "");
  mapValue(matchOp.getRes(), result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateNanosleepOp(Operation *op) {
  // The llvm.nvvm.nanosleep intrinsic doesn't exist in LLVM 7, so emit
  // inline PTX assembly instead.
  LLVMTypeRef paramTy = b.i32Ty();
  LLVMTypeRef fnTy = b.funcTy(b.voidTy(), &paramTy, 1, false);
  LLVMValueRef asmVal =
      b.constInlineAsm(fnTy, "nanosleep.u32 $0;", "r", true, false);
  LLVMValueRef arg = lookupValue(op->getOperand(0));
  b.buildCall(asmVal, &arg, 1, "");
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateReduxSyncOp(Operation *op) {
  auto reduxOp = cast<NVVM::ReduxOp>(op);
  Type valTy = reduxOp.getVal().getType();
  bool isFloat = valTy.isF32();

  llvm::StringRef opStr;
  llvm::StringRef typeStr;
  switch (reduxOp.getKind()) {
  case NVVM::ReductionKind::ADD:  opStr = "add"; typeStr = "s32"; break;
  case NVVM::ReductionKind::AND:  opStr = "and"; typeStr = "b32"; break;
  case NVVM::ReductionKind::MAX:  opStr = "max"; typeStr = "s32"; break;
  case NVVM::ReductionKind::MIN:  opStr = "min"; typeStr = "s32"; break;
  case NVVM::ReductionKind::OR:   opStr = "or";  typeStr = "b32"; break;
  case NVVM::ReductionKind::UMAX: opStr = "max"; typeStr = "u32"; break;
  case NVVM::ReductionKind::UMIN: opStr = "min"; typeStr = "u32"; break;
  case NVVM::ReductionKind::XOR:  opStr = "xor"; typeStr = "b32"; break;
  case NVVM::ReductionKind::FMIN: opStr = "min"; typeStr = "f32"; break;
  case NVVM::ReductionKind::FMAX: opStr = "max"; typeStr = "f32"; break;
  }

  // Build PTX: redux.sync.<op>[.abs][.NaN].<type> $0, $1, $2;
  std::string ptx = "redux.sync." + opStr.str();
  if (isFloat && reduxOp.getAbs())
    ptx += ".abs";
  if (isFloat && reduxOp.getNan())
    ptx += ".NaN";
  ptx += "." + typeStr.str() + " $0, $1, $2;";

  LLVMTypeRef llvmValTy = convertType(valTy);
  LLVMTypeRef paramTys[2] = {llvmValTy, b.i32Ty()};
  LLVMTypeRef fnTy = b.funcTy(llvmValTy, paramTys, 2, false);

  // Constraints: =r/=f for output, r/f for val, r for mask.
  std::string constraints = isFloat ? "=f,f,r" : "=r,r,r";
  LLVMValueRef asmVal =
      b.constInlineAsm(fnTy, ptx.c_str(), constraints.c_str(), true, false);

  LLVMValueRef args[2] = {lookupValue(reduxOp.getVal()),
                           lookupValue(reduxOp.getMaskAndClamp())};
  LLVMValueRef result = b.buildCall(asmVal, args, 2, "");
  mapValue(reduxOp.getRes(), result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateMembarOp(Operation *op) {
  auto membarOp = cast<NVVM::MembarOp>(op);
  const char *intrName = nullptr;
  switch (membarOp.getScope()) {
  case NVVM::MemScopeKind::CTA:     intrName = "llvm.nvvm.membar.cta"; break;
  case NVVM::MemScopeKind::GPU:     intrName = "llvm.nvvm.membar.gl"; break;
  case NVVM::MemScopeKind::SYS:     intrName = "llvm.nvvm.membar.sys"; break;
  case NVVM::MemScopeKind::CLUSTER:
    return llvm::createStringError(llvm::inconvertibleErrorCode(),
                                   "membar with cluster scope not supported");
  }
  LLVMTypeRef fnTy = b.funcTy(b.voidTy(), nullptr, 0, false);
  LLVMValueRef fn = b.getNamedFunction(intrName);
  if (!fn)
    fn = b.addFunction(intrName, fnTy);
  b.buildCall(fn, nullptr, 0, "");
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateSyncWarpOp(Operation *op) {
  LLVMTypeRef paramTy = b.i32Ty();
  LLVMTypeRef fnTy = b.funcTy(b.voidTy(), &paramTy, 1, false);
  LLVMValueRef fn = b.getNamedFunction("llvm.nvvm.bar.warp.sync");
  if (!fn)
    fn = b.addFunction("llvm.nvvm.bar.warp.sync", fnTy);
  LLVMValueRef arg = lookupValue(op->getOperand(0));
  b.buildCall(fn, &arg, 1, "");
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateVoteSyncOp(Operation *op) {
  auto voteOp = cast<NVVM::VoteSyncOp>(op);
  llvm::StringRef kindStr;
  switch (voteOp.getKind()) {
  case NVVM::VoteSyncKind::any:
    kindStr = "any";
    break;
  case NVVM::VoteSyncKind::all:
    kindStr = "all";
    break;
  case NVVM::VoteSyncKind::ballot:
    kindStr = "ballot";
    break;
  case NVVM::VoteSyncKind::uni:
    kindStr = "uni";
    break;
  }

  std::string intrName = ("llvm.nvvm.vote." + kindStr + ".sync").str();

  LLVMTypeRef retTy = convertType(voteOp.getRes().getType());
  LLVMTypeRef paramTys[2] = {b.i32Ty(), b.i1Ty()};
  LLVMTypeRef fnTy = b.funcTy(retTy, paramTys, 2, false);

  LLVMValueRef fn = b.getNamedFunction(intrName.c_str());
  if (!fn)
    fn = b.addFunction(intrName.c_str(), fnTy);

  LLVMValueRef args[2] = {lookupValue(voteOp.getMask()),
                          lookupValue(voteOp.getPred())};
  LLVMValueRef result = b.buildCall(fn, args, 2, "");
  mapValue(voteOp.getRes(), result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateShflOp(Operation *op) {
  auto shflOp = cast<NVVM::ShflOp>(op);
  llvm::StringRef kindStr;
  switch (shflOp.getKind()) {
  case NVVM::ShflKind::bfly:
    kindStr = "bfly";
    break;
  case NVVM::ShflKind::up:
    kindStr = "up";
    break;
  case NVVM::ShflKind::down:
    kindStr = "down";
    break;
  case NVVM::ShflKind::idx:
    kindStr = "idx";
    break;
  }

  Type valTy = shflOp.getVal().getType();
  llvm::StringRef tySuffix = valTy.isF32() ? "f32" : "i32";

  bool returnPred = shflOp.getReturnValueAndIsValid().value_or(false);

  std::string intrName = ("llvm.nvvm.shfl.sync." + kindStr + "." + tySuffix +
                          (returnPred ? "p" : ""))
                             .str();

  LLVMTypeRef llvmValTy = convertType(valTy);
  LLVMTypeRef paramTys[4] = {b.i32Ty(), llvmValTy, b.i32Ty(), b.i32Ty()};
  LLVMTypeRef retTy;
  if (returnPred) {
    LLVMTypeRef elems[2] = {llvmValTy, b.i1Ty()};
    retTy = b.structTy(elems, 2, /*packed=*/false);
  } else {
    retTy = llvmValTy;
  }
  LLVMTypeRef fnTy = b.funcTy(retTy, paramTys, 4, false);

  LLVMValueRef fn = b.getNamedFunction(intrName.c_str());
  if (!fn)
    fn = b.addFunction(intrName.c_str(), fnTy);

  LLVMValueRef args[4] = {
      lookupValue(shflOp.getThreadMask()), lookupValue(shflOp.getVal()),
      lookupValue(shflOp.getOffset()), lookupValue(shflOp.getMaskAndClamp())};
  LLVMValueRef result = b.buildCall(fn, args, 4, "");
  mapValue(shflOp->getResult(0), result);
  return llvm::Error::success();
}

llvm::Error MLIRToLLVM70::translateNVVMOp(Operation *op) {
  return llvm::TypeSwitch<Operation *, llvm::Error>(op)
      .Case<NVVM::ThreadIdXOp, NVVM::ThreadIdYOp, NVVM::ThreadIdZOp,
            NVVM::BlockDimXOp, NVVM::BlockDimYOp, NVVM::BlockDimZOp,
            NVVM::BlockIdXOp, NVVM::BlockIdYOp, NVVM::BlockIdZOp,
            NVVM::GridDimXOp, NVVM::GridDimYOp, NVVM::GridDimZOp,
            NVVM::WarpIdOp, NVVM::WarpDimOp, NVVM::LaneIdOp, NVVM::WarpSizeOp,
            NVVM::SmIdOp,
            NVVM::ClusterIdXOp, NVVM::ClusterIdYOp, NVVM::ClusterIdZOp,
            NVVM::ClusterDimXOp, NVVM::ClusterDimYOp, NVVM::ClusterDimZOp,
            NVVM::ClusterDimBlocksXOp, NVVM::ClusterDimBlocksYOp,
            NVVM::ClusterDimBlocksZOp>(
          [&](auto) { return translateSregOp(op); })
      .Case<NVVM::ElectSyncOp>([&](auto) { return translateElectSyncOp(op); })
      .Case<NVVM::FmaOp>([&](auto) { return translateFmaOp(op); })
      .Case<NVVM::MatchSyncOp>([&](auto) { return translateMatchSyncOp(op); })
      .Case<NVVM::MembarOp>([&](auto) { return translateMembarOp(op); })
      .Case<NVVM::NanosleepOp>([&](auto) { return translateNanosleepOp(op); })
      .Case<NVVM::ReduxOp>([&](auto) { return translateReduxSyncOp(op); })
      .Case<NVVM::ShflOp>([&](auto) { return translateShflOp(op); })
      .Case<NVVM::SyncWarpOp>([&](auto) { return translateSyncWarpOp(op); })
      .Case<NVVM::VoteSyncOp>([&](auto) { return translateVoteSyncOp(op); })
      .Case<NVVM::ClusterArriveOp>(
          [&](auto) { return this->translateClusterArriveOp(op, false); })
      .Case<NVVM::ClusterArriveRelaxedOp>(
          [&](auto) { return this->translateClusterArriveOp(op, true); })
      .Case<NVVM::ClusterWaitOp>(
          [&](auto) { return this->translateClusterWaitOp(op); })
      .Case<NVVM::BarrierOp>(
          [&](auto) { return this->translateBarrierOp(op); })
      .Case<NVVM::Breakpoint>([&](auto) {
        LLVMTypeRef fnTy = b.funcTy(b.voidTy(), nullptr, 0, false);
        LLVMValueRef asmVal =
            b.constInlineAsm(fnTy, "brkpt;", "", true, false);
        b.buildCall(asmVal, nullptr, 0, "");
        return llvm::Error::success();
      })
      .Default([&](Operation *o) {
        return llvm::createStringError(
            llvm::inconvertibleErrorCode(), "unsupported NVVM op: %s",
            o->getName().getStringRef().str().c_str());
      });
}

//===----------------------------------------------------------------------===//
// Debug variable declarations
//===----------------------------------------------------------------------===//

LLVMMetadataRef MLIRToLLVM70::getOrCreateDIType(LLVM::DITypeAttr typeAttr) {
  if (!typeAttr) {
    // Fallback: opaque byte type
    return b.createDIBasicType("byte", 4, 8, llvm::dwarf::DW_ATE_unsigned);
  }

  if (auto basic = dyn_cast<LLVM::DIBasicTypeAttr>(typeAttr)) {
    llvm::StringRef name = basic.getName() ? basic.getName().getValue() : "";
    return b.createDIBasicType(name.data(), name.size(), basic.getSizeInBits(),
                               basic.getEncoding());
  }

  if (auto derived = dyn_cast<LLVM::DIDerivedTypeAttr>(typeAttr)) {
    // For pointer types and other derived types, create a basic type
    // representing the pointer itself (64-bit address on NVPTX).
    uint64_t size = derived.getSizeInBits();
    if (size == 0)
      size = 64; // NVPTX pointers are 64-bit
    return b.createDIBasicType("ptr", 3, size, llvm::dwarf::DW_ATE_address);
  }

  // Composite or unknown types: fall back to sized opaque type
  return b.createDIBasicType("data", 4, 0, llvm::dwarf::DW_ATE_unsigned);
}

llvm::Error MLIRToLLVM70::translateDbgDeclareOp(Operation *op) {
  auto dbgDecl = cast<LLVM::DbgDeclareOp>(op);
  return emitDbgIntrinsic(op, lookupValue(dbgDecl.getAddr()),
                          dbgDecl.getVarInfo(), /*isDeclare=*/true);
}

llvm::Error MLIRToLLVM70::translateDbgValueOp(Operation *op) {
  auto dbgVal = cast<LLVM::DbgValueOp>(op);
  return emitDbgIntrinsic(op, lookupValue(dbgVal.getValue()),
                          dbgVal.getVarInfo(), /*isDeclare=*/false);
}

llvm::Error MLIRToLLVM70::emitDbgIntrinsic(Operation *op, LLVMValueRef val,
                                           LLVM::DILocalVariableAttr varInfo,
                                           bool isDeclare) {
  if (!diCompileUnit || !val || !varInfo)
    return llvm::Error::success();

  auto nameAttr = varInfo.getName();
  if (!nameAttr)
    return llvm::Error::success();
  llvm::StringRef name = nameAttr.getValue();

  unsigned line = varInfo.getLine();
  LLVMMetadataRef diFile = varInfo.getFile()
      ? getOrCreateDIFile(varInfo.getFile().getName().getValue())
      : getOrCreateDIFile("llvm70_module");

  LLVMMetadataRef diType = getOrCreateDIType(varInfo.getType());
  LLVMMetadataRef scope = currentSubprogram ? currentSubprogram : diCompileUnit;

  LLVMMetadataRef diVar =
      b.createDIAutoVariable(scope, name.data(), name.size(), diFile, line,
                              diType, varInfo.getAlignInBits());
  LLVMMetadataRef diExpr = b.createDIExpression(nullptr, 0);

  auto [filename, opLine, opCol] = extractFileLineCol(op->getLoc());
  unsigned dbgLine = opLine ? opLine : line;
  LLVMMetadataRef debugLoc = b.createDebugLocation(dbgLine, opCol, scope);

  if (!diVar || !diExpr || !debugLoc)
    return llvm::Error::success();

  if (isDeclare)
    b.insertDbgDeclare(val, diVar, diExpr, debugLoc);
  else
    b.insertDbgValue(val, diVar, diExpr, debugLoc);
  return llvm::Error::success();
}

//===----------------------------------------------------------------------===//
// Kernel metadata
//===----------------------------------------------------------------------===//

void MLIRToLLVM70::emitKernelMetadata(LLVMValueRef fn, Operation *funcOp) {
  // Set ptx_kernel calling convention (CC 71) so libnvvm preserves the entry
  // point in LTOIR.  The !nvvm.annotations metadata alone is sufficient for
  // PTX but not for -gen-lto.
  b.setFunctionCallConv(fn, 71);

  // !nvvm.annotations = !{!0}
  // !0 = !{void (...)* @kernel, !"kernel", i32 1}
  LLVMValueRef vals[3];
  vals[0] = fn;
  vals[1] = b.mdString("kernel", 6);
  vals[2] = b.constInt(b.i32Ty(), 1, false);
  LLVMValueRef node = b.mdNode(vals, 3);
  b.addNamedMetadataOperand("nvvm.annotations", node);

  auto emitI32Annotation = [&](const char *name, int32_t value) {
    LLVMValueRef v[3];
    v[0] = fn;
    v[1] = b.mdString(name, std::strlen(name));
    v[2] = b.constInt(b.i32Ty(), value, false);
    b.addNamedMetadataOperand("nvvm.annotations", b.mdNode(v, 3));
  };

  if (auto attr = funcOp->getAttrOfType<DenseI32ArrayAttr>("nvvm.maxntid")) {
    auto vals = attr.asArrayRef();
    if (vals.size() >= 1 && vals[0] > 0) emitI32Annotation("maxntidx", vals[0]);
    if (vals.size() >= 2 && vals[1] > 0) emitI32Annotation("maxntidy", vals[1]);
    if (vals.size() >= 3 && vals[2] > 0) emitI32Annotation("maxntidz", vals[2]);
  }
  if (auto attr = funcOp->getAttrOfType<IntegerAttr>("nvvm.minctasm"))
    emitI32Annotation("minctasm", attr.getInt());
  if (auto attr = funcOp->getAttrOfType<IntegerAttr>("nvvm.cluster_max_blocks"))
    emitI32Annotation("maxclusterrank", attr.getInt());
}
