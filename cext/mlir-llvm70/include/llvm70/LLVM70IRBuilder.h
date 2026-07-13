/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
//===- LLVM70IRBuilder.h - Old LLVM C API wrapper ----------------*- C++ -*-===//
//
// C++ wrapper around a dlopen'd libLLVM-7.so (LLVM 7.0.1 C API).
// We reuse the standard LLVM-C opaque handle types (ABI-stable across versions)
// and resolve old C API function pointers via dlopen at runtime.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM70_LLVM70IRBUILDER_H
#define LLVM70_LLVM70IRBUILDER_H

#include "llvm70/CAPILoader.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include <llvm-c/Core.h>
#include <llvm-c/DebugInfo.h>
#include <memory>
#include <string>

namespace llvm70 {

//===----------------------------------------------------------------------===//
// LLVM70IRBuilder — builds LLVM 7 IR via dlopen'd libLLVM-7.so
//===----------------------------------------------------------------------===//
class LLVM70IRBuilder {
public:
  static llvm::Expected<std::unique_ptr<LLVM70IRBuilder>>
  create(llvm::StringRef libLLVMPath);

  ~LLVM70IRBuilder();
  LLVM70IRBuilder(const LLVM70IRBuilder &) = delete;

  // --- Module / Context ---
  LLVMModuleRef getModule() const { return module; }
  LLVMContextRef getContext() const { return ctx; }
  void setDataLayout(const char *dl);
  void setTarget(const char *triple);
  std::string printModuleToString();

  // --- Types ---
  LLVMTypeRef voidTy();
  LLVMTypeRef i1Ty();
  LLVMTypeRef i8Ty();
  LLVMTypeRef i16Ty();
  LLVMTypeRef i32Ty();
  LLVMTypeRef i64Ty();
  LLVMTypeRef intTy(unsigned bits);
  LLVMTypeRef halfTy();
  LLVMTypeRef floatTy();
  LLVMTypeRef doubleTy();
  LLVMTypeRef ptrTy(LLVMTypeRef elemTy, unsigned addrSpace = 0);
  LLVMTypeRef arrayTy(LLVMTypeRef elemTy, unsigned count);
  LLVMTypeRef structTy(LLVMTypeRef *elems, unsigned count, bool packed);
  LLVMTypeRef funcTy(LLVMTypeRef ret, LLVMTypeRef *params, unsigned count,
                     bool varArg);
  LLVMTypeRef vectorTy(LLVMTypeRef elemTy, unsigned count);

  // --- Functions ---
  LLVMValueRef addFunction(const char *name, LLVMTypeRef ty);
  LLVMValueRef getNamedFunction(const char *name);
  void setLinkage(LLVMValueRef val, LLVMLinkage lk);
  void setFunctionCallConv(LLVMValueRef fn, unsigned cc);
  LLVMValueRef getParam(LLVMValueRef fn, unsigned idx);
  unsigned countParams(LLVMValueRef fn);
  void setValueName(LLVMValueRef val, const char *name);
  bool isInstruction(LLVMValueRef val);

  // --- Basic blocks ---
  LLVMBasicBlockRef appendBB(LLVMValueRef fn, const char *name);
  void positionAtEnd(LLVMBasicBlockRef bb);
  LLVMBasicBlockRef getInsertBlock();

  // --- Constants ---
  LLVMValueRef constInt(LLVMTypeRef ty, unsigned long long val, bool signExt);
  LLVMValueRef constReal(LLVMTypeRef ty, double val);
  LLVMValueRef constNull(LLVMTypeRef ty);
  LLVMValueRef getUndef(LLVMTypeRef ty);
  LLVMValueRef constStruct(LLVMValueRef *vals, unsigned count, bool packed);
  LLVMValueRef constArray(LLVMTypeRef elemTy, LLVMValueRef *vals,
                          unsigned count);
  LLVMValueRef constVector(LLVMValueRef *vals, unsigned count);
  LLVMValueRef constBitCast(LLVMValueRef val, LLVMTypeRef ty);

  // --- Inline asm ---
  LLVMValueRef constInlineAsm(LLVMTypeRef fnTy, const char *asmString,
                              const char *constraints, bool hasSideEffects,
                              bool isAlignStack);

  // --- Globals ---
  LLVMValueRef addGlobal(LLVMTypeRef ty, const char *name);
  LLVMValueRef addGlobalInAddressSpace(LLVMTypeRef ty, const char *name,
                                       unsigned addrSpace);
  LLVMValueRef getNamedGlobal(const char *name);
  void setInitializer(LLVMValueRef global, LLVMValueRef init);
  void setGlobalAlignment(LLVMValueRef global, unsigned bytes);
  void setSection(LLVMValueRef global, const char *section);

  // --- Arithmetic ---
  LLVMValueRef buildAdd(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildSub(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildMul(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildSDiv(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildUDiv(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildSRem(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildURem(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildFAdd(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildFSub(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildFMul(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildFDiv(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildFRem(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildFNeg(LLVMValueRef val, const char *name);

  // --- Logical ---
  LLVMValueRef buildAnd(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildOr(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildXor(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildShl(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildLShr(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);
  LLVMValueRef buildAShr(LLVMValueRef lhs, LLVMValueRef rhs, const char *name);

  // --- Comparison ---
  LLVMValueRef buildICmp(LLVMIntPredicate pred, LLVMValueRef lhs,
                         LLVMValueRef rhs, const char *name);
  LLVMValueRef buildFCmp(LLVMRealPredicate pred, LLVMValueRef lhs,
                         LLVMValueRef rhs, const char *name);

  // --- Memory ---
  LLVMValueRef buildAlloca(LLVMTypeRef ty, const char *name);
  LLVMValueRef buildArrayAlloca(LLVMTypeRef ty, LLVMValueRef numElems,
                                const char *name);
  LLVMValueRef buildLoad(LLVMValueRef ptr, const char *name);
  LLVMValueRef buildStore(LLVMValueRef val, LLVMValueRef ptr);
  LLVMValueRef buildGEP(LLVMValueRef ptr, LLVMValueRef *indices,
                        unsigned numIdx, const char *name);
  LLVMValueRef buildInBoundsGEP(LLVMValueRef ptr, LLVMValueRef *indices,
                                unsigned numIdx, const char *name);
  LLVMValueRef buildStructGEP(LLVMValueRef ptr, unsigned idx,
                              const char *name);

  // --- Atomics ---
  LLVMValueRef buildAtomicRMW(LLVMAtomicRMWBinOp op, LLVMValueRef ptr,
                              LLVMValueRef val, LLVMAtomicOrdering ordering,
                              bool singleThread);
  LLVMValueRef buildAtomicCmpXchg(LLVMValueRef ptr, LLVMValueRef cmp,
                                  LLVMValueRef newVal,
                                  LLVMAtomicOrdering successOrdering,
                                  LLVMAtomicOrdering failureOrdering,
                                  bool singleThread);

  // --- Casts ---
  LLVMValueRef buildBitCast(LLVMValueRef val, LLVMTypeRef ty,
                            const char *name);
  LLVMValueRef buildAddrSpaceCast(LLVMValueRef val, LLVMTypeRef ty,
                                  const char *name);
  LLVMValueRef buildIntToPtr(LLVMValueRef val, LLVMTypeRef ty,
                             const char *name);
  LLVMValueRef buildPtrToInt(LLVMValueRef val, LLVMTypeRef ty,
                             const char *name);
  LLVMValueRef buildTrunc(LLVMValueRef val, LLVMTypeRef ty, const char *name);
  LLVMValueRef buildZExt(LLVMValueRef val, LLVMTypeRef ty, const char *name);
  LLVMValueRef buildSExt(LLVMValueRef val, LLVMTypeRef ty, const char *name);
  LLVMValueRef buildFPTrunc(LLVMValueRef val, LLVMTypeRef ty,
                            const char *name);
  LLVMValueRef buildFPExt(LLVMValueRef val, LLVMTypeRef ty, const char *name);
  LLVMValueRef buildFPToSI(LLVMValueRef val, LLVMTypeRef ty, const char *name);
  LLVMValueRef buildFPToUI(LLVMValueRef val, LLVMTypeRef ty, const char *name);
  LLVMValueRef buildSIToFP(LLVMValueRef val, LLVMTypeRef ty, const char *name);
  LLVMValueRef buildUIToFP(LLVMValueRef val, LLVMTypeRef ty, const char *name);

  // --- Control flow ---
  LLVMValueRef buildCall(LLVMValueRef fn, LLVMValueRef *args, unsigned numArgs,
                         const char *name);
  LLVMValueRef buildRet(LLVMValueRef val);
  LLVMValueRef buildRetVoid();
  LLVMValueRef buildBr(LLVMBasicBlockRef dest);
  LLVMValueRef buildCondBr(LLVMValueRef cond, LLVMBasicBlockRef thenBB,
                           LLVMBasicBlockRef elseBB);
  LLVMValueRef buildPhi(LLVMTypeRef ty, const char *name);
  void addIncoming(LLVMValueRef phi, LLVMValueRef *vals,
                   LLVMBasicBlockRef *blocks, unsigned count);
  LLVMValueRef buildSelect(LLVMValueRef cond, LLVMValueRef thenVal,
                           LLVMValueRef elseVal, const char *name);
  LLVMValueRef buildSwitch(LLVMValueRef val, LLVMBasicBlockRef elseBB,
                           unsigned numCases);
  void addCase(LLVMValueRef switchInst, LLVMValueRef onVal,
               LLVMBasicBlockRef dest);
  LLVMValueRef buildUnreachable();

  // --- Aggregate ---
  LLVMValueRef buildExtractValue(LLVMValueRef agg, unsigned idx,
                                 const char *name);
  LLVMValueRef buildInsertValue(LLVMValueRef agg, LLVMValueRef val,
                                unsigned idx, const char *name);

  // --- Vector ---
  LLVMValueRef buildExtractElement(LLVMValueRef vec, LLVMValueRef idx,
                                   const char *name);
  LLVMValueRef buildInsertElement(LLVMValueRef vec, LLVMValueRef val,
                                  LLVMValueRef idx, const char *name);

  // --- Metadata (for !nvvm.annotations) ---
  LLVMValueRef mdString(const char *str, unsigned len);
  LLVMValueRef mdNode(LLVMValueRef *vals, unsigned count);
  void addNamedMetadataOperand(const char *name, LLVMValueRef val);

  // --- Debug info ---
  void initDebugInfo();
  void finalizeDebugInfo();
  LLVMMetadataRef createDIFile(const char *filename, size_t filenameLen,
                               const char *directory, size_t directoryLen);
  LLVMMetadataRef createDICompileUnit(LLVMMetadataRef file,
                                      bool fullDebug = false);
  LLVMMetadataRef createDISubroutineType(LLVMMetadataRef file);
  LLVMMetadataRef createDIFunction(LLVMMetadataRef scope, const char *name,
                                   size_t nameLen, LLVMMetadataRef file,
                                   unsigned lineNo, LLVMMetadataRef type);
  void setSubprogram(LLVMValueRef fn, LLVMMetadataRef sp);
  void setDebugLocation(unsigned line, unsigned col, LLVMMetadataRef scope);
  void clearDebugLocation();
  LLVMMetadataRef createDebugLocation(unsigned line, unsigned col,
                                      LLVMMetadataRef scope);
  LLVMMetadataRef createDIBasicType(const char *name, size_t nameLen,
                                    uint64_t sizeInBits,
                                    LLVMDWARFTypeEncoding encoding);
  LLVMMetadataRef createDIAutoVariable(LLVMMetadataRef scope, const char *name,
                                       size_t nameLen, LLVMMetadataRef file,
                                       unsigned lineNo, LLVMMetadataRef type,
                                       uint32_t alignInBits);
  LLVMMetadataRef createDIExpression(int64_t *ops, size_t count);
  LLVMValueRef insertDbgDeclare(LLVMValueRef storage,
                                LLVMMetadataRef varInfo,
                                LLVMMetadataRef expr,
                                LLVMMetadataRef debugLoc);
  LLVMValueRef insertDbgValue(LLVMValueRef val,
                              LLVMMetadataRef varInfo,
                              LLVMMetadataRef expr,
                              LLVMMetadataRef debugLoc);

  // --- Bitcode serialization ---
  /// Writes the module to an in-memory buffer. Caller must free via
  /// disposeMemoryBuffer().
  LLVMMemoryBufferRef writeBitcodeToMemoryBuffer();
  const char *getBufferStart(LLVMMemoryBufferRef buf);
  size_t getBufferSize(LLVMMemoryBufferRef buf);
  void disposeMemoryBuffer(LLVMMemoryBufferRef buf);

private:
  LLVM70IRBuilder() = default;
  llvm::Error resolveSymbols();

  std::unique_ptr<CAPILoader> loader;
  LLVMContextRef ctx = nullptr;
  LLVMModuleRef module = nullptr;
  LLVMBuilderRef builder = nullptr;

  // Function pointers — resolved from old libLLVM at load time.
  // Grouped by category; all set in resolveSymbols().
#define LLVM_FN(RET, NAME, ...) RET (*NAME)(__VA_ARGS__) = nullptr;

  // Context / module
  LLVM_FN(LLVMContextRef, fnContextCreate)
  LLVM_FN(void, fnContextDispose, LLVMContextRef)
  LLVM_FN(LLVMModuleRef, fnModuleCreateWithNameInContext, const char *,
           LLVMContextRef)
  LLVM_FN(void, fnDisposeModule, LLVMModuleRef)
  LLVM_FN(void, fnSetDataLayout, LLVMModuleRef, const char *)
  LLVM_FN(void, fnSetTarget, LLVMModuleRef, const char *)
  LLVM_FN(char *, fnPrintModuleToString, LLVMModuleRef)
  LLVM_FN(void, fnDisposeMessage, char *)

  // Types
  LLVM_FN(LLVMTypeRef, fnVoidType, LLVMContextRef)
  LLVM_FN(LLVMTypeRef, fnInt1Type, LLVMContextRef)
  LLVM_FN(LLVMTypeRef, fnInt8Type, LLVMContextRef)
  LLVM_FN(LLVMTypeRef, fnInt16Type, LLVMContextRef)
  LLVM_FN(LLVMTypeRef, fnInt32Type, LLVMContextRef)
  LLVM_FN(LLVMTypeRef, fnInt64Type, LLVMContextRef)
  LLVM_FN(LLVMTypeRef, fnIntType, LLVMContextRef, unsigned)
  LLVM_FN(LLVMTypeRef, fnHalfType, LLVMContextRef)
  LLVM_FN(LLVMTypeRef, fnFloatType, LLVMContextRef)
  LLVM_FN(LLVMTypeRef, fnDoubleType, LLVMContextRef)
  LLVM_FN(LLVMTypeRef, fnPointerType, LLVMTypeRef, unsigned)
  LLVM_FN(LLVMTypeRef, fnArrayType, LLVMTypeRef, unsigned)
  LLVM_FN(LLVMTypeRef, fnStructType, LLVMContextRef, LLVMTypeRef *, unsigned,
           LLVMBool)
  LLVM_FN(LLVMTypeRef, fnFunctionType, LLVMTypeRef, LLVMTypeRef *, unsigned,
           LLVMBool)
  LLVM_FN(LLVMTypeRef, fnVectorType, LLVMTypeRef, unsigned)

  // Functions
  LLVM_FN(LLVMValueRef, fnAddFunction, LLVMModuleRef, const char *,
           LLVMTypeRef)
  LLVM_FN(LLVMValueRef, fnGetNamedFunction, LLVMModuleRef, const char *)
  LLVM_FN(void, fnSetLinkage, LLVMValueRef, LLVMLinkage)
  LLVM_FN(void, fnSetFunctionCallConv, LLVMValueRef, unsigned)
  LLVM_FN(LLVMValueRef, fnGetParam, LLVMValueRef, unsigned)
  LLVM_FN(unsigned, fnCountParams, LLVMValueRef)
  LLVM_FN(void, fnSetValueName2, LLVMValueRef, const char *, size_t)
  LLVM_FN(LLVMValueRef, fnIsAInstruction, LLVMValueRef)

  // Basic blocks
  LLVM_FN(LLVMBasicBlockRef, fnAppendBB, LLVMContextRef, LLVMValueRef,
           const char *)
  LLVM_FN(LLVMBasicBlockRef, fnGetInsertBlock, LLVMBuilderRef)

  // Builder
  LLVM_FN(LLVMBuilderRef, fnCreateBuilder, LLVMContextRef)
  LLVM_FN(void, fnPositionAtEnd, LLVMBuilderRef, LLVMBasicBlockRef)
  LLVM_FN(void, fnDisposeBuilder, LLVMBuilderRef)

  // Constants
  LLVM_FN(LLVMValueRef, fnConstInt, LLVMTypeRef, unsigned long long, LLVMBool)
  LLVM_FN(LLVMValueRef, fnConstReal, LLVMTypeRef, double)
  LLVM_FN(LLVMValueRef, fnConstNull, LLVMTypeRef)
  LLVM_FN(LLVMValueRef, fnGetUndef, LLVMTypeRef)
  LLVM_FN(LLVMValueRef, fnConstStruct, LLVMContextRef, LLVMValueRef *,
           unsigned, LLVMBool)
  LLVM_FN(LLVMValueRef, fnConstArray, LLVMTypeRef, LLVMValueRef *, unsigned)
  LLVM_FN(LLVMValueRef, fnConstVector, LLVMValueRef *, unsigned)
  LLVM_FN(LLVMValueRef, fnConstBitCast, LLVMValueRef, LLVMTypeRef)

  // Inline asm
  LLVM_FN(LLVMValueRef, fnConstInlineAsm, LLVMTypeRef, const char *,
           const char *, LLVMBool, LLVMBool)

  // Globals
  LLVM_FN(LLVMValueRef, fnAddGlobal, LLVMModuleRef, LLVMTypeRef, const char *)
  LLVM_FN(LLVMValueRef, fnAddGlobalInAddressSpace, LLVMModuleRef, LLVMTypeRef,
           const char *, unsigned)
  LLVM_FN(LLVMValueRef, fnGetNamedGlobal, LLVMModuleRef, const char *)
  LLVM_FN(void, fnSetInitializer, LLVMValueRef, LLVMValueRef)
  LLVM_FN(void, fnSetAlignment, LLVMValueRef, unsigned)
  LLVM_FN(void, fnSetSection, LLVMValueRef, const char *)

  // Binary ops (all have the same signature: Builder, LHS, RHS, Name)
  using BinOpFn = LLVMValueRef (*)(LLVMBuilderRef, LLVMValueRef, LLVMValueRef,
                                   const char *);
  BinOpFn fnBuildAdd = nullptr, fnBuildSub = nullptr, fnBuildMul = nullptr;
  BinOpFn fnBuildSDiv = nullptr, fnBuildUDiv = nullptr;
  BinOpFn fnBuildSRem = nullptr, fnBuildURem = nullptr;
  BinOpFn fnBuildFAdd = nullptr, fnBuildFSub = nullptr;
  BinOpFn fnBuildFMul = nullptr, fnBuildFDiv = nullptr, fnBuildFRem = nullptr;
  BinOpFn fnBuildAnd = nullptr, fnBuildOr = nullptr, fnBuildXor = nullptr;
  BinOpFn fnBuildShl = nullptr, fnBuildLShr = nullptr, fnBuildAShr = nullptr;

  using UnaryOpFn = LLVMValueRef (*)(LLVMBuilderRef, LLVMValueRef,
                                     const char *);
  UnaryOpFn fnBuildFNeg = nullptr;

  // Comparison
  LLVM_FN(LLVMValueRef, fnBuildICmp, LLVMBuilderRef, LLVMIntPredicate,
           LLVMValueRef, LLVMValueRef, const char *)
  LLVM_FN(LLVMValueRef, fnBuildFCmp, LLVMBuilderRef, LLVMRealPredicate,
           LLVMValueRef, LLVMValueRef, const char *)

  // Memory
  LLVM_FN(LLVMValueRef, fnBuildAlloca, LLVMBuilderRef, LLVMTypeRef,
           const char *)
  LLVM_FN(LLVMValueRef, fnBuildArrayAlloca, LLVMBuilderRef, LLVMTypeRef,
           LLVMValueRef, const char *)
  LLVM_FN(LLVMValueRef, fnBuildLoad, LLVMBuilderRef, LLVMValueRef,
           const char *)
  LLVM_FN(LLVMValueRef, fnBuildStore, LLVMBuilderRef, LLVMValueRef,
           LLVMValueRef)
  LLVM_FN(LLVMValueRef, fnBuildGEP, LLVMBuilderRef, LLVMValueRef,
           LLVMValueRef *, unsigned, const char *)
  LLVM_FN(LLVMValueRef, fnBuildInBoundsGEP, LLVMBuilderRef, LLVMValueRef,
           LLVMValueRef *, unsigned, const char *)
  LLVM_FN(LLVMValueRef, fnBuildStructGEP, LLVMBuilderRef, LLVMValueRef,
           unsigned, const char *)

  // Atomics
  LLVM_FN(LLVMValueRef, fnBuildAtomicRMW, LLVMBuilderRef, LLVMAtomicRMWBinOp,
           LLVMValueRef, LLVMValueRef, LLVMAtomicOrdering, LLVMBool)
  LLVM_FN(LLVMValueRef, fnBuildAtomicCmpXchg, LLVMBuilderRef, LLVMValueRef,
           LLVMValueRef, LLVMValueRef, LLVMAtomicOrdering, LLVMAtomicOrdering,
           LLVMBool)

  // Casts (all: Builder, Val, DestTy, Name)
  using CastFn = LLVMValueRef (*)(LLVMBuilderRef, LLVMValueRef, LLVMTypeRef,
                                  const char *);
  CastFn fnBuildBitCast = nullptr, fnBuildAddrSpaceCast = nullptr;
  CastFn fnBuildIntToPtr = nullptr, fnBuildPtrToInt = nullptr;
  CastFn fnBuildTrunc = nullptr, fnBuildZExt = nullptr, fnBuildSExt = nullptr;
  CastFn fnBuildFPTrunc = nullptr, fnBuildFPExt = nullptr;
  CastFn fnBuildFPToSI = nullptr, fnBuildFPToUI = nullptr;
  CastFn fnBuildSIToFP = nullptr, fnBuildUIToFP = nullptr;

  // Control flow
  LLVM_FN(LLVMValueRef, fnBuildCall, LLVMBuilderRef, LLVMValueRef,
           LLVMValueRef *, unsigned, const char *)
  LLVM_FN(LLVMValueRef, fnBuildRet, LLVMBuilderRef, LLVMValueRef)
  LLVM_FN(LLVMValueRef, fnBuildRetVoid, LLVMBuilderRef)
  LLVM_FN(LLVMValueRef, fnBuildBr, LLVMBuilderRef, LLVMBasicBlockRef)
  LLVM_FN(LLVMValueRef, fnBuildCondBr, LLVMBuilderRef, LLVMValueRef,
           LLVMBasicBlockRef, LLVMBasicBlockRef)
  LLVM_FN(LLVMValueRef, fnBuildPhi, LLVMBuilderRef, LLVMTypeRef, const char *)
  LLVM_FN(void, fnAddIncoming, LLVMValueRef, LLVMValueRef *,
           LLVMBasicBlockRef *, unsigned)
  LLVM_FN(LLVMValueRef, fnBuildSelect, LLVMBuilderRef, LLVMValueRef,
           LLVMValueRef, LLVMValueRef, const char *)
  LLVM_FN(LLVMValueRef, fnBuildSwitch, LLVMBuilderRef, LLVMValueRef,
           LLVMBasicBlockRef, unsigned)
  LLVM_FN(void, fnAddCase, LLVMValueRef, LLVMValueRef, LLVMBasicBlockRef)
  LLVM_FN(LLVMValueRef, fnBuildUnreachable, LLVMBuilderRef)

  // Aggregate
  LLVM_FN(LLVMValueRef, fnBuildExtractValue, LLVMBuilderRef, LLVMValueRef,
           unsigned, const char *)
  LLVM_FN(LLVMValueRef, fnBuildInsertValue, LLVMBuilderRef, LLVMValueRef,
           LLVMValueRef, unsigned, const char *)

  // Vector
  LLVM_FN(LLVMValueRef, fnBuildExtractElement, LLVMBuilderRef, LLVMValueRef,
           LLVMValueRef, const char *)
  LLVM_FN(LLVMValueRef, fnBuildInsertElement, LLVMBuilderRef, LLVMValueRef,
           LLVMValueRef, LLVMValueRef, const char *)

  // Metadata
  LLVM_FN(LLVMValueRef, fnMDString, LLVMContextRef, const char *, unsigned)
  LLVM_FN(LLVMValueRef, fnMDNode, LLVMContextRef, LLVMValueRef *, unsigned)
  LLVM_FN(void, fnAddNamedMetadataOperand, LLVMModuleRef, const char *,
           LLVMValueRef)

  // Debug info
  LLVMDIBuilderRef diBuilder = nullptr;
  LLVM_FN(LLVMDIBuilderRef, fnCreateDIBuilder, LLVMModuleRef)
  LLVM_FN(void, fnDisposeDIBuilder, LLVMDIBuilderRef)
  LLVM_FN(void, fnDIBuilderFinalize, LLVMDIBuilderRef)
  LLVM_FN(LLVMMetadataRef, fnDIBuilderCreateFile, LLVMDIBuilderRef,
           const char *, size_t, const char *, size_t)
  LLVM_FN(LLVMMetadataRef, fnDIBuilderCreateCompileUnit, LLVMDIBuilderRef,
           LLVMDWARFSourceLanguage, LLVMMetadataRef, const char *, size_t,
           LLVMBool, const char *, size_t, unsigned, const char *, size_t,
           LLVMDWARFEmissionKind, unsigned, LLVMBool, LLVMBool)
  LLVM_FN(LLVMMetadataRef, fnDIBuilderCreateSubroutineType, LLVMDIBuilderRef,
           LLVMMetadataRef, LLVMMetadataRef *, unsigned, LLVMDIFlags)
  LLVM_FN(LLVMMetadataRef, fnDIBuilderCreateFunction, LLVMDIBuilderRef,
           LLVMMetadataRef, const char *, size_t, const char *, size_t,
           LLVMMetadataRef, unsigned, LLVMMetadataRef, LLVMBool, LLVMBool,
           unsigned, LLVMDIFlags, LLVMBool)
  LLVM_FN(LLVMMetadataRef, fnDIBuilderCreateDebugLocation, LLVMContextRef,
           unsigned, unsigned, LLVMMetadataRef, LLVMMetadataRef)
  LLVM_FN(void, fnSetCurrentDebugLocation, LLVMBuilderRef, LLVMValueRef)
  LLVM_FN(LLVMValueRef, fnMetadataAsValue, LLVMContextRef, LLVMMetadataRef)
  LLVM_FN(void, fnSetSubprogram, LLVMValueRef, LLVMMetadataRef)
  LLVM_FN(LLVMMetadataRef, fnDIBuilderCreateBasicType, LLVMDIBuilderRef,
           const char *, size_t, uint64_t, LLVMDWARFTypeEncoding)
  LLVM_FN(LLVMMetadataRef, fnDIBuilderCreateAutoVariable, LLVMDIBuilderRef,
           LLVMMetadataRef, const char *, size_t, LLVMMetadataRef, unsigned,
           LLVMMetadataRef, LLVMBool, LLVMDIFlags, uint32_t)
  LLVM_FN(LLVMMetadataRef, fnDIBuilderCreateExpression, LLVMDIBuilderRef,
           int64_t *, size_t)
  LLVM_FN(LLVMValueRef, fnDIBuilderInsertDeclareAtEnd, LLVMDIBuilderRef,
           LLVMValueRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef,
           LLVMBasicBlockRef)
  LLVM_FN(LLVMValueRef, fnDIBuilderInsertDbgValueAtEnd, LLVMDIBuilderRef,
           LLVMValueRef, LLVMMetadataRef, LLVMMetadataRef, LLVMMetadataRef,
           LLVMBasicBlockRef)

  // Bitcode
  LLVM_FN(LLVMMemoryBufferRef, fnWriteBitcodeToMemoryBuffer, LLVMModuleRef)
  LLVM_FN(const char *, fnGetBufferStart, LLVMMemoryBufferRef)
  LLVM_FN(size_t, fnGetBufferSize, LLVMMemoryBufferRef)
  LLVM_FN(void, fnDisposeMemoryBuffer, LLVMMemoryBufferRef)

#undef LLVM_FN
};

} // namespace llvm70

#endif // LLVM70_LLVM70IRBUILDER_H
