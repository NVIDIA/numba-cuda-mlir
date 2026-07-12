/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
//===- LLVM70IRBuilder.cpp - Old LLVM C API wrapper --------------*- C++ -*-===//
//
//===----------------------------------------------------------------------===//

#include "llvm70/LLVM70IRBuilder.h"

using namespace llvm70;

llvm::Expected<std::unique_ptr<LLVM70IRBuilder>>
LLVM70IRBuilder::create(llvm::StringRef libLLVMPath) {
  auto loaderOrErr = CAPILoader::create(libLLVMPath);
  if (!loaderOrErr)
    return loaderOrErr.takeError();

  auto b = std::unique_ptr<LLVM70IRBuilder>(new LLVM70IRBuilder());
  b->loader = std::move(*loaderOrErr);

  if (auto err = b->resolveSymbols())
    return std::move(err);

  b->ctx = b->fnContextCreate();
  b->module =
      b->fnModuleCreateWithNameInContext("llvm70_module", b->ctx);
  b->builder = b->fnCreateBuilder(b->ctx);
  return std::move(b);
}

LLVM70IRBuilder::~LLVM70IRBuilder() {
  if (diBuilder && fnDisposeDIBuilder)
    fnDisposeDIBuilder(diBuilder);
  if (builder && fnDisposeBuilder)
    fnDisposeBuilder(builder);
  if (module && fnDisposeModule)
    fnDisposeModule(module);
  if (ctx && fnContextDispose)
    fnContextDispose(ctx);
}

//===----------------------------------------------------------------------===//
// Symbol resolution — one big block so failures are caught early
//===----------------------------------------------------------------------===//

#define RESOLVE(FIELD, SYM)                                                    \
  do {                                                                         \
    auto r = loader->resolve<decltype(FIELD)>(SYM);                            \
    if (!r)                                                                    \
      return r.takeError();                                                    \
    FIELD = *r;                                                                \
  } while (0)

llvm::Error LLVM70IRBuilder::resolveSymbols() {
  // Context / module
  RESOLVE(fnContextCreate, "LLVMContextCreate");
  RESOLVE(fnContextDispose, "LLVMContextDispose");
  RESOLVE(fnModuleCreateWithNameInContext, "LLVMModuleCreateWithNameInContext");
  RESOLVE(fnDisposeModule, "LLVMDisposeModule");
  RESOLVE(fnSetDataLayout, "LLVMSetDataLayout");
  RESOLVE(fnSetTarget, "LLVMSetTarget");
  RESOLVE(fnPrintModuleToString, "LLVMPrintModuleToString");
  RESOLVE(fnDisposeMessage, "LLVMDisposeMessage");

  // Types
  RESOLVE(fnVoidType, "LLVMVoidTypeInContext");
  RESOLVE(fnInt1Type, "LLVMInt1TypeInContext");
  RESOLVE(fnInt8Type, "LLVMInt8TypeInContext");
  RESOLVE(fnInt16Type, "LLVMInt16TypeInContext");
  RESOLVE(fnInt32Type, "LLVMInt32TypeInContext");
  RESOLVE(fnInt64Type, "LLVMInt64TypeInContext");
  RESOLVE(fnIntType, "LLVMIntTypeInContext");
  RESOLVE(fnHalfType, "LLVMHalfTypeInContext");
  RESOLVE(fnFloatType, "LLVMFloatTypeInContext");
  RESOLVE(fnDoubleType, "LLVMDoubleTypeInContext");
  RESOLVE(fnPointerType, "LLVMPointerType");
  RESOLVE(fnArrayType, "LLVMArrayType");
  RESOLVE(fnStructType, "LLVMStructTypeInContext");
  RESOLVE(fnFunctionType, "LLVMFunctionType");
  RESOLVE(fnVectorType, "LLVMVectorType");

  // Functions
  RESOLVE(fnAddFunction, "LLVMAddFunction");
  RESOLVE(fnGetNamedFunction, "LLVMGetNamedFunction");
  RESOLVE(fnSetLinkage, "LLVMSetLinkage");
  RESOLVE(fnSetFunctionCallConv, "LLVMSetFunctionCallConv");
  RESOLVE(fnGetParam, "LLVMGetParam");
  RESOLVE(fnCountParams, "LLVMCountParams");
  RESOLVE(fnSetValueName2, "LLVMSetValueName2");

  // Basic blocks
  RESOLVE(fnAppendBB, "LLVMAppendBasicBlockInContext");
  RESOLVE(fnGetInsertBlock, "LLVMGetInsertBlock");

  // Builder
  RESOLVE(fnCreateBuilder, "LLVMCreateBuilderInContext");
  RESOLVE(fnPositionAtEnd, "LLVMPositionBuilderAtEnd");
  RESOLVE(fnDisposeBuilder, "LLVMDisposeBuilder");

  // Constants
  RESOLVE(fnConstInt, "LLVMConstInt");
  RESOLVE(fnConstReal, "LLVMConstReal");
  RESOLVE(fnConstNull, "LLVMConstNull");
  RESOLVE(fnGetUndef, "LLVMGetUndef");
  RESOLVE(fnConstStruct, "LLVMConstStructInContext");
  RESOLVE(fnConstArray, "LLVMConstArray");
  RESOLVE(fnConstVector, "LLVMConstVector");
  RESOLVE(fnConstBitCast, "LLVMConstBitCast");

  // Inline asm
  RESOLVE(fnConstInlineAsm, "LLVMConstInlineAsm");

  // Globals
  RESOLVE(fnAddGlobal, "LLVMAddGlobal");
  RESOLVE(fnAddGlobalInAddressSpace, "LLVMAddGlobalInAddressSpace");
  RESOLVE(fnGetNamedGlobal, "LLVMGetNamedGlobal");
  RESOLVE(fnSetInitializer, "LLVMSetInitializer");
  RESOLVE(fnSetAlignment, "LLVMSetAlignment");
  RESOLVE(fnSetSection, "LLVMSetSection");

  // Arithmetic
  RESOLVE(fnBuildAdd, "LLVMBuildAdd");
  RESOLVE(fnBuildSub, "LLVMBuildSub");
  RESOLVE(fnBuildMul, "LLVMBuildMul");
  RESOLVE(fnBuildSDiv, "LLVMBuildSDiv");
  RESOLVE(fnBuildUDiv, "LLVMBuildUDiv");
  RESOLVE(fnBuildSRem, "LLVMBuildSRem");
  RESOLVE(fnBuildURem, "LLVMBuildURem");
  RESOLVE(fnBuildFAdd, "LLVMBuildFAdd");
  RESOLVE(fnBuildFSub, "LLVMBuildFSub");
  RESOLVE(fnBuildFMul, "LLVMBuildFMul");
  RESOLVE(fnBuildFDiv, "LLVMBuildFDiv");
  RESOLVE(fnBuildFRem, "LLVMBuildFRem");
  RESOLVE(fnBuildFNeg, "LLVMBuildFNeg");

  // Logical
  RESOLVE(fnBuildAnd, "LLVMBuildAnd");
  RESOLVE(fnBuildOr, "LLVMBuildOr");
  RESOLVE(fnBuildXor, "LLVMBuildXor");
  RESOLVE(fnBuildShl, "LLVMBuildShl");
  RESOLVE(fnBuildLShr, "LLVMBuildLShr");
  RESOLVE(fnBuildAShr, "LLVMBuildAShr");

  // Comparison
  RESOLVE(fnBuildICmp, "LLVMBuildICmp");
  RESOLVE(fnBuildFCmp, "LLVMBuildFCmp");

  // Memory
  RESOLVE(fnBuildAlloca, "LLVMBuildAlloca");
  RESOLVE(fnBuildArrayAlloca, "LLVMBuildArrayAlloca");
  RESOLVE(fnBuildLoad, "LLVMBuildLoad");
  RESOLVE(fnBuildStore, "LLVMBuildStore");
  RESOLVE(fnBuildGEP, "LLVMBuildGEP");
  RESOLVE(fnBuildInBoundsGEP, "LLVMBuildInBoundsGEP");
  RESOLVE(fnBuildStructGEP, "LLVMBuildStructGEP");

  // Atomics
  RESOLVE(fnBuildAtomicRMW, "LLVMBuildAtomicRMW");
  RESOLVE(fnBuildAtomicCmpXchg, "LLVMBuildAtomicCmpXchg");

  // Casts
  RESOLVE(fnBuildBitCast, "LLVMBuildBitCast");
  RESOLVE(fnBuildAddrSpaceCast, "LLVMBuildAddrSpaceCast");
  RESOLVE(fnBuildIntToPtr, "LLVMBuildIntToPtr");
  RESOLVE(fnBuildPtrToInt, "LLVMBuildPtrToInt");
  RESOLVE(fnBuildTrunc, "LLVMBuildTrunc");
  RESOLVE(fnBuildZExt, "LLVMBuildZExt");
  RESOLVE(fnBuildSExt, "LLVMBuildSExt");
  RESOLVE(fnBuildFPTrunc, "LLVMBuildFPTrunc");
  RESOLVE(fnBuildFPExt, "LLVMBuildFPExt");
  RESOLVE(fnBuildFPToSI, "LLVMBuildFPToSI");
  RESOLVE(fnBuildFPToUI, "LLVMBuildFPToUI");
  RESOLVE(fnBuildSIToFP, "LLVMBuildSIToFP");
  RESOLVE(fnBuildUIToFP, "LLVMBuildUIToFP");

  // Control flow
  RESOLVE(fnBuildCall, "LLVMBuildCall");
  RESOLVE(fnBuildRet, "LLVMBuildRet");
  RESOLVE(fnBuildRetVoid, "LLVMBuildRetVoid");
  RESOLVE(fnBuildBr, "LLVMBuildBr");
  RESOLVE(fnBuildCondBr, "LLVMBuildCondBr");
  RESOLVE(fnBuildPhi, "LLVMBuildPhi");
  RESOLVE(fnAddIncoming, "LLVMAddIncoming");
  RESOLVE(fnBuildSelect, "LLVMBuildSelect");
  RESOLVE(fnBuildSwitch, "LLVMBuildSwitch");
  RESOLVE(fnAddCase, "LLVMAddCase");
  RESOLVE(fnBuildUnreachable, "LLVMBuildUnreachable");

  // Aggregate
  RESOLVE(fnBuildExtractValue, "LLVMBuildExtractValue");
  RESOLVE(fnBuildInsertValue, "LLVMBuildInsertValue");

  // Vector
  RESOLVE(fnBuildExtractElement, "LLVMBuildExtractElement");
  RESOLVE(fnBuildInsertElement, "LLVMBuildInsertElement");

  // Metadata
  RESOLVE(fnMDString, "LLVMMDStringInContext");
  RESOLVE(fnMDNode, "LLVMMDNodeInContext");
  RESOLVE(fnAddNamedMetadataOperand, "LLVMAddNamedMetadataOperand");

  // Debug info
  RESOLVE(fnCreateDIBuilder, "LLVMCreateDIBuilder");
  RESOLVE(fnDisposeDIBuilder, "LLVMDisposeDIBuilder");
  RESOLVE(fnDIBuilderFinalize, "LLVMDIBuilderFinalize");
  RESOLVE(fnDIBuilderCreateFile, "LLVMDIBuilderCreateFile");
  RESOLVE(fnDIBuilderCreateCompileUnit, "LLVMDIBuilderCreateCompileUnit");
  RESOLVE(fnDIBuilderCreateSubroutineType, "LLVMDIBuilderCreateSubroutineType");
  RESOLVE(fnDIBuilderCreateFunction, "LLVMDIBuilderCreateFunction");
  RESOLVE(fnDIBuilderCreateDebugLocation, "LLVMDIBuilderCreateDebugLocation");
  RESOLVE(fnSetCurrentDebugLocation, "LLVMSetCurrentDebugLocation");
  RESOLVE(fnMetadataAsValue, "LLVMMetadataAsValue");
  RESOLVE(fnSetSubprogram, "LLVMSetSubprogram");
  RESOLVE(fnDIBuilderCreateBasicType, "LLVMDIBuilderCreateBasicType");
  RESOLVE(fnDIBuilderCreateAutoVariable, "LLVMDIBuilderCreateAutoVariable");
  RESOLVE(fnDIBuilderCreateExpression, "LLVMDIBuilderCreateExpression");
  RESOLVE(fnDIBuilderInsertDeclareAtEnd, "LLVMDIBuilderInsertDeclareAtEnd");
  RESOLVE(fnDIBuilderInsertDbgValueAtEnd, "LLVMDIBuilderInsertDbgValueAtEnd");

  // Bitcode
  RESOLVE(fnWriteBitcodeToMemoryBuffer, "LLVMWriteBitcodeToMemoryBuffer");
  RESOLVE(fnGetBufferStart, "LLVMGetBufferStart");
  RESOLVE(fnGetBufferSize, "LLVMGetBufferSize");
  RESOLVE(fnDisposeMemoryBuffer, "LLVMDisposeMemoryBuffer");

  return llvm::Error::success();
}

#undef RESOLVE

//===----------------------------------------------------------------------===//
// Thin forwarding methods
//===----------------------------------------------------------------------===//

// Module
void LLVM70IRBuilder::setDataLayout(const char *dl) {
  fnSetDataLayout(module, dl);
}
void LLVM70IRBuilder::setTarget(const char *triple) {
  fnSetTarget(module, triple);
}
std::string LLVM70IRBuilder::printModuleToString() {
  char *s = fnPrintModuleToString(module);
  std::string result(s);
  fnDisposeMessage(s);
  return result;
}

// Types
LLVMTypeRef LLVM70IRBuilder::voidTy() { return fnVoidType(ctx); }
LLVMTypeRef LLVM70IRBuilder::i1Ty() { return fnInt1Type(ctx); }
LLVMTypeRef LLVM70IRBuilder::i8Ty() { return fnInt8Type(ctx); }
LLVMTypeRef LLVM70IRBuilder::i16Ty() { return fnInt16Type(ctx); }
LLVMTypeRef LLVM70IRBuilder::i32Ty() { return fnInt32Type(ctx); }
LLVMTypeRef LLVM70IRBuilder::i64Ty() { return fnInt64Type(ctx); }
LLVMTypeRef LLVM70IRBuilder::intTy(unsigned bits) { return fnIntType(ctx, bits); }
LLVMTypeRef LLVM70IRBuilder::halfTy() { return fnHalfType(ctx); }
LLVMTypeRef LLVM70IRBuilder::floatTy() { return fnFloatType(ctx); }
LLVMTypeRef LLVM70IRBuilder::doubleTy() { return fnDoubleType(ctx); }
LLVMTypeRef LLVM70IRBuilder::ptrTy(LLVMTypeRef e, unsigned as) {
  return fnPointerType(e, as);
}
LLVMTypeRef LLVM70IRBuilder::arrayTy(LLVMTypeRef e, unsigned n) {
  return fnArrayType(e, n);
}
LLVMTypeRef LLVM70IRBuilder::structTy(LLVMTypeRef *e, unsigned n, bool p) {
  return fnStructType(ctx, e, n, p);
}
LLVMTypeRef LLVM70IRBuilder::funcTy(LLVMTypeRef r, LLVMTypeRef *p, unsigned n,
                                   bool v) {
  return fnFunctionType(r, p, n, v);
}
LLVMTypeRef LLVM70IRBuilder::vectorTy(LLVMTypeRef e, unsigned n) {
  return fnVectorType(e, n);
}

// Functions
LLVMValueRef LLVM70IRBuilder::addFunction(const char *name, LLVMTypeRef ty) {
  return fnAddFunction(module, name, ty);
}
LLVMValueRef LLVM70IRBuilder::getNamedFunction(const char *name) {
  return fnGetNamedFunction(module, name);
}
void LLVM70IRBuilder::setLinkage(LLVMValueRef v, LLVMLinkage lk) {
  fnSetLinkage(v, lk);
}
void LLVM70IRBuilder::setFunctionCallConv(LLVMValueRef fn, unsigned cc) {
  fnSetFunctionCallConv(fn, cc);
}
LLVMValueRef LLVM70IRBuilder::getParam(LLVMValueRef fn, unsigned idx) {
  return fnGetParam(fn, idx);
}
unsigned LLVM70IRBuilder::countParams(LLVMValueRef fn) {
  return fnCountParams(fn);
}
void LLVM70IRBuilder::setValueName(LLVMValueRef v, const char *name) {
  fnSetValueName2(v, name, strlen(name));
}

// Basic blocks
LLVMBasicBlockRef LLVM70IRBuilder::appendBB(LLVMValueRef fn, const char *name) {
  return fnAppendBB(ctx, fn, name);
}
void LLVM70IRBuilder::positionAtEnd(LLVMBasicBlockRef bb) {
  fnPositionAtEnd(builder, bb);
}
LLVMBasicBlockRef LLVM70IRBuilder::getInsertBlock() {
  return fnGetInsertBlock(builder);
}

// Constants
LLVMValueRef LLVM70IRBuilder::constInt(LLVMTypeRef ty, unsigned long long val,
                                      bool se) {
  return fnConstInt(ty, val, se);
}
LLVMValueRef LLVM70IRBuilder::constReal(LLVMTypeRef ty, double val) {
  return fnConstReal(ty, val);
}
LLVMValueRef LLVM70IRBuilder::constNull(LLVMTypeRef ty) {
  return fnConstNull(ty);
}
LLVMValueRef LLVM70IRBuilder::getUndef(LLVMTypeRef ty) {
  return fnGetUndef(ty);
}
LLVMValueRef LLVM70IRBuilder::constStruct(LLVMValueRef *v, unsigned n,
                                         bool p) {
  return fnConstStruct(ctx, v, n, p);
}
LLVMValueRef LLVM70IRBuilder::constArray(LLVMTypeRef e, LLVMValueRef *v,
                                        unsigned n) {
  return fnConstArray(e, v, n);
}
LLVMValueRef LLVM70IRBuilder::constVector(LLVMValueRef *v, unsigned n) {
  return fnConstVector(v, n);
}
LLVMValueRef LLVM70IRBuilder::constBitCast(LLVMValueRef val, LLVMTypeRef ty) {
  return fnConstBitCast(val, ty);
}

// Inline asm
LLVMValueRef LLVM70IRBuilder::constInlineAsm(LLVMTypeRef fnTy,
                                            const char *asmString,
                                            const char *constraints,
                                            bool hasSideEffects,
                                            bool isAlignStack) {
  return fnConstInlineAsm(fnTy, asmString, constraints, hasSideEffects,
                          isAlignStack);
}

// Globals
LLVMValueRef LLVM70IRBuilder::addGlobal(LLVMTypeRef ty, const char *name) {
  return fnAddGlobal(module, ty, name);
}
LLVMValueRef LLVM70IRBuilder::addGlobalInAddressSpace(LLVMTypeRef ty,
                                                     const char *name,
                                                     unsigned addrSpace) {
  return fnAddGlobalInAddressSpace(module, ty, name, addrSpace);
}
LLVMValueRef LLVM70IRBuilder::getNamedGlobal(const char *name) {
  return fnGetNamedGlobal(module, name);
}
void LLVM70IRBuilder::setInitializer(LLVMValueRef g, LLVMValueRef init) {
  fnSetInitializer(g, init);
}
void LLVM70IRBuilder::setGlobalAlignment(LLVMValueRef g, unsigned bytes) {
  fnSetAlignment(g, bytes);
}
void LLVM70IRBuilder::setSection(LLVMValueRef g, const char *section) {
  fnSetSection(g, section);
}

// Arithmetic
LLVMValueRef LLVM70IRBuilder::buildAdd(LLVMValueRef l, LLVMValueRef r,
                                      const char *n) {
  return fnBuildAdd(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildSub(LLVMValueRef l, LLVMValueRef r,
                                      const char *n) {
  return fnBuildSub(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildMul(LLVMValueRef l, LLVMValueRef r,
                                      const char *n) {
  return fnBuildMul(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildSDiv(LLVMValueRef l, LLVMValueRef r,
                                       const char *n) {
  return fnBuildSDiv(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildUDiv(LLVMValueRef l, LLVMValueRef r,
                                       const char *n) {
  return fnBuildUDiv(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildSRem(LLVMValueRef l, LLVMValueRef r,
                                       const char *n) {
  return fnBuildSRem(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildURem(LLVMValueRef l, LLVMValueRef r,
                                       const char *n) {
  return fnBuildURem(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildFAdd(LLVMValueRef l, LLVMValueRef r,
                                       const char *n) {
  return fnBuildFAdd(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildFSub(LLVMValueRef l, LLVMValueRef r,
                                       const char *n) {
  return fnBuildFSub(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildFMul(LLVMValueRef l, LLVMValueRef r,
                                       const char *n) {
  return fnBuildFMul(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildFDiv(LLVMValueRef l, LLVMValueRef r,
                                       const char *n) {
  return fnBuildFDiv(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildFRem(LLVMValueRef l, LLVMValueRef r,
                                       const char *n) {
  return fnBuildFRem(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildFNeg(LLVMValueRef v, const char *n) {
  return fnBuildFNeg(builder, v, n);
}

// Logical
LLVMValueRef LLVM70IRBuilder::buildAnd(LLVMValueRef l, LLVMValueRef r,
                                      const char *n) {
  return fnBuildAnd(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildOr(LLVMValueRef l, LLVMValueRef r,
                                     const char *n) {
  return fnBuildOr(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildXor(LLVMValueRef l, LLVMValueRef r,
                                      const char *n) {
  return fnBuildXor(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildShl(LLVMValueRef l, LLVMValueRef r,
                                      const char *n) {
  return fnBuildShl(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildLShr(LLVMValueRef l, LLVMValueRef r,
                                       const char *n) {
  return fnBuildLShr(builder, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildAShr(LLVMValueRef l, LLVMValueRef r,
                                       const char *n) {
  return fnBuildAShr(builder, l, r, n);
}

// Comparison
LLVMValueRef LLVM70IRBuilder::buildICmp(LLVMIntPredicate pred, LLVMValueRef l,
                                       LLVMValueRef r, const char *n) {
  return fnBuildICmp(builder, pred, l, r, n);
}
LLVMValueRef LLVM70IRBuilder::buildFCmp(LLVMRealPredicate pred, LLVMValueRef l,
                                       LLVMValueRef r, const char *n) {
  return fnBuildFCmp(builder, pred, l, r, n);
}

// Memory
LLVMValueRef LLVM70IRBuilder::buildAlloca(LLVMTypeRef ty, const char *n) {
  return fnBuildAlloca(builder, ty, n);
}
LLVMValueRef LLVM70IRBuilder::buildArrayAlloca(LLVMTypeRef ty,
                                              LLVMValueRef numElems,
                                              const char *n) {
  return fnBuildArrayAlloca(builder, ty, numElems, n);
}
LLVMValueRef LLVM70IRBuilder::buildLoad(LLVMValueRef ptr, const char *n) {
  return fnBuildLoad(builder, ptr, n);
}
LLVMValueRef LLVM70IRBuilder::buildStore(LLVMValueRef val, LLVMValueRef ptr) {
  return fnBuildStore(builder, val, ptr);
}
LLVMValueRef LLVM70IRBuilder::buildGEP(LLVMValueRef ptr, LLVMValueRef *idx,
                                      unsigned n, const char *nm) {
  return fnBuildGEP(builder, ptr, idx, n, nm);
}
LLVMValueRef LLVM70IRBuilder::buildInBoundsGEP(LLVMValueRef ptr,
                                              LLVMValueRef *idx, unsigned n,
                                              const char *nm) {
  return fnBuildInBoundsGEP(builder, ptr, idx, n, nm);
}
LLVMValueRef LLVM70IRBuilder::buildStructGEP(LLVMValueRef ptr, unsigned idx,
                                            const char *nm) {
  return fnBuildStructGEP(builder, ptr, idx, nm);
}

// Atomics
LLVMValueRef LLVM70IRBuilder::buildAtomicRMW(LLVMAtomicRMWBinOp op,
                                            LLVMValueRef ptr, LLVMValueRef val,
                                            LLVMAtomicOrdering ordering,
                                            bool singleThread) {
  return fnBuildAtomicRMW(builder, op, ptr, val, ordering, singleThread);
}
LLVMValueRef LLVM70IRBuilder::buildAtomicCmpXchg(
    LLVMValueRef ptr, LLVMValueRef cmp, LLVMValueRef newVal,
    LLVMAtomicOrdering successOrdering, LLVMAtomicOrdering failureOrdering,
    bool singleThread) {
  return fnBuildAtomicCmpXchg(builder, ptr, cmp, newVal, successOrdering,
                              failureOrdering, singleThread);
}

// Casts
LLVMValueRef LLVM70IRBuilder::buildBitCast(LLVMValueRef v, LLVMTypeRef ty,
                                          const char *n) {
  return fnBuildBitCast(builder, v, ty, n);
}
LLVMValueRef LLVM70IRBuilder::buildAddrSpaceCast(LLVMValueRef v, LLVMTypeRef ty,
                                                const char *n) {
  return fnBuildAddrSpaceCast(builder, v, ty, n);
}
LLVMValueRef LLVM70IRBuilder::buildIntToPtr(LLVMValueRef v, LLVMTypeRef ty,
                                           const char *n) {
  return fnBuildIntToPtr(builder, v, ty, n);
}
LLVMValueRef LLVM70IRBuilder::buildPtrToInt(LLVMValueRef v, LLVMTypeRef ty,
                                           const char *n) {
  return fnBuildPtrToInt(builder, v, ty, n);
}
LLVMValueRef LLVM70IRBuilder::buildTrunc(LLVMValueRef v, LLVMTypeRef ty,
                                        const char *n) {
  return fnBuildTrunc(builder, v, ty, n);
}
LLVMValueRef LLVM70IRBuilder::buildZExt(LLVMValueRef v, LLVMTypeRef ty,
                                       const char *n) {
  return fnBuildZExt(builder, v, ty, n);
}
LLVMValueRef LLVM70IRBuilder::buildSExt(LLVMValueRef v, LLVMTypeRef ty,
                                       const char *n) {
  return fnBuildSExt(builder, v, ty, n);
}
LLVMValueRef LLVM70IRBuilder::buildFPTrunc(LLVMValueRef v, LLVMTypeRef ty,
                                          const char *n) {
  return fnBuildFPTrunc(builder, v, ty, n);
}
LLVMValueRef LLVM70IRBuilder::buildFPExt(LLVMValueRef v, LLVMTypeRef ty,
                                        const char *n) {
  return fnBuildFPExt(builder, v, ty, n);
}
LLVMValueRef LLVM70IRBuilder::buildFPToSI(LLVMValueRef v, LLVMTypeRef ty,
                                         const char *n) {
  return fnBuildFPToSI(builder, v, ty, n);
}
LLVMValueRef LLVM70IRBuilder::buildFPToUI(LLVMValueRef v, LLVMTypeRef ty,
                                         const char *n) {
  return fnBuildFPToUI(builder, v, ty, n);
}
LLVMValueRef LLVM70IRBuilder::buildSIToFP(LLVMValueRef v, LLVMTypeRef ty,
                                         const char *n) {
  return fnBuildSIToFP(builder, v, ty, n);
}
LLVMValueRef LLVM70IRBuilder::buildUIToFP(LLVMValueRef v, LLVMTypeRef ty,
                                         const char *n) {
  return fnBuildUIToFP(builder, v, ty, n);
}

// Control flow
LLVMValueRef LLVM70IRBuilder::buildCall(LLVMValueRef fn, LLVMValueRef *args,
                                       unsigned n, const char *nm) {
  return fnBuildCall(builder, fn, args, n, nm);
}
LLVMValueRef LLVM70IRBuilder::buildRet(LLVMValueRef val) {
  return fnBuildRet(builder, val);
}
LLVMValueRef LLVM70IRBuilder::buildRetVoid() {
  return fnBuildRetVoid(builder);
}
LLVMValueRef LLVM70IRBuilder::buildBr(LLVMBasicBlockRef dest) {
  return fnBuildBr(builder, dest);
}
LLVMValueRef LLVM70IRBuilder::buildCondBr(LLVMValueRef cond,
                                         LLVMBasicBlockRef t,
                                         LLVMBasicBlockRef f) {
  return fnBuildCondBr(builder, cond, t, f);
}
LLVMValueRef LLVM70IRBuilder::buildPhi(LLVMTypeRef ty, const char *n) {
  return fnBuildPhi(builder, ty, n);
}
void LLVM70IRBuilder::addIncoming(LLVMValueRef phi, LLVMValueRef *vals,
                                 LLVMBasicBlockRef *bbs, unsigned count) {
  fnAddIncoming(phi, vals, bbs, count);
}
LLVMValueRef LLVM70IRBuilder::buildSelect(LLVMValueRef c, LLVMValueRef t,
                                         LLVMValueRef f, const char *n) {
  return fnBuildSelect(builder, c, t, f, n);
}
LLVMValueRef LLVM70IRBuilder::buildSwitch(LLVMValueRef val,
                                         LLVMBasicBlockRef elseBB,
                                         unsigned numCases) {
  return fnBuildSwitch(builder, val, elseBB, numCases);
}
void LLVM70IRBuilder::addCase(LLVMValueRef switchInst, LLVMValueRef onVal,
                             LLVMBasicBlockRef dest) {
  fnAddCase(switchInst, onVal, dest);
}
LLVMValueRef LLVM70IRBuilder::buildUnreachable() {
  return fnBuildUnreachable(builder);
}

// Aggregate
LLVMValueRef LLVM70IRBuilder::buildExtractValue(LLVMValueRef agg, unsigned idx,
                                               const char *n) {
  return fnBuildExtractValue(builder, agg, idx, n);
}
LLVMValueRef LLVM70IRBuilder::buildInsertValue(LLVMValueRef agg,
                                              LLVMValueRef val, unsigned idx,
                                              const char *n) {
  return fnBuildInsertValue(builder, agg, val, idx, n);
}

// Vector
LLVMValueRef LLVM70IRBuilder::buildExtractElement(LLVMValueRef vec,
                                                 LLVMValueRef idx,
                                                 const char *n) {
  return fnBuildExtractElement(builder, vec, idx, n);
}
LLVMValueRef LLVM70IRBuilder::buildInsertElement(LLVMValueRef vec,
                                                LLVMValueRef val,
                                                LLVMValueRef idx,
                                                const char *n) {
  return fnBuildInsertElement(builder, vec, val, idx, n);
}

// Metadata
LLVMValueRef LLVM70IRBuilder::mdString(const char *s, unsigned len) {
  return fnMDString(ctx, s, len);
}
LLVMValueRef LLVM70IRBuilder::mdNode(LLVMValueRef *vals, unsigned count) {
  return fnMDNode(ctx, vals, count);
}
void LLVM70IRBuilder::addNamedMetadataOperand(const char *name,
                                             LLVMValueRef val) {
  fnAddNamedMetadataOperand(module, name, val);
}

// Debug info
void LLVM70IRBuilder::initDebugInfo() {
  diBuilder = fnCreateDIBuilder(module);
}
void LLVM70IRBuilder::finalizeDebugInfo() {
  if (diBuilder)
    fnDIBuilderFinalize(diBuilder);
}
LLVMMetadataRef LLVM70IRBuilder::createDIFile(const char *filename,
                                             size_t filenameLen,
                                             const char *directory,
                                             size_t directoryLen) {
  return fnDIBuilderCreateFile(diBuilder, filename, filenameLen, directory,
                               directoryLen);
}
LLVMMetadataRef LLVM70IRBuilder::createDICompileUnit(LLVMMetadataRef file,
                                                     bool fullDebug) {
  // DebugDirectivesOnly (3) emits .file/.loc without .target ..., debug
  auto emissionKind = fullDebug ? LLVMDWARFEmissionFull
                                : static_cast<LLVMDWARFEmissionKind>(3);
  return fnDIBuilderCreateCompileUnit(
      diBuilder, LLVMDWARFSourceLanguageC, file,
      "llvm70", 6, /*isOptimized=*/false, /*Flags=*/"", 0,
      /*RuntimeVer=*/0, /*SplitName=*/"", 0,
      emissionKind, /*DWOId=*/0,
      /*SplitDebugInlining=*/false, /*DebugInfoForProfiling=*/false);
}
LLVMMetadataRef
LLVM70IRBuilder::createDISubroutineType(LLVMMetadataRef file) {
  return fnDIBuilderCreateSubroutineType(diBuilder, file, nullptr, 0,
                                         LLVMDIFlagZero);
}
LLVMMetadataRef LLVM70IRBuilder::createDIFunction(LLVMMetadataRef scope,
                                                 const char *name,
                                                 size_t nameLen,
                                                 LLVMMetadataRef file,
                                                 unsigned lineNo,
                                                 LLVMMetadataRef type) {
  return fnDIBuilderCreateFunction(
      diBuilder, scope, name, nameLen, name, nameLen, file, lineNo, type,
      /*IsLocalToUnit=*/false, /*IsDefinition=*/true,
      /*ScopeLine=*/lineNo, LLVMDIFlagZero, /*IsOptimized=*/false);
}
void LLVM70IRBuilder::setSubprogram(LLVMValueRef fn, LLVMMetadataRef sp) {
  fnSetSubprogram(fn, sp);
}
void LLVM70IRBuilder::setDebugLocation(unsigned line, unsigned col,
                                      LLVMMetadataRef scope) {
  LLVMMetadataRef loc =
      fnDIBuilderCreateDebugLocation(ctx, line, col, scope, nullptr);
  LLVMValueRef locVal = fnMetadataAsValue(ctx, loc);
  fnSetCurrentDebugLocation(builder, locVal);
}
void LLVM70IRBuilder::clearDebugLocation() {
  fnSetCurrentDebugLocation(builder, nullptr);
}
LLVMMetadataRef LLVM70IRBuilder::createDebugLocation(unsigned line, unsigned col,
                                                     LLVMMetadataRef scope) {
  return fnDIBuilderCreateDebugLocation(ctx, line, col, scope, nullptr);
}
LLVMMetadataRef LLVM70IRBuilder::createDIBasicType(const char *name,
                                                   size_t nameLen,
                                                   uint64_t sizeInBits,
                                                   LLVMDWARFTypeEncoding enc) {
  return fnDIBuilderCreateBasicType(diBuilder, name, nameLen, sizeInBits, enc);
}
LLVMMetadataRef LLVM70IRBuilder::createDIAutoVariable(
    LLVMMetadataRef scope, const char *name, size_t nameLen,
    LLVMMetadataRef file, unsigned lineNo, LLVMMetadataRef type,
    uint32_t alignInBits) {
  return fnDIBuilderCreateAutoVariable(diBuilder, scope, name, nameLen, file,
                                       lineNo, type, /*AlwaysPreserve=*/false,
                                       LLVMDIFlagZero, alignInBits);
}
LLVMMetadataRef LLVM70IRBuilder::createDIExpression(int64_t *ops, size_t count) {
  return fnDIBuilderCreateExpression(diBuilder, ops, count);
}
LLVMValueRef LLVM70IRBuilder::insertDbgDeclare(LLVMValueRef storage,
                                               LLVMMetadataRef varInfo,
                                               LLVMMetadataRef expr,
                                               LLVMMetadataRef debugLoc) {
  return fnDIBuilderInsertDeclareAtEnd(diBuilder, storage, varInfo, expr,
                                       debugLoc, getInsertBlock());
}
LLVMValueRef LLVM70IRBuilder::insertDbgValue(LLVMValueRef val,
                                             LLVMMetadataRef varInfo,
                                             LLVMMetadataRef expr,
                                             LLVMMetadataRef debugLoc) {
  return fnDIBuilderInsertDbgValueAtEnd(diBuilder, val, varInfo, expr,
                                        debugLoc, getInsertBlock());
}

// Bitcode
LLVMMemoryBufferRef LLVM70IRBuilder::writeBitcodeToMemoryBuffer() {
  return fnWriteBitcodeToMemoryBuffer(module);
}
const char *LLVM70IRBuilder::getBufferStart(LLVMMemoryBufferRef buf) {
  return fnGetBufferStart(buf);
}
size_t LLVM70IRBuilder::getBufferSize(LLVMMemoryBufferRef buf) {
  return fnGetBufferSize(buf);
}
void LLVM70IRBuilder::disposeMemoryBuffer(LLVMMemoryBufferRef buf) {
  fnDisposeMemoryBuffer(buf);
}
