/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "llvm_downgrade.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#include <cstring>
#include <string_view>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// LLVM C API types and constants (obtained from llvm-c/Core.h).
// Redeclared here because we dlopen libMLIRPythonCAPI.so at runtime rather
// than linking against LLVM headers.
// ---------------------------------------------------------------------------

typedef void* LLVMModuleRef;
typedef void* LLVMContextRef;
typedef void* LLVMTypeRef;
typedef void* LLVMValueRef;
typedef void* LLVMBuilderRef;
typedef void* LLVMAttributeRef;
typedef void* LLVMMetadataRef;
typedef void* LLVMBasicBlockRef;
typedef void* LLVMUseRef;
typedef void* LLVMMemoryBufferRef;

enum LLVMAtomicRMWBinOp_ {
    LLVMAtomicRMWBinOpFAdd = 11,
    LLVMAtomicRMWBinOpFMax = 13,
    LLVMAtomicRMWBinOpFMin = 14,
    LLVMAtomicRMWBinOpFMaximum = 19,
    LLVMAtomicRMWBinOpFMinimum = 20,
};

enum LLVMModuleFlagBehavior_ {
    LLVMModuleFlagBehaviorWarning = 2,
};

enum LLVMCallConv_ {
    LLVMPTXKernelCallConv = 71,
};

enum LLVMLinkage_ {
    LLVMAppendingLinkage = 7,
};

enum LLVMTypeKind_ {
    LLVMVoidTypeKind = 0,
};

// ---------------------------------------------------------------------------
// Function pointer types
// ---------------------------------------------------------------------------

// Required symbols from libMLIRPythonCAPI.so.
#define LLVM_CAPI_REQUIRED(X) \
    X(void, LLVMContextDispose, (LLVMContextRef)) \
    X(void, LLVMDisposeModule, (LLVMModuleRef)) \
    X(char*, LLVMPrintModuleToString, (LLVMModuleRef)) \
    X(void, LLVMDisposeMessage, (char*)) \
    /* Iteration */ \
    X(LLVMValueRef, LLVMGetFirstFunction, (LLVMModuleRef)) \
    X(LLVMValueRef, LLVMGetNextFunction, (LLVMValueRef)) \
    X(LLVMBasicBlockRef, LLVMGetFirstBasicBlock, (LLVMValueRef)) \
    X(LLVMBasicBlockRef, LLVMGetNextBasicBlock, (LLVMBasicBlockRef)) \
    X(LLVMValueRef, LLVMGetFirstInstruction, (LLVMBasicBlockRef)) \
    X(LLVMValueRef, LLVMGetNextInstruction, (LLVMValueRef)) \
    /* Functions */ \
    X(LLVMValueRef, LLVMGetNamedFunction, (LLVMModuleRef, const char*)) \
    X(LLVMValueRef, LLVMAddFunction, (LLVMModuleRef, const char*, LLVMTypeRef)) \
    X(unsigned, LLVMCountParams, (LLVMValueRef)) \
    X(unsigned, LLVMGetFunctionCallConv, (LLVMValueRef)) \
    X(void, LLVMSetLinkage, (LLVMValueRef, unsigned)) \
    X(LLVMValueRef, LLVMGetCalledValue, (LLVMValueRef)) \
    /* Attributes */ \
    X(unsigned, LLVMGetAttributeCountAtIndex, (LLVMValueRef, unsigned)) \
    X(void, LLVMGetAttributesAtIndex, (LLVMValueRef, unsigned, LLVMAttributeRef*)) \
    X(int, LLVMIsEnumAttribute, (LLVMAttributeRef)) \
    X(int, LLVMIsStringAttribute, (LLVMAttributeRef)) \
    X(unsigned, LLVMGetEnumAttributeKind, (LLVMAttributeRef)) \
    X(const char*, LLVMGetStringAttributeKind, (LLVMAttributeRef, unsigned*)) \
    X(void, LLVMRemoveEnumAttributeAtIndex, (LLVMValueRef, unsigned, unsigned)) \
    X(unsigned, LLVMGetEnumAttributeKindForName, (const char*, size_t)) \
    /* Types */ \
    X(LLVMTypeRef, LLVMVoidTypeInContext, (LLVMContextRef)) \
    X(LLVMTypeRef, LLVMInt1TypeInContext, (LLVMContextRef)) \
    X(LLVMTypeRef, LLVMInt32TypeInContext, (LLVMContextRef)) \
    X(LLVMTypeRef, LLVMFloatTypeInContext, (LLVMContextRef)) \
    X(LLVMTypeRef, LLVMDoubleTypeInContext, (LLVMContextRef)) \
    X(LLVMTypeRef, LLVMHalfTypeInContext, (LLVMContextRef)) \
    X(LLVMTypeRef, LLVMBFloatTypeInContext, (LLVMContextRef)) \
    X(LLVMTypeRef, LLVMFunctionType, (LLVMTypeRef, LLVMTypeRef*, unsigned, int)) \
    X(LLVMTypeRef, LLVMPointerTypeInContext, (LLVMContextRef, unsigned)) \
    X(LLVMTypeRef, LLVMTypeOf, (LLVMValueRef)) \
    X(unsigned, LLVMGetTypeKind, (LLVMTypeRef)) \
    X(LLVMTypeRef, LLVMArrayType2, (LLVMTypeRef, uint64_t)) \
    X(LLVMTypeRef, LLVMGetReturnType, (LLVMTypeRef)) \
    /* Values */ \
    X(void, LLVMReplaceAllUsesWith, (LLVMValueRef, LLVMValueRef)) \
    X(LLVMValueRef, LLVMGetOperand, (LLVMValueRef, unsigned)) \
    X(int, LLVMGetNumOperands, (LLVMValueRef)) \
    X(LLVMValueRef, LLVMConstInt, (LLVMTypeRef, unsigned long long, int)) \
    /* Instructions */ \
    X(void, LLVMInstructionEraseFromParent, (LLVMValueRef)) \
    /* AtomicRMW */ \
    X(int, LLVMGetAtomicRMWBinOp, (LLVMValueRef)) \
    X(void, LLVMSetAtomicRMWBinOp, (LLVMValueRef, int)) \
    /* Builder */ \
    X(LLVMBuilderRef, LLVMCreateBuilderInContext, (LLVMContextRef)) \
    X(void, LLVMPositionBuilderBefore, (LLVMBuilderRef, LLVMValueRef)) \
    X(void, LLVMDisposeBuilder, (LLVMBuilderRef)) \
    X(LLVMValueRef, LLVMBuildCall2, (LLVMBuilderRef, LLVMTypeRef, LLVMValueRef, LLVMValueRef*, unsigned, const char*)) \
    X(LLVMValueRef, LLVMBuildZExt, (LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, const char*)) \
    X(LLVMValueRef, LLVMBuildTrunc, (LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, const char*)) \
    X(LLVMValueRef, LLVMBuildFPExt, (LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, const char*)) \
    X(LLVMValueRef, LLVMBuildFPTrunc, (LLVMBuilderRef, LLVMValueRef, LLVMTypeRef, const char*)) \
    /* Inline asm */ \
    X(LLVMValueRef, LLVMGetInlineAsm, (LLVMTypeRef, const char*, size_t, const char*, size_t, int, int, int, int)) \
    /* Uses */ \
    X(LLVMUseRef, LLVMGetFirstUse, (LLVMValueRef)) \
    X(LLVMUseRef, LLVMGetNextUse, (LLVMUseRef)) \
    X(LLVMValueRef, LLVMGetUser, (LLVMUseRef)) \
    /* Metadata */ \
    X(LLVMMetadataRef, LLVMMDStringInContext2, (LLVMContextRef, const char*, size_t)) \
    X(LLVMMetadataRef, LLVMMDNodeInContext2, (LLVMContextRef, LLVMMetadataRef*, size_t)) \
    X(LLVMMetadataRef, LLVMValueAsMetadata, (LLVMValueRef)) \
    X(LLVMValueRef, LLVMMetadataAsValue, (LLVMContextRef, LLVMMetadataRef)) \
    X(void, LLVMAddNamedMetadataOperand, (LLVMModuleRef, const char*, LLVMValueRef)) \
    X(unsigned, LLVMGetNamedMetadataNumOperands, (LLVMModuleRef, const char*)) \
    X(void, LLVMGetNamedMetadataOperands, (LLVMModuleRef, const char*, LLVMValueRef*)) \
    /* Globals */ \
    X(LLVMValueRef, LLVMAddGlobal, (LLVMModuleRef, LLVMTypeRef, const char*)) \
    X(void, LLVMSetInitializer, (LLVMValueRef, LLVMValueRef)) \
    X(void, LLVMSetSection, (LLVMValueRef, const char*)) \
    X(LLVMValueRef, LLVMConstArray2, (LLVMTypeRef, LLVMValueRef*, uint64_t)) \
    /* MDNode inspection */ \
    X(unsigned, LLVMGetMDNodeNumOperands, (LLVMValueRef)) \
    X(void, LLVMGetMDNodeOperands, (LLVMValueRef, LLVMValueRef*)) \
    X(void, LLVMReplaceMDNodeOperandWith, (LLVMValueRef, unsigned, LLVMMetadataRef)) \
    X(const char*, LLVMGetMDString, (LLVMValueRef, unsigned*)) \
    X(unsigned long long, LLVMConstIntGetZExtValue, (LLVMValueRef)) \
    /* Value type checks */ \
    X(LLVMValueRef, LLVMIsACallInst, (LLVMValueRef)) \
    X(LLVMValueRef, LLVMIsAAtomicRMWInst, (LLVMValueRef)) \
    X(LLVMValueRef, LLVMIsAConstantInt, (LLVMValueRef)) \
    X(unsigned, LLVMGetPointerAddressSpace, (LLVMTypeRef)) \
    X(void, LLVMDeleteFunction, (LLVMValueRef))

// Optional symbols -- bitcode output.
// Requires LLVMBitWriter linked into libMLIRPythonCAPI.so (TODO: upstream change).
// When unavailable, we fall back to LLVMPrintModuleToString to produce text IR instead.
#define LLVM_CAPI_OPTIONAL(X) \
    X(LLVMMemoryBufferRef, LLVMWriteBitcodeToMemoryBuffer, (LLVMModuleRef)) \
    X(const char*, LLVMGetBufferStart, (LLVMMemoryBufferRef)) \
    X(size_t, LLVMGetBufferSize, (LLVMMemoryBufferRef)) \
    X(void, LLVMDisposeMemoryBuffer, (LLVMMemoryBufferRef))

// Declare function pointer globals
#define DECLARE_FN(ret, name, args) \
    using name##_fn = ret(*)args; \
    static name##_fn g_##name;

LLVM_CAPI_REQUIRED(DECLARE_FN)
LLVM_CAPI_OPTIONAL(DECLARE_FN)
#undef DECLARE_FN

static void* g_mlir_lib_handle;

static void* load_symbol(void* handle, const char* name) {
#ifdef _WIN32
    return reinterpret_cast<void*>(
        GetProcAddress(reinterpret_cast<HMODULE>(handle), name));
#else
    return dlsym(handle, name);
#endif
}

Status load_mlir_capi(const char* lib_path) {
    if (g_mlir_lib_handle)
        return OK;

#ifdef _WIN32
    g_mlir_lib_handle = LoadLibraryA(lib_path);
    if (!g_mlir_lib_handle)
        return raise(PyExc_RuntimeError, "Failed to load %s: error %lu",
                     lib_path, GetLastError());
#else
    g_mlir_lib_handle = dlopen(lib_path, RTLD_NOW | RTLD_GLOBAL);
    if (!g_mlir_lib_handle)
        return raise(PyExc_RuntimeError, "Failed to load %s: %s",
                     lib_path, dlerror());
#endif

#define LOAD_REQUIRED(ret, name, args) \
    g_##name = reinterpret_cast<name##_fn>(load_symbol(g_mlir_lib_handle, #name)); \
    if (!g_##name) \
        return raise(PyExc_RuntimeError, \
                     "Symbol '%s' not found in %s", #name, lib_path);

    LLVM_CAPI_REQUIRED(LOAD_REQUIRED)
#undef LOAD_REQUIRED

#define LOAD_OPTIONAL(ret, name, args) \
    g_##name = reinterpret_cast<name##_fn>(load_symbol(g_mlir_lib_handle, #name));

    LLVM_CAPI_OPTIONAL(LOAD_OPTIONAL)
#undef LOAD_OPTIONAL

    return OK;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<LLVMValueRef> collect_call_users(LLVMValueRef fn) {
    std::vector<LLVMValueRef> calls;
    for (LLVMUseRef u = g_LLVMGetFirstUse(fn); u; u = g_LLVMGetNextUse(u)) {
        LLVMValueRef user = g_LLVMGetUser(u);
        if (g_LLVMIsACallInst(user) && g_LLVMGetCalledValue(user) == fn)
            calls.push_back(user);
    }
    return calls;
}

static LLVMValueRef get_or_add_function(LLVMModuleRef mod, const char* name,
                                        LLVMTypeRef ty) {
    LLVMValueRef fn = g_LLVMGetNamedFunction(mod, name);
    if (!fn)
        fn = g_LLVMAddFunction(mod, name, ty);
    return fn;
}

// Prefer bitcode (faster, no re-parsing by libnvvm, requires
// LLVMBitWriter linked upstream). Fall back to text IR if unavailable.
static PyObject* serialize_module(LLVMModuleRef mod) {
    if (g_LLVMWriteBitcodeToMemoryBuffer) {
        LLVMMemoryBufferRef buf = g_LLVMWriteBitcodeToMemoryBuffer(mod);
        if (!buf) {
            PyErr_SetString(PyExc_RuntimeError,
                            "LLVMWriteBitcodeToMemoryBuffer failed");
            return nullptr;
        }
        PyObject* result = PyBytes_FromStringAndSize(
            g_LLVMGetBufferStart(buf),
            static_cast<Py_ssize_t>(g_LLVMGetBufferSize(buf)));
        g_LLVMDisposeMemoryBuffer(buf);
        return result;
    }
    char* ir_text = g_LLVMPrintModuleToString(mod);
    PyObject* result = PyBytes_FromString(ir_text);
    g_LLVMDisposeMessage(ir_text);
    return result;
}

// Replace all calls to a named intrinsic with inline PTX assembly.
static void replace_intrinsic_with_asm(
        LLVMModuleRef mod, LLVMContextRef ctx,
        const char* intrinsic_name, LLVMTypeRef fn_ty,
        const char* asm_str, const char* constraints,
        bool has_side_effects) {
    LLVMValueRef old_fn = g_LLVMGetNamedFunction(mod, intrinsic_name);
    if (!old_fn) return;

    LLVMValueRef inline_asm = g_LLVMGetInlineAsm(
        fn_ty, asm_str, strlen(asm_str),
        constraints, strlen(constraints),
        has_side_effects, /*isAlignStack=*/0,
        /*dialect=*/0, /*canThrow=*/0);

    bool returns_value =
        g_LLVMGetTypeKind(g_LLVMGetReturnType(fn_ty)) != LLVMVoidTypeKind;

    LLVMBuilderRef builder = g_LLVMCreateBuilderInContext(ctx);
    for (LLVMValueRef call : collect_call_users(old_fn)) {
        g_LLVMPositionBuilderBefore(builder, call);
        unsigned total_operands = g_LLVMGetNumOperands(call);
        unsigned num_args = (total_operands > 0 ? total_operands - 1 : 0);
        std::vector<LLVMValueRef> args(num_args);
        for (unsigned i = 0; i < num_args; ++i)
            args[i] = g_LLVMGetOperand(call, i);
        LLVMValueRef result = g_LLVMBuildCall2(
            builder, fn_ty, inline_asm,
            num_args ? args.data() : nullptr, num_args, "");
        if (returns_value)
            g_LLVMReplaceAllUsesWith(call, result);
        g_LLVMInstructionEraseFromParent(call);
    }
    g_LLVMDisposeBuilder(builder);
}

// ===================================================================
// adapt_for_libnvvm -- semantic adaptations for libnvvm
//
// These patches bridge the gap between MLIR's LLVM IR output meant for
// the NVPTX backend and the LLVM IR that libnvvm actually accepts.
// They are required regardless of CTK version because MLIR and libnvvm
// disagree on intrinsic names, instruction legality, metadata requirements.
// ===================================================================

// Barrier sync: llvm.nvvm.barrier.cta.sync.aligned.all(i32) ->
//   barrier0()     if arg is constant 0
//   bar.sync(i32)  otherwise
static void adapt_barrier_sync(LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMValueRef old_fn = g_LLVMGetNamedFunction(
        mod, "llvm.nvvm.barrier.cta.sync.aligned.all");
    if (!old_fn) return;

    LLVMTypeRef void_ty = g_LLVMVoidTypeInContext(ctx);
    LLVMTypeRef i32_ty = g_LLVMInt32TypeInContext(ctx);

    LLVMTypeRef barrier0_fn_ty = g_LLVMFunctionType(void_ty, nullptr, 0, 0);
    LLVMValueRef barrier0_fn = get_or_add_function(
        mod, "llvm.nvvm.barrier0", barrier0_fn_ty);

    LLVMTypeRef bar_sync_param = i32_ty;
    LLVMTypeRef bar_sync_fn_ty = g_LLVMFunctionType(
        void_ty, &bar_sync_param, 1, 0);
    LLVMValueRef bar_sync_fn = get_or_add_function(
        mod, "llvm.nvvm.bar.sync", bar_sync_fn_ty);

    LLVMBuilderRef builder = g_LLVMCreateBuilderInContext(ctx);
    auto calls = collect_call_users(old_fn);
    for (LLVMValueRef call : calls) {
        g_LLVMPositionBuilderBefore(builder, call);
        LLVMValueRef arg = g_LLVMGetOperand(call, 0);
        bool is_zero = g_LLVMIsAConstantInt(arg) &&
                       g_LLVMConstIntGetZExtValue(arg) == 0;
        if (is_zero) {
            g_LLVMBuildCall2(builder, barrier0_fn_ty, barrier0_fn,
                             nullptr, 0, "");
        } else {
            g_LLVMBuildCall2(builder, bar_sync_fn_ty, bar_sync_fn,
                             &arg, 1, "");
        }
        g_LLVMInstructionEraseFromParent(call);
    }
    g_LLVMDisposeBuilder(builder);
}

// Barrier reduction:
//   llvm.nvvm.barrier.cta.red.{and,or,popc}.aligned.all(i32, i1) ->
//   llvm.nvvm.barrier0.{and,or,popc}(i32)
//   with i1->i32 zext on input, and i32->i1 trunc on output for and/or
static void adapt_barrier_reduction(LLVMModuleRef mod, LLVMContextRef ctx) {
    const char* ops[] = {"and", "or", "popc"};

    LLVMTypeRef i1_ty = g_LLVMInt1TypeInContext(ctx);
    LLVMTypeRef i32_ty = g_LLVMInt32TypeInContext(ctx);

    LLVMTypeRef new_param = i32_ty;
    LLVMTypeRef new_fn_ty = g_LLVMFunctionType(i32_ty, &new_param, 1, 0);

    for (const char* op : ops) {
        char old_name[128];
        snprintf(old_name, sizeof(old_name),
                 "llvm.nvvm.barrier.cta.red.%s.aligned.all", op);
        char new_name[128];
        snprintf(new_name, sizeof(new_name), "llvm.nvvm.barrier0.%s", op);

        LLVMValueRef old_fn = g_LLVMGetNamedFunction(mod, old_name);
        if (!old_fn) continue;

        LLVMValueRef new_fn = get_or_add_function(mod, new_name, new_fn_ty);
        bool returns_i1 = (strcmp(op, "and") == 0 || strcmp(op, "or") == 0);

        LLVMBuilderRef builder = g_LLVMCreateBuilderInContext(ctx);
        auto calls = collect_call_users(old_fn);
        for (LLVMValueRef call : calls) {
            g_LLVMPositionBuilderBefore(builder, call);
            LLVMValueRef pred = g_LLVMGetOperand(call, 1);
            LLVMValueRef pred_i32 = g_LLVMBuildZExt(builder, pred, i32_ty, "");
            LLVMValueRef new_call = g_LLVMBuildCall2(
                builder, new_fn_ty, new_fn, &pred_i32, 1, "");
            if (returns_i1) {
                LLVMValueRef result_i1 = g_LLVMBuildTrunc(
                    builder, new_call, i1_ty, "");
                g_LLVMReplaceAllUsesWith(call, result_i1);
            } else {
                g_LLVMReplaceAllUsesWith(call, new_call);
            }
            g_LLVMInstructionEraseFromParent(call);
        }
        g_LLVMDisposeBuilder(builder);
    }
}

// Unsupported intrinsics replaced with inline PTX assembly:
//   llvm.nvvm.nanosleep(i32)         -> nanosleep.u32
//   llvm.stacksave.p0()              -> stacksave.u64
//   llvm.stackrestore.p0(ptr)        -> stackrestore.u64
//   llvm.nvvm.mapa(ptr, i32)         -> mapa.u64
static void adapt_inline_asm_intrinsics(LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMTypeRef void_ty = g_LLVMVoidTypeInContext(ctx);
    LLVMTypeRef i32_ty = g_LLVMInt32TypeInContext(ctx);
    LLVMTypeRef ptr_ty = g_LLVMPointerTypeInContext(ctx, 0);

    LLVMTypeRef ns_param = i32_ty;
    replace_intrinsic_with_asm(mod, ctx, "llvm.nvvm.nanosleep",
        g_LLVMFunctionType(void_ty, &ns_param, 1, 0),
        "nanosleep.u32 $0;", "r", true);

    replace_intrinsic_with_asm(mod, ctx, "llvm.stacksave.p0",
        g_LLVMFunctionType(ptr_ty, nullptr, 0, 0),
        "stacksave.u64 $0;", "=l", true);

    LLVMTypeRef sr_param = ptr_ty;
    replace_intrinsic_with_asm(mod, ctx, "llvm.stackrestore.p0",
        g_LLVMFunctionType(void_ty, &sr_param, 1, 0),
        "stackrestore.u64 $0;", "l", true);

    LLVMTypeRef mapa_params[] = {ptr_ty, i32_ty};
    replace_intrinsic_with_asm(mod, ctx, "llvm.nvvm.mapa",
        g_LLVMFunctionType(ptr_ty, mapa_params, 2, 0),
        "mapa.u64 $0, $1, $2;", "=l,l,r", false);
}

// atomicrmw fminimum/fmaximum -> fmin/fmax (binop enum rename)
// atomicrmw fadd float/double -> inline PTX asm
static void adapt_atomicrmw(LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMTypeRef float_ty = g_LLVMFloatTypeInContext(ctx);
    LLVMTypeRef double_ty = g_LLVMDoubleTypeInContext(ctx);
    LLVMBuilderRef builder = g_LLVMCreateBuilderInContext(ctx);

    auto lower_fadd = [&](LLVMValueRef inst) {
        LLVMTypeRef val_ty = g_LLVMTypeOf(inst);
        if (val_ty != float_ty && val_ty != double_ty) return;

        LLVMValueRef ptr = g_LLVMGetOperand(inst, 0);
        LLVMValueRef val = g_LLVMGetOperand(inst, 1);
        LLVMTypeRef ptr_ty = g_LLVMTypeOf(ptr);

        bool is_f32 = (val_ty == float_ty);
        const char* constraints = is_f32 ? "=f,l,f" : "=d,l,d";
        unsigned addrspace = g_LLVMGetPointerAddressSpace(ptr_ty);
        const char* space = (addrspace == 3) ? "shared."
                          : (addrspace == 1) ? "global." : "";

        char asm_str[128];
        snprintf(asm_str, sizeof(asm_str),
                 "atom.%sadd.%s $0, [$1], $2;", space, is_f32 ? "f32" : "f64");

        LLVMTypeRef asm_params[] = {ptr_ty, val_ty};
        LLVMTypeRef asm_fn_ty = g_LLVMFunctionType(val_ty, asm_params, 2, 0);
        LLVMValueRef inline_asm = g_LLVMGetInlineAsm(
            asm_fn_ty, asm_str, strlen(asm_str),
            constraints, strlen(constraints), 1, 0, 0, 0);

        g_LLVMPositionBuilderBefore(builder, inst);
        LLVMValueRef args[] = {ptr, val};
        LLVMValueRef asm_call = g_LLVMBuildCall2(
            builder, asm_fn_ty, inline_asm, args, 2, "");
        g_LLVMReplaceAllUsesWith(inst, asm_call);
        g_LLVMInstructionEraseFromParent(inst);
    };

    for (LLVMValueRef fn = g_LLVMGetFirstFunction(mod); fn;
         fn = g_LLVMGetNextFunction(fn)) {
        for (LLVMBasicBlockRef bb = g_LLVMGetFirstBasicBlock(fn); bb;
             bb = g_LLVMGetNextBasicBlock(bb)) {
            LLVMValueRef inst = g_LLVMGetFirstInstruction(bb);
            while (inst) {
                LLVMValueRef next = g_LLVMGetNextInstruction(inst);
                if (g_LLVMIsAAtomicRMWInst(inst)) {
                    int binop = g_LLVMGetAtomicRMWBinOp(inst);
                    if (binop == LLVMAtomicRMWBinOpFMinimum)
                        g_LLVMSetAtomicRMWBinOp(inst, LLVMAtomicRMWBinOpFMin);
                    else if (binop == LLVMAtomicRMWBinOpFMaximum)
                        g_LLVMSetAtomicRMWBinOp(inst, LLVMAtomicRMWBinOpFMax);
                    else if (binop == LLVMAtomicRMWBinOpFAdd)
                        lower_fadd(inst);
                }
                inst = next;
            }
        }
    }
    g_LLVMDisposeBuilder(builder);
}

// llvm.trunc.{f32,f64,f16,bf16} -> libdevice __nv_trunc/__nv_truncf calls
static void adapt_trunc(LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMTypeRef float_ty = g_LLVMFloatTypeInContext(ctx);
    LLVMTypeRef double_ty = g_LLVMDoubleTypeInContext(ctx);
    LLVMTypeRef f32_fn_ty = g_LLVMFunctionType(float_ty, &float_ty, 1, 0);
    LLVMTypeRef f64_fn_ty = g_LLVMFunctionType(double_ty, &double_ty, 1, 0);

    struct { const char* intrinsic; const char* libdevice; LLVMTypeRef ty; } mappings[] = {
        {"llvm.trunc.f64",  "__nv_trunc",  double_ty},
        {"llvm.trunc.f32",  "__nv_truncf", float_ty},
        {"llvm.trunc.f16",  "__nv_truncf", g_LLVMHalfTypeInContext(ctx)},
        {"llvm.trunc.bf16", "__nv_truncf", g_LLVMBFloatTypeInContext(ctx)},
    };

    LLVMBuilderRef builder = g_LLVMCreateBuilderInContext(ctx);
    for (auto& m : mappings) {
        LLVMValueRef old_fn = g_LLVMGetNamedFunction(mod, m.intrinsic);
        if (!old_fn) continue;

        bool is_f64 = (m.ty == double_ty);
        bool promote = (m.ty != float_ty && !is_f64);
        LLVMTypeRef lib_fn_ty = is_f64 ? f64_fn_ty : f32_fn_ty;
        LLVMValueRef lib_fn = get_or_add_function(mod, m.libdevice, lib_fn_ty);

        for (LLVMValueRef call : collect_call_users(old_fn)) {
            g_LLVMPositionBuilderBefore(builder, call);
            LLVMValueRef arg = g_LLVMGetOperand(call, 0);
            if (promote)
                arg = g_LLVMBuildFPExt(builder, arg, float_ty, "");
            LLVMValueRef result = g_LLVMBuildCall2(
                builder, lib_fn_ty, lib_fn, &arg, 1, "");
            if (promote)
                result = g_LLVMBuildFPTrunc(builder, result, m.ty, "");
            g_LLVMReplaceAllUsesWith(call, result);
            g_LLVMInstructionEraseFromParent(call);
        }
    }
    g_LLVMDisposeBuilder(builder);
}

// Add !nvvm.annotations {ptr @fn, !"kernel", i32 1} and @llvm.used for each
// ptx_kernel function in the module.
static void adapt_nvvm_annotations(LLVMModuleRef mod, LLVMContextRef ctx) {
    if (g_LLVMGetNamedMetadataNumOperands(mod, "nvvm.annotations") > 0)
        return;

    std::vector<LLVMValueRef> kernel_fns;
    for (LLVMValueRef fn = g_LLVMGetFirstFunction(mod); fn;
         fn = g_LLVMGetNextFunction(fn)) {
        if (g_LLVMGetFunctionCallConv(fn) == LLVMPTXKernelCallConv)
            kernel_fns.push_back(fn);
    }
    if (kernel_fns.empty()) return;

    LLVMTypeRef i32_ty = g_LLVMInt32TypeInContext(ctx);
    LLVMTypeRef ptr_ty = g_LLVMPointerTypeInContext(ctx, 0);
    LLVMMetadataRef kernel_str = g_LLVMMDStringInContext2(ctx, "kernel", 6);
    LLVMValueRef one = g_LLVMConstInt(i32_ty, 1, 0);

    for (LLVMValueRef fn : kernel_fns) {
        LLVMMetadataRef md_ops[] = {
            g_LLVMValueAsMetadata(fn), kernel_str,
            g_LLVMValueAsMetadata(one)
        };
        LLVMMetadataRef node = g_LLVMMDNodeInContext2(ctx, md_ops, 3);
        g_LLVMAddNamedMetadataOperand(mod, "nvvm.annotations",
                                      g_LLVMMetadataAsValue(ctx, node));
    }

    uint64_t n = kernel_fns.size();
    LLVMTypeRef arr_ty = g_LLVMArrayType2(ptr_ty, n);
    LLVMValueRef used = g_LLVMAddGlobal(mod, arr_ty, "llvm.used");
    g_LLVMSetLinkage(used, LLVMAppendingLinkage);
    g_LLVMSetSection(used, "llvm.metadata");
    g_LLVMSetInitializer(used, g_LLVMConstArray2(ptr_ty, kernel_fns.data(), n));
}

// Add !nvvmir.version = !{i32 2, i32 0}
// Version 2.0 is accepted by all supported CTKs. Newer CTKs may use higher
// versions (e.g., CTK 13.1 uses NVVM IR 2.1) but are backward-compatible.
static void adapt_nvvmir_version(LLVMModuleRef mod, LLVMContextRef ctx) {
    if (g_LLVMGetNamedMetadataNumOperands(mod, "nvvmir.version") > 0)
        return;

    LLVMTypeRef i32_ty = g_LLVMInt32TypeInContext(ctx);
    LLVMValueRef two = g_LLVMConstInt(i32_ty, 2, 0);
    LLVMValueRef zero = g_LLVMConstInt(i32_ty, 0, 0);

    LLVMMetadataRef ops[] = {
        g_LLVMValueAsMetadata(two),
        g_LLVMValueAsMetadata(zero)
    };
    LLVMMetadataRef node = g_LLVMMDNodeInContext2(ctx, ops, 2);
    LLVMValueRef node_val = g_LLVMMetadataAsValue(ctx, node);
    g_LLVMAddNamedMetadataOperand(mod, "nvvmir.version", node_val);
}

// Debug Info Version module flag: behavior Warning(2) -> Error(1)
// Needed for LTOIR compatibility -- NVRTC uses Error behavior.
static void adapt_debug_info_version(LLVMModuleRef mod, LLVMContextRef ctx) {
    unsigned num_flags = g_LLVMGetNamedMetadataNumOperands(
        mod, "llvm.module.flags");
    if (num_flags == 0) return;

    std::vector<LLVMValueRef> flags(num_flags);
    g_LLVMGetNamedMetadataOperands(mod, "llvm.module.flags", flags.data());

    for (LLVMValueRef flag : flags) {
        if (g_LLVMGetMDNodeNumOperands(flag) < 3) continue;
        LLVMValueRef ops[3];
        g_LLVMGetMDNodeOperands(flag, ops);
        unsigned key_len;
        const char* key = g_LLVMGetMDString(ops[1], &key_len);
        if (!key || std::string_view(key, key_len) != "Debug Info Version")
            continue;
        if (g_LLVMConstIntGetZExtValue(ops[0]) != LLVMModuleFlagBehaviorWarning)
            continue;
        LLVMValueRef one = g_LLVMConstInt(g_LLVMInt32TypeInContext(ctx), 1, 0);
        g_LLVMReplaceMDNodeOperandWith(flag, 0, g_LLVMValueAsMetadata(one));
    }
}

static void adapt_for_libnvvm(LLVMModuleRef mod, LLVMContextRef ctx) {
    adapt_barrier_sync(mod, ctx);
    adapt_barrier_reduction(mod, ctx);
    adapt_inline_asm_intrinsics(mod, ctx);
    adapt_atomicrmw(mod, ctx);
    adapt_trunc(mod, ctx);
    adapt_nvvm_annotations(mod, ctx);
    adapt_nvvmir_version(mod, ctx);
    adapt_debug_info_version(mod, ctx);
}

// ===================================================================
// downgrade_for_libnvvm -- CTK-version-gated LLVM version gap fixes
// ===================================================================

// Strip lifetime intrinsics -- MLIR emits them with a 1-arg signature (ptr)
// but libnvvm expects the 2-arg form (i64, ptr). Just remove them.
// They're optimization hints with minimal impact on GPU code.
static void downgrade_lifetime(LLVMModuleRef mod, LLVMContextRef) {
    const char* names[] = {"llvm.lifetime.start.p0", "llvm.lifetime.end.p0"};
    for (const char* name : names) {
        LLVMValueRef fn = g_LLVMGetNamedFunction(mod, name);
        if (!fn || g_LLVMCountParams(fn) == 2) continue;
        for (LLVMValueRef call : collect_call_users(fn))
            g_LLVMInstructionEraseFromParent(call);
        g_LLVMDeleteFunction(fn);
    }
}

// Strip enum attributes not recognized by libnvvm's parser:
//   nocreateundeforpoison -> strip  (LLVM 23, rejected by all current libnvvm)
//   captures(...)         -> strip  (CTK < 13.0 only; LLVM 23 enum attr kind)
static void downgrade_attributes(LLVMModuleRef mod, LLVMContextRef,
                                 int ctk_major, int) {
    static const char nocup_name[] = "nocreateundeforpoison";
    static const char captures_name[] = "captures";
    unsigned nocup_kind = g_LLVMGetEnumAttributeKindForName(
        nocup_name, sizeof(nocup_name) - 1);
    unsigned captures_kind = g_LLVMGetEnumAttributeKindForName(
        captures_name, sizeof(captures_name) - 1);
    bool downgrade_captures = (ctk_major < 13);

    auto process = [&](LLVMValueRef fn, unsigned idx) {
        unsigned count = g_LLVMGetAttributeCountAtIndex(fn, idx);
        if (count == 0) return;

        std::vector<LLVMAttributeRef> attrs(count);
        g_LLVMGetAttributesAtIndex(fn, idx, attrs.data());

        for (LLVMAttributeRef attr : attrs) {
            if (!g_LLVMIsEnumAttribute(attr)) continue;
            unsigned kind = g_LLVMGetEnumAttributeKind(attr);
            if (kind == nocup_kind && nocup_kind != 0)
                g_LLVMRemoveEnumAttributeAtIndex(fn, idx, kind);
            else if (downgrade_captures && kind == captures_kind && captures_kind != 0)
                g_LLVMRemoveEnumAttributeAtIndex(fn, idx, kind);
        }
    };

    for (LLVMValueRef fn = g_LLVMGetFirstFunction(mod); fn;
         fn = g_LLVMGetNextFunction(fn)) {
        unsigned n = g_LLVMCountParams(fn);
        for (unsigned i = 0; i <= n; ++i)
            process(fn, i);
        process(fn, ~0U);
    }
}

// Emit legacy !nvvm.annotations grid_constant metadata for byval kernel params
// with "nvvm.grid_constant" string attribute. libnvvm's LLVM may not honor
// the string attribute form and instead requires the metadata:
//   !{ptr @fn, !"grid_constant", !{i32 1, i32 2}}  (1-based param indices)
static void downgrade_grid_constant(LLVMModuleRef mod, LLVMContextRef ctx) {
    for (LLVMValueRef fn = g_LLVMGetFirstFunction(mod); fn;
         fn = g_LLVMGetNextFunction(fn)) {
        if (g_LLVMGetFunctionCallConv(fn) != LLVMPTXKernelCallConv)
            continue;

        unsigned n = g_LLVMCountParams(fn);
        std::vector<LLVMValueRef> gc_indices;
        LLVMTypeRef i32_ty = g_LLVMInt32TypeInContext(ctx);

        for (unsigned i = 0; i < n; ++i) {
            unsigned idx = i + 1; // LLVM C API: param attributes at index 1..n
            unsigned count = g_LLVMGetAttributeCountAtIndex(fn, idx);
            if (count == 0) continue;

            std::vector<LLVMAttributeRef> attrs(count);
            g_LLVMGetAttributesAtIndex(fn, idx, attrs.data());

            for (LLVMAttributeRef attr : attrs) {
                if (!g_LLVMIsStringAttribute(attr)) continue;
                unsigned key_len;
                const char* key = g_LLVMGetStringAttributeKind(attr, &key_len);
                if (std::string_view(key, key_len) == "nvvm.grid_constant") {
                    gc_indices.push_back(g_LLVMConstInt(i32_ty, idx, 0));
                    break;
                }
            }
        }

        if (gc_indices.empty()) continue;

        std::vector<LLVMMetadataRef> idx_md(gc_indices.size());
        for (size_t i = 0; i < gc_indices.size(); ++i)
            idx_md[i] = g_LLVMValueAsMetadata(gc_indices[i]);
        LLVMMetadataRef idx_node = g_LLVMMDNodeInContext2(
            ctx, idx_md.data(), idx_md.size());

        LLVMMetadataRef md_ops[] = {
            g_LLVMValueAsMetadata(fn),
            g_LLVMMDStringInContext2(ctx, "grid_constant", 13),
            idx_node
        };
        LLVMMetadataRef node = g_LLVMMDNodeInContext2(ctx, md_ops, 3);
        g_LLVMAddNamedMetadataOperand(mod, "nvvm.annotations",
                                      g_LLVMMetadataAsValue(ctx, node));
    }
}

// These patches strip or rewrite LLVM IR syntax that the libnvvm parser in
// a given CTK doesn't yet accept. They become unnecessary for newer CTKs
// whose libnvvm accepts the syntax.
static void downgrade_for_libnvvm(LLVMModuleRef mod, LLVMContextRef ctx,
                                  int ctk_major, int ctk_minor) {
    downgrade_lifetime(mod, ctx); // All current CTKs need this
    downgrade_attributes(mod, ctx, ctk_major, ctk_minor); // CTK < 13.0 only
    downgrade_grid_constant(mod, ctx); // All current CTKs need this
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------

PyObject* py_downgrade_for_libnvvm(PyObject* /*self*/, PyObject* args) {
    unsigned long long mod_ptr_int, ctx_ptr_int;
    int ctk_major, ctk_minor;
    const char* lib_path;

    if (!PyArg_ParseTuple(args, "KKiis",
                          &mod_ptr_int, &ctx_ptr_int,
                          &ctk_major, &ctk_minor, &lib_path))
        return nullptr;

    if (!load_mlir_capi(lib_path))
        return nullptr;

    LLVMModuleRef llvm_mod = reinterpret_cast<LLVMModuleRef>(mod_ptr_int);
    LLVMContextRef llvm_ctx = reinterpret_cast<LLVMContextRef>(ctx_ptr_int);

    adapt_for_libnvvm(llvm_mod, llvm_ctx);
    downgrade_for_libnvvm(llvm_mod, llvm_ctx, ctk_major, ctk_minor);

    if (PyErr_Occurred()) {
        g_LLVMDisposeModule(llvm_mod);
        g_LLVMContextDispose(llvm_ctx);
        return nullptr;
    }

    PyObject* result = serialize_module(llvm_mod);
    g_LLVMDisposeModule(llvm_mod);
    g_LLVMContextDispose(llvm_ctx);
    return result;
}

PyMethodDef llvm_downgrade_methods[] = {
    {"downgrade_for_libnvvm", py_downgrade_for_libnvvm, METH_VARARGS,
     "Apply libnvvm compatibility transforms to an LLVMModuleRef and serialize."},
    {}
};

} // anonymous namespace

Status llvm_downgrade_init(PyObject* m) {
    if (PyModule_AddFunctions(m, llvm_downgrade_methods) < 0)
        return ErrorRaised;
    return OK;
}
