/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include "ModernBridge.h"

#include "mlir-c/IR.h"
#include "mlir-c/RegisterEverything.h"
#include "mlir-c/Target/LLVMIR.h"
#include "llvm-c/BitWriter.h"
#include "llvm-c/Core.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct MlirContextOwner {
    MlirContext value{nullptr};
    ~MlirContextOwner() {
        if (value.ptr)
            mlirContextDestroy(value);
    }
};

struct MlirOperationOwner {
    MlirOperation value{nullptr};
    ~MlirOperationOwner() {
        if (value.ptr)
            mlirOperationDestroy(value);
    }
};

struct LLVMContextOwner {
    LLVMContextRef value = nullptr;
    ~LLVMContextOwner() {
        if (value)
            LLVMContextDispose(value);
    }
};

struct LLVMModuleOwner {
    LLVMModuleRef value = nullptr;
    ~LLVMModuleOwner() {
        if (value)
            LLVMDisposeModule(value);
    }
};

static void set_error(char **error_out, const std::string &message) {
    if (!error_out)
        return;
    *error_out = nullptr;
    char *buffer = static_cast<char *>(std::malloc(message.size() + 1));
    if (!buffer)
        return;
    std::memcpy(buffer, message.data(), message.size());
    buffer[message.size()] = '\0';
    *error_out = buffer;
}

static bool copy_bytes(const char *data, size_t size, char **out,
                       size_t *out_len, char **error_out) {
    char *buffer = static_cast<char *>(std::malloc(size == 0 ? 1 : size));
    if (!buffer) {
        set_error(error_out, "failed to allocate output buffer");
        return false;
    }
    if (size)
        std::memcpy(buffer, data, size);
    *out = buffer;
    *out_len = size;
    return true;
}

static bool dump_module_to_stderr(LLVMModuleRef mod, char **error_out) {
    char *ir_text = LLVMPrintModuleToString(mod);
    if (!ir_text) {
        set_error(error_out, "LLVMPrintModuleToString failed");
        return false;
    }
    std::fprintf(stderr, "=============== LLVM IR ===============\n\n%s\n\n",
                 ir_text);
    LLVMDisposeMessage(ir_text);
    return true;
}

static bool serialize_module(LLVMModuleRef mod, char **out, size_t *out_len,
                             char **error_out) {
    LLVMMemoryBufferRef bitcode = LLVMWriteBitcodeToMemoryBuffer(mod);
    if (bitcode) {
        bool ok = copy_bytes(LLVMGetBufferStart(bitcode),
                             LLVMGetBufferSize(bitcode), out, out_len,
                             error_out);
        LLVMDisposeMemoryBuffer(bitcode);
        return ok;
    }

    char *ir_text = LLVMPrintModuleToString(mod);
    if (!ir_text) {
        set_error(error_out,
                  "LLVMWriteBitcodeToMemoryBuffer and LLVMPrintModuleToString failed");
        return false;
    }
    bool ok = copy_bytes(ir_text, std::strlen(ir_text), out, out_len,
                         error_out);
    LLVMDisposeMessage(ir_text);
    return ok;
}

static std::vector<LLVMValueRef> collect_call_users(LLVMValueRef fn) {
    std::vector<LLVMValueRef> calls;
    for (LLVMUseRef use = LLVMGetFirstUse(fn); use; use = LLVMGetNextUse(use)) {
        LLVMValueRef user = LLVMGetUser(use);
        if (LLVMIsACallInst(user) && LLVMGetCalledValue(user) == fn)
            calls.push_back(user);
    }
    return calls;
}

static LLVMValueRef get_or_add_function(LLVMModuleRef mod, const char *name,
                                        LLVMTypeRef ty) {
    LLVMValueRef fn = LLVMGetNamedFunction(mod, name);
    if (!fn)
        fn = LLVMAddFunction(mod, name, ty);
    return fn;
}

static void replace_intrinsic_with_asm(
    LLVMModuleRef mod, LLVMContextRef ctx, const char *intrinsic_name,
    LLVMTypeRef fn_ty, const char *asm_str, const char *constraints,
    bool has_side_effects) {
    LLVMValueRef old_fn = LLVMGetNamedFunction(mod, intrinsic_name);
    if (!old_fn)
        return;

    LLVMValueRef inline_asm = LLVMGetInlineAsm(
        fn_ty, asm_str, std::strlen(asm_str), constraints,
        std::strlen(constraints), has_side_effects, false,
        LLVMInlineAsmDialectATT, false);

    bool returns_value =
        LLVMGetTypeKind(LLVMGetReturnType(fn_ty)) != LLVMVoidTypeKind;

    LLVMBuilderRef builder = LLVMCreateBuilderInContext(ctx);
    for (LLVMValueRef call : collect_call_users(old_fn)) {
        LLVMPositionBuilderBefore(builder, call);
        unsigned total_operands = LLVMGetNumOperands(call);
        unsigned num_args = total_operands > 0 ? total_operands - 1 : 0;
        std::vector<LLVMValueRef> args(num_args);
        for (unsigned i = 0; i < num_args; ++i)
            args[i] = LLVMGetOperand(call, i);
        LLVMValueRef result = LLVMBuildCall2(
            builder, fn_ty, inline_asm, num_args ? args.data() : nullptr,
            num_args, "");
        if (returns_value)
            LLVMReplaceAllUsesWith(call, result);
        LLVMInstructionEraseFromParent(call);
    }
    LLVMDisposeBuilder(builder);
}

static void adapt_barrier_sync(LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMValueRef old_fn = LLVMGetNamedFunction(
        mod, "llvm.nvvm.barrier.cta.sync.aligned.all");
    if (!old_fn)
        return;

    LLVMTypeRef void_ty = LLVMVoidTypeInContext(ctx);
    LLVMTypeRef i32_ty = LLVMInt32TypeInContext(ctx);

    LLVMTypeRef barrier0_fn_ty = LLVMFunctionType(void_ty, nullptr, 0, false);
    LLVMValueRef barrier0_fn =
        get_or_add_function(mod, "llvm.nvvm.barrier0", barrier0_fn_ty);

    LLVMTypeRef bar_sync_param = i32_ty;
    LLVMTypeRef bar_sync_fn_ty =
        LLVMFunctionType(void_ty, &bar_sync_param, 1, false);
    LLVMValueRef bar_sync_fn =
        get_or_add_function(mod, "llvm.nvvm.bar.sync", bar_sync_fn_ty);

    LLVMBuilderRef builder = LLVMCreateBuilderInContext(ctx);
    for (LLVMValueRef call : collect_call_users(old_fn)) {
        LLVMPositionBuilderBefore(builder, call);
        LLVMValueRef arg = LLVMGetOperand(call, 0);
        bool is_zero =
            LLVMIsAConstantInt(arg) && LLVMConstIntGetZExtValue(arg) == 0;
        if (is_zero) {
            LLVMBuildCall2(builder, barrier0_fn_ty, barrier0_fn, nullptr, 0,
                           "");
        } else {
            LLVMBuildCall2(builder, bar_sync_fn_ty, bar_sync_fn, &arg, 1, "");
        }
        LLVMInstructionEraseFromParent(call);
    }
    LLVMDisposeBuilder(builder);
}

static void adapt_barrier_reduction(LLVMModuleRef mod, LLVMContextRef ctx) {
    const char *ops[] = {"and", "or", "popc"};

    LLVMTypeRef i1_ty = LLVMInt1TypeInContext(ctx);
    LLVMTypeRef i32_ty = LLVMInt32TypeInContext(ctx);

    LLVMTypeRef new_param = i32_ty;
    LLVMTypeRef new_fn_ty = LLVMFunctionType(i32_ty, &new_param, 1, false);

    for (const char *op : ops) {
        char old_name[128];
        std::snprintf(old_name, sizeof(old_name),
                      "llvm.nvvm.barrier.cta.red.%s.aligned.all", op);
        char new_name[128];
        std::snprintf(new_name, sizeof(new_name), "llvm.nvvm.barrier0.%s",
                      op);

        LLVMValueRef old_fn = LLVMGetNamedFunction(mod, old_name);
        if (!old_fn)
            continue;

        LLVMValueRef new_fn = get_or_add_function(mod, new_name, new_fn_ty);
        bool returns_i1 = std::strcmp(op, "and") == 0 ||
                          std::strcmp(op, "or") == 0;

        LLVMBuilderRef builder = LLVMCreateBuilderInContext(ctx);
        for (LLVMValueRef call : collect_call_users(old_fn)) {
            LLVMPositionBuilderBefore(builder, call);
            LLVMValueRef pred = LLVMGetOperand(call, 1);
            LLVMValueRef pred_i32 = LLVMBuildZExt(builder, pred, i32_ty, "");
            LLVMValueRef new_call =
                LLVMBuildCall2(builder, new_fn_ty, new_fn, &pred_i32, 1, "");
            if (returns_i1) {
                LLVMValueRef result_i1 =
                    LLVMBuildTrunc(builder, new_call, i1_ty, "");
                LLVMReplaceAllUsesWith(call, result_i1);
            } else {
                LLVMReplaceAllUsesWith(call, new_call);
            }
            LLVMInstructionEraseFromParent(call);
        }
        LLVMDisposeBuilder(builder);
    }
}

static void adapt_inline_asm_intrinsics(LLVMModuleRef mod,
                                        LLVMContextRef ctx) {
    LLVMTypeRef void_ty = LLVMVoidTypeInContext(ctx);
    LLVMTypeRef i32_ty = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef ptr_ty = LLVMPointerTypeInContext(ctx, 0);

    LLVMTypeRef ns_param = i32_ty;
    replace_intrinsic_with_asm(
        mod, ctx, "llvm.nvvm.nanosleep",
        LLVMFunctionType(void_ty, &ns_param, 1, false), "nanosleep.u32 $0;",
        "r", true);

    replace_intrinsic_with_asm(
        mod, ctx, "llvm.stacksave.p0",
        LLVMFunctionType(ptr_ty, nullptr, 0, false), "stacksave.u64 $0;",
        "=l", true);

    LLVMTypeRef sr_param = ptr_ty;
    replace_intrinsic_with_asm(
        mod, ctx, "llvm.stackrestore.p0",
        LLVMFunctionType(void_ty, &sr_param, 1, false), "stackrestore.u64 $0;",
        "l", true);

    LLVMTypeRef mapa_params[] = {ptr_ty, i32_ty};
    replace_intrinsic_with_asm(
        mod, ctx, "llvm.nvvm.mapa",
        LLVMFunctionType(ptr_ty, mapa_params, 2, false),
        "mapa.u64 $0, $1, $2;", "=l,l,r", false);
}

static void adapt_atomicrmw(LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMTypeRef float_ty = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef double_ty = LLVMDoubleTypeInContext(ctx);
    LLVMBuilderRef builder = LLVMCreateBuilderInContext(ctx);

    auto lower_fadd = [&](LLVMValueRef inst) {
        LLVMTypeRef val_ty = LLVMTypeOf(inst);
        if (val_ty != float_ty && val_ty != double_ty)
            return;

        LLVMValueRef ptr = LLVMGetOperand(inst, 0);
        LLVMValueRef val = LLVMGetOperand(inst, 1);
        LLVMTypeRef ptr_ty = LLVMTypeOf(ptr);

        bool is_f32 = val_ty == float_ty;
        const char *constraints = is_f32 ? "=f,l,f" : "=d,l,d";
        unsigned addrspace = LLVMGetPointerAddressSpace(ptr_ty);
        const char *space =
            addrspace == 3 ? "shared." : addrspace == 1 ? "global." : "";

        char asm_str[128];
        std::snprintf(asm_str, sizeof(asm_str), "atom.%sadd.%s $0, [$1], $2;",
                      space, is_f32 ? "f32" : "f64");

        LLVMTypeRef asm_params[] = {ptr_ty, val_ty};
        LLVMTypeRef asm_fn_ty =
            LLVMFunctionType(val_ty, asm_params, 2, false);
        LLVMValueRef inline_asm = LLVMGetInlineAsm(
            asm_fn_ty, asm_str, std::strlen(asm_str), constraints,
            std::strlen(constraints), true, false, LLVMInlineAsmDialectATT,
            false);

        LLVMPositionBuilderBefore(builder, inst);
        LLVMValueRef args[] = {ptr, val};
        LLVMValueRef asm_call =
            LLVMBuildCall2(builder, asm_fn_ty, inline_asm, args, 2, "");
        LLVMReplaceAllUsesWith(inst, asm_call);
        LLVMInstructionEraseFromParent(inst);
    };

    for (LLVMValueRef fn = LLVMGetFirstFunction(mod); fn;
         fn = LLVMGetNextFunction(fn)) {
        for (LLVMBasicBlockRef bb = LLVMGetFirstBasicBlock(fn); bb;
             bb = LLVMGetNextBasicBlock(bb)) {
            LLVMValueRef inst = LLVMGetFirstInstruction(bb);
            while (inst) {
                LLVMValueRef next = LLVMGetNextInstruction(inst);
                if (LLVMIsAAtomicRMWInst(inst)) {
                    LLVMAtomicRMWBinOp binop = LLVMGetAtomicRMWBinOp(inst);
                    if (binop == LLVMAtomicRMWBinOpFMinimum)
                        LLVMSetAtomicRMWBinOp(inst, LLVMAtomicRMWBinOpFMin);
                    else if (binop == LLVMAtomicRMWBinOpFMaximum)
                        LLVMSetAtomicRMWBinOp(inst, LLVMAtomicRMWBinOpFMax);
                    else if (binop == LLVMAtomicRMWBinOpFAdd)
                        lower_fadd(inst);
                }
                inst = next;
            }
        }
    }
    LLVMDisposeBuilder(builder);
}

static void adapt_trunc(LLVMModuleRef mod, LLVMContextRef ctx) {
    LLVMTypeRef float_ty = LLVMFloatTypeInContext(ctx);
    LLVMTypeRef double_ty = LLVMDoubleTypeInContext(ctx);
    LLVMTypeRef f32_fn_ty = LLVMFunctionType(float_ty, &float_ty, 1, false);
    LLVMTypeRef f64_fn_ty = LLVMFunctionType(double_ty, &double_ty, 1, false);

    struct Mapping {
        const char *intrinsic;
        const char *libdevice;
        LLVMTypeRef ty;
    };
    Mapping mappings[] = {
        {"llvm.trunc.f64", "__nv_trunc", double_ty},
        {"llvm.trunc.f32", "__nv_truncf", float_ty},
        {"llvm.trunc.f16", "__nv_truncf", LLVMHalfTypeInContext(ctx)},
        {"llvm.trunc.bf16", "__nv_truncf", LLVMBFloatTypeInContext(ctx)},
    };

    LLVMBuilderRef builder = LLVMCreateBuilderInContext(ctx);
    for (auto &mapping : mappings) {
        LLVMValueRef old_fn = LLVMGetNamedFunction(mod, mapping.intrinsic);
        if (!old_fn)
            continue;

        bool is_f64 = mapping.ty == double_ty;
        bool promote = mapping.ty != float_ty && !is_f64;
        LLVMTypeRef lib_fn_ty = is_f64 ? f64_fn_ty : f32_fn_ty;
        LLVMValueRef lib_fn =
            get_or_add_function(mod, mapping.libdevice, lib_fn_ty);

        for (LLVMValueRef call : collect_call_users(old_fn)) {
            LLVMPositionBuilderBefore(builder, call);
            LLVMValueRef arg = LLVMGetOperand(call, 0);
            if (promote)
                arg = LLVMBuildFPExt(builder, arg, float_ty, "");
            LLVMValueRef result =
                LLVMBuildCall2(builder, lib_fn_ty, lib_fn, &arg, 1, "");
            if (promote)
                result = LLVMBuildFPTrunc(builder, result, mapping.ty, "");
            LLVMReplaceAllUsesWith(call, result);
            LLVMInstructionEraseFromParent(call);
        }
    }
    LLVMDisposeBuilder(builder);
}

static void adapt_nvvm_annotations(LLVMModuleRef mod, LLVMContextRef ctx) {
    if (LLVMGetNamedMetadataNumOperands(mod, "nvvm.annotations") > 0)
        return;

    std::vector<LLVMValueRef> kernel_fns;
    for (LLVMValueRef fn = LLVMGetFirstFunction(mod); fn;
         fn = LLVMGetNextFunction(fn)) {
        if (LLVMGetFunctionCallConv(fn) == LLVMPTXKernelCallConv)
            kernel_fns.push_back(fn);
    }
    if (kernel_fns.empty())
        return;

    LLVMTypeRef i32_ty = LLVMInt32TypeInContext(ctx);
    LLVMTypeRef ptr_ty = LLVMPointerTypeInContext(ctx, 0);
    LLVMMetadataRef kernel_str = LLVMMDStringInContext2(ctx, "kernel", 6);
    LLVMValueRef one = LLVMConstInt(i32_ty, 1, false);

    for (LLVMValueRef fn : kernel_fns) {
        LLVMMetadataRef md_ops[] = {LLVMValueAsMetadata(fn), kernel_str,
                                    LLVMValueAsMetadata(one)};
        LLVMMetadataRef node = LLVMMDNodeInContext2(ctx, md_ops, 3);
        LLVMAddNamedMetadataOperand(mod, "nvvm.annotations",
                                    LLVMMetadataAsValue(ctx, node));
    }

    uint64_t n = kernel_fns.size();
    LLVMTypeRef arr_ty = LLVMArrayType2(ptr_ty, n);
    LLVMValueRef used = LLVMAddGlobal(mod, arr_ty, "llvm.used");
    LLVMSetLinkage(used, LLVMAppendingLinkage);
    LLVMSetSection(used, "llvm.metadata");
    LLVMSetInitializer(used, LLVMConstArray2(ptr_ty, kernel_fns.data(), n));
}

static void adapt_nvvmir_version(LLVMModuleRef mod, LLVMContextRef ctx) {
    if (LLVMGetNamedMetadataNumOperands(mod, "nvvmir.version") > 0)
        return;

    LLVMTypeRef i32_ty = LLVMInt32TypeInContext(ctx);
    LLVMValueRef two = LLVMConstInt(i32_ty, 2, false);
    LLVMValueRef zero = LLVMConstInt(i32_ty, 0, false);

    LLVMMetadataRef ops[] = {LLVMValueAsMetadata(two),
                             LLVMValueAsMetadata(zero)};
    LLVMMetadataRef node = LLVMMDNodeInContext2(ctx, ops, 2);
    LLVMAddNamedMetadataOperand(mod, "nvvmir.version",
                                LLVMMetadataAsValue(ctx, node));
}

static void adapt_debug_info_version(LLVMModuleRef mod, LLVMContextRef ctx) {
    unsigned num_flags =
        LLVMGetNamedMetadataNumOperands(mod, "llvm.module.flags");
    if (num_flags == 0)
        return;

    std::vector<LLVMValueRef> flags(num_flags);
    LLVMGetNamedMetadataOperands(mod, "llvm.module.flags", flags.data());

    for (LLVMValueRef flag : flags) {
        if (LLVMGetMDNodeNumOperands(flag) < 3)
            continue;
        LLVMValueRef ops[3];
        LLVMGetMDNodeOperands(flag, ops);
        unsigned key_len;
        const char *key = LLVMGetMDString(ops[1], &key_len);
        if (!key || std::string_view(key, key_len) != "Debug Info Version")
            continue;
        if (LLVMConstIntGetZExtValue(ops[0]) != LLVMModuleFlagBehaviorWarning)
            continue;
        LLVMValueRef one = LLVMConstInt(LLVMInt32TypeInContext(ctx), 1, false);
        LLVMReplaceMDNodeOperandWith(flag, 0, LLVMValueAsMetadata(one));
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

static void downgrade_lifetime(LLVMModuleRef mod, LLVMContextRef) {
    const char *names[] = {"llvm.lifetime.start.p0", "llvm.lifetime.end.p0"};
    for (const char *name : names) {
        LLVMValueRef fn = LLVMGetNamedFunction(mod, name);
        if (!fn || LLVMCountParams(fn) == 2)
            continue;
        for (LLVMValueRef call : collect_call_users(fn))
            LLVMInstructionEraseFromParent(call);
        LLVMDeleteFunction(fn);
    }
}

static void downgrade_attributes(LLVMModuleRef mod, LLVMContextRef,
                                 int ctk_major, int) {
    static const char nocup_name[] = "nocreateundeforpoison";
    static const char captures_name[] = "captures";
    unsigned nocup_kind =
        LLVMGetEnumAttributeKindForName(nocup_name, sizeof(nocup_name) - 1);
    unsigned captures_kind = LLVMGetEnumAttributeKindForName(
        captures_name, sizeof(captures_name) - 1);
    bool downgrade_captures = ctk_major < 13;

    auto process = [&](LLVMValueRef fn, unsigned idx) {
        unsigned count = LLVMGetAttributeCountAtIndex(fn, idx);
        if (count == 0)
            return;

        std::vector<LLVMAttributeRef> attrs(count);
        LLVMGetAttributesAtIndex(fn, idx, attrs.data());

        for (LLVMAttributeRef attr : attrs) {
            if (!LLVMIsEnumAttribute(attr))
                continue;
            unsigned kind = LLVMGetEnumAttributeKind(attr);
            if (kind == nocup_kind && nocup_kind != 0)
                LLVMRemoveEnumAttributeAtIndex(fn, idx, kind);
            else if (downgrade_captures && kind == captures_kind &&
                     captures_kind != 0)
                LLVMRemoveEnumAttributeAtIndex(fn, idx, kind);
        }
    };

    for (LLVMValueRef fn = LLVMGetFirstFunction(mod); fn;
         fn = LLVMGetNextFunction(fn)) {
        unsigned n = LLVMCountParams(fn);
        for (unsigned i = 0; i <= n; ++i)
            process(fn, i);
        process(fn, ~0U);
    }
}

static void downgrade_grid_constant(LLVMModuleRef mod, LLVMContextRef ctx) {
    for (LLVMValueRef fn = LLVMGetFirstFunction(mod); fn;
         fn = LLVMGetNextFunction(fn)) {
        if (LLVMGetFunctionCallConv(fn) != LLVMPTXKernelCallConv)
            continue;

        unsigned n = LLVMCountParams(fn);
        std::vector<LLVMValueRef> gc_indices;
        LLVMTypeRef i32_ty = LLVMInt32TypeInContext(ctx);

        for (unsigned i = 0; i < n; ++i) {
            unsigned idx = i + 1;
            unsigned count = LLVMGetAttributeCountAtIndex(fn, idx);
            if (count == 0)
                continue;

            std::vector<LLVMAttributeRef> attrs(count);
            LLVMGetAttributesAtIndex(fn, idx, attrs.data());

            for (LLVMAttributeRef attr : attrs) {
                if (!LLVMIsStringAttribute(attr))
                    continue;
                unsigned key_len;
                const char *key = LLVMGetStringAttributeKind(attr, &key_len);
                if (key && std::string_view(key, key_len) ==
                               "nvvm.grid_constant") {
                    gc_indices.push_back(LLVMConstInt(i32_ty, idx, false));
                    break;
                }
            }
        }

        if (gc_indices.empty())
            continue;

        std::vector<LLVMMetadataRef> idx_md(gc_indices.size());
        for (size_t i = 0; i < gc_indices.size(); ++i)
            idx_md[i] = LLVMValueAsMetadata(gc_indices[i]);
        LLVMMetadataRef idx_node =
            LLVMMDNodeInContext2(ctx, idx_md.data(), idx_md.size());

        LLVMMetadataRef md_ops[] = {
            LLVMValueAsMetadata(fn),
            LLVMMDStringInContext2(ctx, "grid_constant", 13), idx_node};
        LLVMMetadataRef node = LLVMMDNodeInContext2(ctx, md_ops, 3);
        LLVMAddNamedMetadataOperand(mod, "nvvm.annotations",
                                    LLVMMetadataAsValue(ctx, node));
    }
}

static void downgrade_for_libnvvm(LLVMModuleRef mod, LLVMContextRef ctx,
                                  int ctk_major, int ctk_minor) {
    downgrade_lifetime(mod, ctx);
    downgrade_attributes(mod, ctx, ctk_major, ctk_minor);
    downgrade_grid_constant(mod, ctx);
}

static bool initialize_mlir_context(MlirContextOwner &context,
                                    char **error_out) {
    MlirDialectRegistry registry = mlirDialectRegistryCreate();
    if (!registry.ptr) {
        set_error(error_out, "mlirDialectRegistryCreate failed");
        return false;
    }

    mlirRegisterAllDialects(registry);
    context.value = mlirContextCreateWithRegistry(registry, false);
    mlirDialectRegistryDestroy(registry);

    if (!context.value.ptr) {
        set_error(error_out, "mlirContextCreateWithRegistry failed");
        return false;
    }

    mlirRegisterAllLLVMTranslations(context.value);
    mlirContextLoadAllAvailableDialects(context.value);
    return true;
}

} // namespace

extern "C" MLIR_MODERN_TO_NVVM_EXPORT int
mlir_modern_to_nvvm_translate_for_libnvvm(
    const char *mlir_text, size_t mlir_text_len, int ctk_major, int ctk_minor,
    int dump_llvmir, char **out, size_t *out_len, char **error_out) {
    if (out)
        *out = nullptr;
    if (out_len)
        *out_len = 0;
    if (error_out)
        *error_out = nullptr;

    if (!mlir_text || !out || !out_len) {
        set_error(error_out, "invalid null argument");
        return 1;
    }

    MlirContextOwner mlir_context;
    if (!initialize_mlir_context(mlir_context, error_out))
        return 1;

    MlirOperationOwner mlir_op;
    mlir_op.value = mlirOperationCreateParse(
        mlir_context.value, mlirStringRefCreate(mlir_text, mlir_text_len),
        mlirStringRefCreateFromCString("numba-cuda-mlir-gpu-module.mlir"));
    if (mlirOperationIsNull(mlir_op.value)) {
        set_error(error_out, "failed to parse MLIR gpu.module");
        return 1;
    }
    if (!mlirOperationVerify(mlir_op.value)) {
        set_error(error_out, "MLIR gpu.module verification failed");
        return 1;
    }

    LLVMContextOwner llvm_context;
    llvm_context.value = LLVMContextCreate();
    if (!llvm_context.value) {
        set_error(error_out, "LLVMContextCreate failed");
        return 1;
    }

    LLVMModuleOwner llvm_module;
    llvm_module.value =
        mlirTranslateModuleToLLVMIR(mlir_op.value, llvm_context.value);
    if (!llvm_module.value) {
        set_error(error_out, "mlirTranslateModuleToLLVMIR failed");
        return 1;
    }

    if (dump_llvmir && !dump_module_to_stderr(llvm_module.value, error_out))
        return 1;

    adapt_for_libnvvm(llvm_module.value, llvm_context.value);
    downgrade_for_libnvvm(llvm_module.value, llvm_context.value, ctk_major,
                          ctk_minor);

    return serialize_module(llvm_module.value, out, out_len, error_out) ? 0 : 1;
}

extern "C" MLIR_MODERN_TO_NVVM_EXPORT void
mlir_modern_to_nvvm_free(void *ptr) {
    std::free(ptr);
}
