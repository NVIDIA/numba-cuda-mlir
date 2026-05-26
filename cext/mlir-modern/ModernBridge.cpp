/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */
#include "ModernBridge.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

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

static bool dump_module_to_stderr(llvm::Module &mod, char **error_out) {
    std::string ir_text;
    llvm::raw_string_ostream os(ir_text);
    mod.print(os, nullptr);
    os.flush();
    if (ir_text.empty()) {
        set_error(error_out, "LLVM module print produced no output");
        return false;
    }
    std::fprintf(stderr, "=============== LLVM IR ===============\n\n%s\n\n",
                 ir_text.c_str());
    return true;
}

static bool serialize_module(llvm::Module &mod, char **out, size_t *out_len,
                             char **error_out) {
    llvm::SmallVector<char, 0> bitcode;
    llvm::raw_svector_ostream os(bitcode);
    llvm::WriteBitcodeToFile(mod, os);
    if (bitcode.empty()) {
        set_error(error_out, "LLVM bitcode serialization produced no output");
        return false;
    }
    return copy_bytes(bitcode.data(), bitcode.size(), out, out_len, error_out);
}

static std::vector<llvm::CallBase *> collect_call_users(llvm::Function *fn) {
    std::vector<llvm::CallBase *> calls;
    for (llvm::Use &use : fn->uses()) {
        auto *call = llvm::dyn_cast<llvm::CallBase>(use.getUser());
        if (!call)
            continue;
        if (call->getCalledOperand()->stripPointerCasts() == fn)
            calls.push_back(call);
    }
    return calls;
}

static llvm::Function *get_or_add_function(llvm::Module &mod,
                                           llvm::StringRef name,
                                           llvm::FunctionType *ty) {
    if (llvm::Function *fn = mod.getFunction(name))
        return fn;
    return llvm::Function::Create(ty, llvm::GlobalValue::ExternalLinkage,
                                  name, mod);
}

static void replace_intrinsic_with_asm(
    llvm::Module &mod, llvm::LLVMContext &ctx, llvm::StringRef intrinsic_name,
    llvm::FunctionType *fn_ty, llvm::StringRef asm_str,
    llvm::StringRef constraints, bool has_side_effects) {
    llvm::Function *old_fn = mod.getFunction(intrinsic_name);
    if (!old_fn)
        return;

    llvm::InlineAsm *inline_asm =
        llvm::InlineAsm::get(fn_ty, asm_str, constraints, has_side_effects,
                             false, llvm::InlineAsm::AD_ATT, false);

    bool returns_value = !fn_ty->getReturnType()->isVoidTy();

    llvm::IRBuilder<> builder(ctx);
    for (llvm::CallBase *call : collect_call_users(old_fn)) {
        builder.SetInsertPoint(call);
        std::vector<llvm::Value *> args;
        args.reserve(call->arg_size());
        for (llvm::Use &arg : call->args())
            args.push_back(arg.get());

        llvm::CallInst *result =
            builder.CreateCall(fn_ty, inline_asm, args, "");
        if (returns_value)
            call->replaceAllUsesWith(result);
        call->eraseFromParent();
    }
}

static void adapt_barrier_sync(llvm::Module &mod, llvm::LLVMContext &ctx) {
    llvm::Function *old_fn =
        mod.getFunction("llvm.nvvm.barrier.cta.sync.aligned.all");
    if (!old_fn)
        return;

    llvm::Type *void_ty = llvm::Type::getVoidTy(ctx);
    llvm::Type *i32_ty = llvm::Type::getInt32Ty(ctx);

    llvm::FunctionType *barrier0_fn_ty =
        llvm::FunctionType::get(void_ty, {}, false);
    llvm::Function *barrier0_fn =
        get_or_add_function(mod, "llvm.nvvm.barrier0", barrier0_fn_ty);

    llvm::FunctionType *bar_sync_fn_ty =
        llvm::FunctionType::get(void_ty, {i32_ty}, false);
    llvm::Function *bar_sync_fn =
        get_or_add_function(mod, "llvm.nvvm.bar.sync", bar_sync_fn_ty);

    llvm::IRBuilder<> builder(ctx);
    for (llvm::CallBase *call : collect_call_users(old_fn)) {
        builder.SetInsertPoint(call);
        llvm::Value *arg = call->getArgOperand(0);
        auto *constant = llvm::dyn_cast<llvm::ConstantInt>(arg);
        if (constant && constant->isZero())
            builder.CreateCall(barrier0_fn_ty, barrier0_fn, {});
        else
            builder.CreateCall(bar_sync_fn_ty, bar_sync_fn, {arg});
        call->eraseFromParent();
    }
}

static void adapt_barrier_reduction(llvm::Module &mod,
                                    llvm::LLVMContext &ctx) {
    const char *ops[] = {"and", "or", "popc"};

    llvm::Type *i1_ty = llvm::Type::getInt1Ty(ctx);
    llvm::Type *i32_ty = llvm::Type::getInt32Ty(ctx);
    llvm::FunctionType *new_fn_ty =
        llvm::FunctionType::get(i32_ty, {i32_ty}, false);

    for (const char *op : ops) {
        std::string old_name =
            "llvm.nvvm.barrier.cta.red." + std::string(op) + ".aligned.all";
        std::string new_name = "llvm.nvvm.barrier0." + std::string(op);

        llvm::Function *old_fn = mod.getFunction(old_name);
        if (!old_fn)
            continue;

        llvm::Function *new_fn =
            get_or_add_function(mod, new_name, new_fn_ty);
        bool returns_i1 =
            std::strcmp(op, "and") == 0 || std::strcmp(op, "or") == 0;

        llvm::IRBuilder<> builder(ctx);
        for (llvm::CallBase *call : collect_call_users(old_fn)) {
            builder.SetInsertPoint(call);
            llvm::Value *pred = call->getArgOperand(1);
            llvm::Value *pred_i32 = builder.CreateZExt(pred, i32_ty, "");
            llvm::CallInst *new_call =
                builder.CreateCall(new_fn_ty, new_fn, {pred_i32}, "");
            if (returns_i1) {
                llvm::Value *result_i1 =
                    builder.CreateTrunc(new_call, i1_ty, "");
                call->replaceAllUsesWith(result_i1);
            } else {
                call->replaceAllUsesWith(new_call);
            }
            call->eraseFromParent();
        }
    }
}

static void adapt_inline_asm_intrinsics(llvm::Module &mod,
                                        llvm::LLVMContext &ctx) {
    llvm::Type *void_ty = llvm::Type::getVoidTy(ctx);
    llvm::Type *i32_ty = llvm::Type::getInt32Ty(ctx);
    llvm::Type *ptr_ty = llvm::PointerType::get(ctx, 0);

    replace_intrinsic_with_asm(
        mod, ctx, "llvm.nvvm.nanosleep",
        llvm::FunctionType::get(void_ty, {i32_ty}, false),
        "nanosleep.u32 $0;", "r", true);

    replace_intrinsic_with_asm(
        mod, ctx, "llvm.stacksave.p0",
        llvm::FunctionType::get(ptr_ty, {}, false), "stacksave.u64 $0;",
        "=l", true);

    replace_intrinsic_with_asm(
        mod, ctx, "llvm.stackrestore.p0",
        llvm::FunctionType::get(void_ty, {ptr_ty}, false),
        "stackrestore.u64 $0;", "l", true);

    replace_intrinsic_with_asm(
        mod, ctx, "llvm.nvvm.mapa",
        llvm::FunctionType::get(ptr_ty, {ptr_ty, i32_ty}, false),
        "mapa.u64 $0, $1, $2;", "=l,l,r", false);
}

static void adapt_atomicrmw(llvm::Module &mod, llvm::LLVMContext &ctx) {
    llvm::Type *float_ty = llvm::Type::getFloatTy(ctx);
    llvm::Type *double_ty = llvm::Type::getDoubleTy(ctx);
    llvm::IRBuilder<> builder(ctx);

    auto lower_fadd = [&](llvm::AtomicRMWInst *inst) {
        llvm::Type *val_ty = inst->getType();
        if (val_ty != float_ty && val_ty != double_ty)
            return;

        llvm::Value *ptr = inst->getPointerOperand();
        llvm::Value *val = inst->getValOperand();
        llvm::Type *ptr_ty = ptr->getType();

        bool is_f32 = val_ty == float_ty;
        llvm::StringRef constraints = is_f32 ? "=f,l,f" : "=d,l,d";
        unsigned addrspace = inst->getPointerAddressSpace();
        const char *space =
            addrspace == 3 ? "shared." : addrspace == 1 ? "global." : "";

        char asm_str[128];
        std::snprintf(asm_str, sizeof(asm_str), "atom.%sadd.%s $0, [$1], $2;",
                      space, is_f32 ? "f32" : "f64");

        llvm::FunctionType *asm_fn_ty =
            llvm::FunctionType::get(val_ty, {ptr_ty, val_ty}, false);
        llvm::InlineAsm *inline_asm =
            llvm::InlineAsm::get(asm_fn_ty, asm_str, constraints, true,
                                 false, llvm::InlineAsm::AD_ATT, false);

        builder.SetInsertPoint(inst);
        llvm::CallInst *asm_call =
            builder.CreateCall(asm_fn_ty, inline_asm, {ptr, val}, "");
        inst->replaceAllUsesWith(asm_call);
        inst->eraseFromParent();
    };

    for (llvm::Function &fn : mod) {
        for (llvm::BasicBlock &bb : fn) {
            for (auto it = bb.begin(); it != bb.end();) {
                llvm::Instruction *inst = &*it++;
                auto *atomic = llvm::dyn_cast<llvm::AtomicRMWInst>(inst);
                if (!atomic)
                    continue;
                switch (atomic->getOperation()) {
                case llvm::AtomicRMWInst::FMinimum:
                    atomic->setOperation(llvm::AtomicRMWInst::FMin);
                    break;
                case llvm::AtomicRMWInst::FMaximum:
                    atomic->setOperation(llvm::AtomicRMWInst::FMax);
                    break;
                case llvm::AtomicRMWInst::FAdd:
                    lower_fadd(atomic);
                    break;
                default:
                    break;
                }
            }
        }
    }
}

static void adapt_trunc(llvm::Module &mod, llvm::LLVMContext &ctx) {
    llvm::Type *float_ty = llvm::Type::getFloatTy(ctx);
    llvm::Type *double_ty = llvm::Type::getDoubleTy(ctx);
    llvm::FunctionType *f32_fn_ty =
        llvm::FunctionType::get(float_ty, {float_ty}, false);
    llvm::FunctionType *f64_fn_ty =
        llvm::FunctionType::get(double_ty, {double_ty}, false);

    struct Mapping {
        const char *intrinsic;
        const char *libdevice;
        llvm::Type *ty;
    };
    Mapping mappings[] = {
        {"llvm.trunc.f64", "__nv_trunc", double_ty},
        {"llvm.trunc.f32", "__nv_truncf", float_ty},
        {"llvm.trunc.f16", "__nv_truncf", llvm::Type::getHalfTy(ctx)},
        {"llvm.trunc.bf16", "__nv_truncf", llvm::Type::getBFloatTy(ctx)},
    };

    llvm::IRBuilder<> builder(ctx);
    for (const Mapping &mapping : mappings) {
        llvm::Function *old_fn = mod.getFunction(mapping.intrinsic);
        if (!old_fn)
            continue;

        bool is_f64 = mapping.ty == double_ty;
        bool promote = mapping.ty != float_ty && !is_f64;
        llvm::FunctionType *lib_fn_ty = is_f64 ? f64_fn_ty : f32_fn_ty;
        llvm::Function *lib_fn =
            get_or_add_function(mod, mapping.libdevice, lib_fn_ty);

        for (llvm::CallBase *call : collect_call_users(old_fn)) {
            builder.SetInsertPoint(call);
            llvm::Value *arg = call->getArgOperand(0);
            if (promote)
                arg = builder.CreateFPExt(arg, float_ty, "");
            llvm::CallInst *result =
                builder.CreateCall(lib_fn_ty, lib_fn, {arg}, "");
            llvm::Value *final_result = result;
            if (promote)
                final_result = builder.CreateFPTrunc(result, mapping.ty, "");
            call->replaceAllUsesWith(final_result);
            call->eraseFromParent();
        }
    }
}

static void adapt_nvvm_annotations(llvm::Module &mod,
                                   llvm::LLVMContext &ctx) {
    if (llvm::NamedMDNode *existing = mod.getNamedMetadata("nvvm.annotations"))
        if (existing->getNumOperands() > 0)
            return;

    std::vector<llvm::Function *> kernel_fns;
    for (llvm::Function &fn : mod) {
        if (fn.getCallingConv() == llvm::CallingConv::PTX_Kernel)
            kernel_fns.push_back(&fn);
    }
    if (kernel_fns.empty())
        return;

    llvm::Type *i32_ty = llvm::Type::getInt32Ty(ctx);
    llvm::Type *ptr_ty = llvm::PointerType::get(ctx, 0);
    llvm::Metadata *kernel_str = llvm::MDString::get(ctx, "kernel");
    llvm::Constant *one = llvm::ConstantInt::get(i32_ty, 1);
    llvm::NamedMDNode *annotations =
        mod.getOrInsertNamedMetadata("nvvm.annotations");

    std::vector<llvm::Constant *> used_fns;
    used_fns.reserve(kernel_fns.size());
    for (llvm::Function *fn : kernel_fns) {
        llvm::MDNode *node = llvm::MDNode::get(
            ctx, {llvm::ValueAsMetadata::get(fn), kernel_str,
                  llvm::ValueAsMetadata::get(one)});
        annotations->addOperand(node);
        used_fns.push_back(fn);
    }

    llvm::ArrayType *arr_ty = llvm::ArrayType::get(ptr_ty, used_fns.size());
    auto *used = new llvm::GlobalVariable(
        mod, arr_ty, false, llvm::GlobalValue::AppendingLinkage,
        llvm::ConstantArray::get(arr_ty, used_fns), "llvm.used");
    used->setSection("llvm.metadata");
}

static void adapt_nvvmir_version(llvm::Module &mod, llvm::LLVMContext &ctx) {
    if (llvm::NamedMDNode *existing = mod.getNamedMetadata("nvvmir.version"))
        if (existing->getNumOperands() > 0)
            return;

    llvm::Type *i32_ty = llvm::Type::getInt32Ty(ctx);
    llvm::Metadata *two =
        llvm::ValueAsMetadata::get(llvm::ConstantInt::get(i32_ty, 2));
    llvm::Metadata *zero =
        llvm::ValueAsMetadata::get(llvm::ConstantInt::get(i32_ty, 0));
    llvm::MDNode *node = llvm::MDNode::get(ctx, {two, zero});
    mod.getOrInsertNamedMetadata("nvvmir.version")->addOperand(node);
}

static void adapt_debug_info_version(llvm::Module &mod,
                                     llvm::LLVMContext &ctx) {
    llvm::NamedMDNode *flags = mod.getNamedMetadata("llvm.module.flags");
    if (!flags)
        return;

    for (llvm::MDNode *flag : flags->operands()) {
        if (flag->getNumOperands() < 3)
            continue;

        auto *behavior = llvm::dyn_cast_or_null<llvm::ConstantAsMetadata>(
            flag->getOperand(0).get());
        auto *key = llvm::dyn_cast_or_null<llvm::MDString>(
            flag->getOperand(1).get());
        if (!behavior || !key || key->getString() != "Debug Info Version")
            continue;

        auto *behavior_value =
            llvm::dyn_cast<llvm::ConstantInt>(behavior->getValue());
        if (!behavior_value ||
            behavior_value->getZExtValue() != llvm::Module::Warning)
            continue;

        llvm::Metadata *one = llvm::ValueAsMetadata::get(
            llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 1));
        flag->replaceOperandWith(0, one);
    }
}

static void adapt_for_libnvvm(llvm::Module &mod, llvm::LLVMContext &ctx) {
    adapt_barrier_sync(mod, ctx);
    adapt_barrier_reduction(mod, ctx);
    adapt_inline_asm_intrinsics(mod, ctx);
    adapt_atomicrmw(mod, ctx);
    adapt_trunc(mod, ctx);
    adapt_nvvm_annotations(mod, ctx);
    adapt_nvvmir_version(mod, ctx);
    adapt_debug_info_version(mod, ctx);
}

static void downgrade_lifetime(llvm::Module &mod) {
    const char *names[] = {"llvm.lifetime.start.p0", "llvm.lifetime.end.p0"};
    for (const char *name : names) {
        llvm::Function *fn = mod.getFunction(name);
        if (!fn || fn->arg_size() == 2)
            continue;
        for (llvm::CallBase *call : collect_call_users(fn))
            call->eraseFromParent();
        if (fn->use_empty())
            fn->eraseFromParent();
    }
}

static void downgrade_attributes(llvm::Module &mod, int ctk_major) {
    llvm::Attribute::AttrKind nocup_kind =
        llvm::Attribute::getAttrKindFromName("nocreateundeforpoison");
    llvm::Attribute::AttrKind captures_kind =
        llvm::Attribute::getAttrKindFromName("captures");
    bool downgrade_captures = ctk_major < 13;

    auto process = [&](llvm::Function &fn, unsigned idx) {
        if (nocup_kind != llvm::Attribute::None &&
            fn.hasAttributeAtIndex(idx, nocup_kind))
            fn.removeAttributeAtIndex(idx, nocup_kind);
        if (downgrade_captures && captures_kind != llvm::Attribute::None &&
            fn.hasAttributeAtIndex(idx, captures_kind))
            fn.removeAttributeAtIndex(idx, captures_kind);
    };

    for (llvm::Function &fn : mod) {
        for (unsigned i = 0; i <= fn.arg_size(); ++i)
            process(fn, i);
        process(fn, llvm::AttributeList::FunctionIndex);
    }
}

static void downgrade_grid_constant(llvm::Module &mod,
                                    llvm::LLVMContext &ctx) {
    llvm::NamedMDNode *annotations =
        mod.getOrInsertNamedMetadata("nvvm.annotations");
    llvm::Type *i32_ty = llvm::Type::getInt32Ty(ctx);

    for (llvm::Function &fn : mod) {
        if (fn.getCallingConv() != llvm::CallingConv::PTX_Kernel)
            continue;

        std::vector<llvm::Metadata *> gc_indices;
        for (unsigned i = 0; i < fn.arg_size(); ++i) {
            unsigned idx = i + llvm::AttributeList::FirstArgIndex;
            llvm::AttributeSet attrs = fn.getAttributes().getAttributes(idx);
            for (llvm::Attribute attr : attrs) {
                if (!attr.isStringAttribute())
                    continue;
                if (attr.getKindAsString() == "nvvm.grid_constant") {
                    gc_indices.push_back(llvm::ValueAsMetadata::get(
                        llvm::ConstantInt::get(i32_ty, idx)));
                    break;
                }
            }
        }

        if (gc_indices.empty())
            continue;

        llvm::MDNode *idx_node = llvm::MDNode::get(ctx, gc_indices);
        llvm::MDNode *node = llvm::MDNode::get(
            ctx, {llvm::ValueAsMetadata::get(&fn),
                  llvm::MDString::get(ctx, "grid_constant"), idx_node});
        annotations->addOperand(node);
    }
}

static void downgrade_for_libnvvm(llvm::Module &mod, llvm::LLVMContext &ctx,
                                  int ctk_major, int) {
    downgrade_lifetime(mod);
    downgrade_attributes(mod, ctk_major);
    downgrade_grid_constant(mod, ctx);
}

static bool initialize_mlir_context(mlir::MLIRContext &context) {
    context.loadAllAvailableDialects();
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

    mlir::DialectRegistry registry;
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerGPUDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerNVVMDialectTranslation(registry);
    mlir::MLIRContext mlir_context(registry);
    if (!initialize_mlir_context(mlir_context)) {
        set_error(error_out, "failed to initialize MLIR context");
        return 1;
    }

    std::string diagnostics;
    mlir_context.getDiagEngine().registerHandler([&](mlir::Diagnostic &diag) {
        llvm::raw_string_ostream os(diagnostics);
        diag.print(os);
        os << '\n';
        return mlir::success();
    });

    mlir::ParserConfig parser_config(&mlir_context, true);
    mlir::OwningOpRef<mlir::Operation *> mlir_op =
        mlir::parseSourceString<mlir::Operation *>(
            llvm::StringRef(mlir_text, mlir_text_len), parser_config,
            "numba-cuda-mlir-gpu-module.mlir");
    if (!mlir_op) {
        set_error(error_out, diagnostics.empty()
                                 ? "failed to parse MLIR gpu.module"
                                 : diagnostics);
        return 1;
    }

    llvm::LLVMContext llvm_context;
    std::unique_ptr<llvm::Module> llvm_module =
        mlir::translateModuleToLLVMIR(mlir_op.get(), llvm_context,
                                      "numba-cuda-mlir-gpu-module");
    if (!llvm_module) {
        set_error(error_out, diagnostics.empty()
                                 ? "mlir::translateModuleToLLVMIR failed"
                                 : diagnostics);
        return 1;
    }

    if (dump_llvmir && !dump_module_to_stderr(*llvm_module, error_out))
        return 1;

    adapt_for_libnvvm(*llvm_module, llvm_context);
    downgrade_for_libnvvm(*llvm_module, llvm_context, ctk_major, ctk_minor);

    return serialize_module(*llvm_module, out, out_len, error_out) ? 0 : 1;
}

extern "C" MLIR_MODERN_TO_NVVM_EXPORT void
mlir_modern_to_nvvm_free(void *ptr) {
    std::free(ptr);
}
