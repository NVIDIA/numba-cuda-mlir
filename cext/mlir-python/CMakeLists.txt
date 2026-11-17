# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Adjustments to upstream's MLIR Python CAPI library (target MLIRPythonCAPI).
#
# This directory is an LLVM external project (LLVM_EXTERNAL_PROJECTS, see
# ci/build-llvm-modern.sh), so it is processed after mlir/python defines the
# target.

if(NOT TARGET MLIRPythonCAPI)
  message(FATAL_ERROR "numba_cuda_mlir_python needs MLIR_ENABLE_BINDINGS_PYTHON=ON")
endif()

# Loaders reuse an already-loaded library of the same name (glibc by soname,
# Windows by module name), and every MLIR Python embedder gets upstream's
# default name, so make ours unique like the nanobind and support libraries.
set_target_properties(MLIRPythonCAPI PROPERTIES
  OUTPUT_NAME MLIRPythonCAPI-${MLIR_BINDINGS_PYTHON_NB_DOMAIN})

if(WIN32)
  # A DLL exports only what is compiled into it, not the LLVM static libs it
  # links. Export the LLVM C API that cext/launcher/llvm_downgrade.cpp and
  # lowering_utilities/llvm_utils.py load from this library, as the Linux .so
  # already does. LLVMWriteBitcodeToMemoryBuffer is left out on both: the
  # bit writer is not linked in, and llvm_downgrade.cpp falls back to text IR.
  set(_llvm_c_api
    LLVMContextCreate
    LLVMContextDispose
    LLVMDisposeModule
    LLVMPrintModuleToString
    LLVMDisposeMessage
    LLVMGetFirstFunction
    LLVMGetNextFunction
    LLVMGetFirstBasicBlock
    LLVMGetNextBasicBlock
    LLVMGetFirstInstruction
    LLVMGetNextInstruction
    LLVMGetNamedFunction
    LLVMAddFunction
    LLVMCountParams
    LLVMGetFunctionCallConv
    LLVMSetLinkage
    LLVMGetCalledValue
    LLVMGetAttributeCountAtIndex
    LLVMGetAttributesAtIndex
    LLVMIsEnumAttribute
    LLVMIsStringAttribute
    LLVMGetEnumAttributeKind
    LLVMGetStringAttributeKind
    LLVMRemoveEnumAttributeAtIndex
    LLVMGetEnumAttributeKindForName
    LLVMVoidTypeInContext
    LLVMInt1TypeInContext
    LLVMInt32TypeInContext
    LLVMFloatTypeInContext
    LLVMDoubleTypeInContext
    LLVMHalfTypeInContext
    LLVMBFloatTypeInContext
    LLVMFunctionType
    LLVMPointerTypeInContext
    LLVMTypeOf
    LLVMGetTypeKind
    LLVMArrayType2
    LLVMGetReturnType
    LLVMReplaceAllUsesWith
    LLVMGetOperand
    LLVMGetNumOperands
    LLVMConstInt
    LLVMInstructionEraseFromParent
    LLVMGetAtomicRMWBinOp
    LLVMSetAtomicRMWBinOp
    LLVMCreateBuilderInContext
    LLVMPositionBuilderBefore
    LLVMDisposeBuilder
    LLVMBuildCall2
    LLVMBuildZExt
    LLVMBuildTrunc
    LLVMBuildFPExt
    LLVMBuildFPTrunc
    LLVMGetInlineAsm
    LLVMGetFirstUse
    LLVMGetNextUse
    LLVMGetUser
    LLVMMDStringInContext2
    LLVMMDNodeInContext2
    LLVMValueAsMetadata
    LLVMMetadataAsValue
    LLVMAddNamedMetadataOperand
    LLVMGetNamedMetadataNumOperands
    LLVMGetNamedMetadataOperands
    LLVMAddGlobal
    LLVMSetInitializer
    LLVMSetSection
    LLVMConstArray2
    LLVMGetMDNodeNumOperands
    LLVMGetMDNodeOperands
    LLVMReplaceMDNodeOperandWith
    LLVMGetMDString
    LLVMConstIntGetZExtValue
    LLVMIsACallInst
    LLVMIsAAtomicRMWInst
    LLVMIsAConstantInt
    LLVMGetPointerAddressSpace
    LLVMDeleteFunction
    LLVMGetBufferStart
    LLVMGetBufferSize
    LLVMDisposeMemoryBuffer
  )
  list(TRANSFORM _llvm_c_api PREPEND "/EXPORT:")
  target_link_options(MLIRPythonCAPI PRIVATE ${_llvm_c_api})

  # Upstream installs the DLL but not the import library cext/mlir-llvm70
  # links against.
  install(FILES "$<TARGET_LINKER_FILE:MLIRPythonCAPI>"
    DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs")
endif()
