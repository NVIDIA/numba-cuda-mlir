# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# numba_cuda_mlir._mlir: the MLIR Python package numba-cuda-mlir ships.
#
# This directory is an LLVM external project (LLVM_EXTERNAL_PROJECTS, see
# ci/build-llvm-modern.sh). It assembles the package from the pieces upstream
# declares in mlir/python/CMakeLists.txt, but only for the dialects
# numba-cuda-mlir imports. Registration.cpp replaces _mlirRegisterEverything,
# which would link every upstream dialect, conversion and pass.

if(NOT MLIR_ENABLE_BINDINGS_PYTHON)
  message(FATAL_ERROR "numba_cuda_mlir_python needs MLIR_ENABLE_BINDINGS_PYTHON=ON")
endif()

set(MLIR_MAIN_SRC_DIR "${LLVM_MAIN_SRC_DIR}/../mlir")
list(APPEND CMAKE_MODULE_PATH "${MLIR_MAIN_SRC_DIR}/cmake/modules")
include(AddMLIR)
include(AddMLIRPython)
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

include_directories(
  "${MLIR_MAIN_SRC_DIR}/include"
  "${LLVM_BINARY_DIR}/tools/mlir/include"
  "${CMAKE_CURRENT_SOURCE_DIR}"
)
add_compile_definitions("MLIR_PYTHON_PACKAGE_PREFIX=${MLIR_PYTHON_PACKAGE_PREFIX}.")

if(WIN32)
  # Upstream sources are outside this directory, so object paths embed their full
  # path, and _mlirDialectsTransform reuses _mlir's object names in a longer dir.
  # Have CMake hash them early enough to stay under MSVC's 260-character limit.
  set(CMAKE_OBJECT_PATH_MAX 230)
endif()

################################################################################
# Dialect and pass registration (C API, embedded in the MLIR Python CAPI library)
################################################################################

add_mlir_public_c_api_library(NumbaCudaMlirCAPIRegistration
  Registration.cpp

  # SiteInitialize.cpp, in this directory, belongs to the Python extension.
  PARTIAL_SOURCES_INTENDED

  DEPENDS
  mlir-headers

  LINK_LIBS PUBLIC
  MLIRCAPIIR

  # Dialects
  MLIRAffineDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRComplexDialect
  MLIRControlFlowDialect
  MLIRDLTIDialect
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRIndexDialect
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRNVGPUDialect
  MLIRNVVMDialect
  MLIRSCFDialect
  MLIRShapeDialect
  MLIRTensorDialect
  MLIRUBDialect
  MLIRVectorDialect

  # External interface models and extensions
  MLIRArithTransforms
  MLIRArithValueBoundsOpInterfaceImpl
  MLIRBufferizationTransforms
  MLIRCastInterfaces
  MLIRControlFlowTransforms
  MLIRFuncInlinerExtension
  MLIRGPUTransforms
  MLIRLLVMIRTransforms
  MLIRLinalgTransforms
  MLIRMemRefTransforms
  MLIRNVGPUTransforms
  MLIRSCFTransforms
  MLIRShapeOpsTransforms
  MLIRTensorInferTypeOpInterfaceImpl
  MLIRTensorTilingInterfaceImpl
  MLIRTensorTransforms
  MLIRVectorTransforms

  # Conversions
  MLIRAffineToStandard
  MLIRArithToLLVM
  MLIRComplexToLLVM
  MLIRComplexToStandard
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUToNVVMTransforms
  MLIRIndexToLLVM
  MLIRMathToLLVM
  MLIRMathToNVVM
  MLIRMathTransforms
  MLIRMemRefToLLVM
  MLIRNVGPUToNVVM
  MLIRNVVMToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRSCFToControlFlow
  MLIRShapeToStandard
  MLIRTransforms
  MLIRUBToLLVM
  MLIRVectorToLLVM
  MLIRVectorToSCF

  # LLVM IR translations
  MLIRBuiltinToLLVMIRTranslation
  MLIRGPUToLLVMIRTranslation
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMToLLVMIRTranslation
)

################################################################################
# Python package
################################################################################

declare_mlir_python_sources(NumbaCudaMlirPythonSources)

declare_mlir_python_extension(NumbaCudaMlirPythonSources.SiteInitialize
  MODULE_NAME _site_initialize_0
  ADD_TO_PARENT NumbaCudaMlirPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  SOURCES
    SiteInitialize.cpp
  PRIVATE_LINK_LIBS
    LLVMSupport
  EMBED_CAPI_LINK_LIBS
    MLIRCAPIIR
    # The rewrite and conversion C API the core _mlir module calls; upstream's
    # package gets it through _mlirRegisterEverything.
    MLIRCAPITransforms
    NumbaCudaMlirCAPIRegistration
)

# The dialects numba-cuda-mlir imports from numba_cuda_mlir._mlir.dialects.
set(_ncm_declared_sources
  NumbaCudaMlirPythonSources
  MLIRPythonSources.Core
  MLIRPythonSources.ExecutionEngine
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.bufferization
  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.cf
  MLIRPythonSources.Dialects.complex
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.gpu
  MLIRPythonSources.Dialects.linalg
  MLIRPythonSources.Dialects.llvm
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.nvgpu
  MLIRPythonSources.Dialects.nvvm
  MLIRPythonSources.Dialects.scf
  MLIRPythonSources.Dialects.shape
  MLIRPythonSources.Dialects.tensor
  MLIRPythonSources.Dialects.vector
)

set(_ncm_package_root "${CMAKE_CURRENT_BINARY_DIR}/${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}")

add_mlir_python_common_capi_library(NumbaCudaMlirPythonCAPI
  INSTALL_COMPONENT NumbaCudaMlirPythonModules
  INSTALL_DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
  OUTPUT_DIRECTORY "${_ncm_package_root}/_mlir_libs"
  RELATIVE_INSTALL_ROOT "../../../.."
  DECLARED_HEADERS
    MLIRPythonCAPI.HeaderSources
  DECLARED_SOURCES
    ${_ncm_declared_sources}
)
# Loaders reuse an already-loaded library of the same name (glibc by soname,
# Windows by module name), and every MLIR Python embedder gets upstream's
# default name, so make ours unique like the nanobind and support libraries.
set_target_properties(NumbaCudaMlirPythonCAPI PROPERTIES
  OUTPUT_NAME MLIRPythonCAPI-${MLIR_BINDINGS_PYTHON_NB_DOMAIN})

add_mlir_python_modules(NumbaCudaMlirPythonModules
  ROOT_PREFIX "${_ncm_package_root}"
  INSTALL_PREFIX "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}"
  DECLARED_SOURCES
    ${_ncm_declared_sources}
  COMMON_CAPI_LINK_LIBS
    NumbaCudaMlirPythonCAPI
)

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
  target_link_options(NumbaCudaMlirPythonCAPI PRIVATE ${_llvm_c_api})

  # The import library cext/mlir-llvm70 links against is not installed with the
  # DLL; put it in this package's component so the distribution includes it.
  install(FILES "$<TARGET_LINKER_FILE:NumbaCudaMlirPythonCAPI>"
    DESTINATION "${MLIR_BINDINGS_PYTHON_INSTALL_PREFIX}/_mlir_libs"
    COMPONENT NumbaCudaMlirPythonModules)
endif()

################################################################################
# Distribution components
################################################################################

# ci/build-llvm-modern.sh builds and installs only LLVM_DISTRIBUTION_COMPONENTS.
# Besides this package, that is what the C++ code in cext/ links against
# through find_package(MLIR). Keep this list in sync with the link libraries in
# cext/mlir-llvm70/{lib,tools}/CMakeLists.txt.
set(NUMBA_CUDA_MLIR_CXX_LINK_LIBS
  LLVMSupport
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRIR
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIROpenMPDialect
  MLIRParser
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRToLLVMIRTranslationRegistration
)

# Installed libraries reachable from the given ones through link dependencies.
# The exported CMake package must contain all of them.
function(_ncm_installed_link_closure out_var)
  set(_seen)
  set(_queue ${ARGN})
  list(LENGTH _queue _remaining)
  while(_remaining GREATER 0)
    list(POP_FRONT _queue _lib)
    list(LENGTH _queue _remaining)
    if(_lib IN_LIST _seen OR NOT TARGET "${_lib}" OR NOT TARGET "install-${_lib}")
      continue()
    endif()
    get_target_property(_type ${_lib} TYPE)
    if(NOT _type MATCHES "^(STATIC|SHARED|INTERFACE)_LIBRARY$")
      continue()
    endif()
    list(APPEND _seen ${_lib})
    foreach(_property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
      get_target_property(_deps ${_lib} ${_property})
      if(NOT _deps)
        continue()
      endif()
      foreach(_dep IN LISTS _deps)
        string(REGEX REPLACE "^\\$<LINK_ONLY:(.+)>$" "\\1" _dep "${_dep}")
        if(NOT _dep MATCHES "^\\$<")
          list(APPEND _queue ${_dep})
        endif()
      endforeach()
    endforeach()
    list(LENGTH _queue _remaining)
  endwhile()
  set(${out_var} ${_seen} PARENT_SCOPE)
endfunction()

_ncm_installed_link_closure(_ncm_cxx_libs ${NUMBA_CUDA_MLIR_CXX_LINK_LIBS})
list(SORT _ncm_cxx_libs)
set(_ncm_distribution_components
  NumbaCudaMlirPythonModules
  cmake-exports
  llvm-headers
  mlir-cmake-exports
  mlir-headers
  mlir-tblgen
  ${_ncm_cxx_libs}
)
file(WRITE "${CMAKE_BINARY_DIR}/numba_cuda_mlir_distribution_components.txt"
  "${_ncm_distribution_components}")
