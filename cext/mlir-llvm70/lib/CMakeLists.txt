# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(LLVM70_TRANSLATE_SOURCES
  LLVM70IRBuilder.cpp
  LibNVVMCompiler.cpp
  LLVM70Target.cpp
  Dialect/LLVM70Dialect.cpp
  Target/Target.cpp
)

add_library(LLVM70TranslateObjects OBJECT
  ${LLVM70_TRANSLATE_SOURCES}
)

add_dependencies(LLVM70TranslateObjects LLVM70OpsIncGen)

target_include_directories(LLVM70TranslateObjects PUBLIC
  ${LLVM70_SOURCE_DIR}/include
  ${LLVM70_BINARY_DIR}/include
)

set_target_properties(LLVM70TranslateObjects PROPERTIES
  POSITION_INDEPENDENT_CODE ON
)
if(MSVC)
  target_compile_options(LLVM70TranslateObjects PRIVATE /wd4706)
endif()

add_library(LLVM70Translate
  $<TARGET_OBJECTS:LLVM70TranslateObjects>
)

target_include_directories(LLVM70Translate PUBLIC
  ${LLVM70_SOURCE_DIR}/include
  ${LLVM70_BINARY_DIR}/include
)

target_link_libraries(LLVM70Translate
  PUBLIC
  MLIRGPUDialect
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIROpenMPDialect
  MLIRIR
  MLIRSupport
  LLVMSupport
  ${CMAKE_DL_LIBS}
)

install(TARGETS LLVM70Translate DESTINATION lib)

# Shared library with C API for in-process use from Python.
# Set LLVM70_MLIR_PYTHON_CAPI to the path of libMLIRPythonCAPI.so to link
# against it and share TypeIDs with the Python MLIR bindings, enabling
# raw Operation* passing without text roundtrip.
add_library(MLIRToLLVM70 SHARED CAPI.cpp)
add_dependencies(MLIRToLLVM70 LLVM70TranslateObjects)
target_include_directories(MLIRToLLVM70 PRIVATE
  ${LLVM70_SOURCE_DIR}/include
  ${LLVM70_BINARY_DIR}/include
)
if(WIN32)
  set_target_properties(MLIRToLLVM70 PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
if(MSVC)
  target_compile_options(MLIRToLLVM70 PRIVATE /wd4706)
endif()

# On Windows, the in-process Python C API bridge must share MLIR C++ symbols
# with MLIRPythonCAPI.dll.  Link the translator objects directly instead of
# pulling in LLVM70Translate's static MLIR library dependencies.
if(MSVC AND LLVM70_MLIR_PYTHON_CAPI)
  target_sources(MLIRToLLVM70 PRIVATE $<TARGET_OBJECTS:LLVM70TranslateObjects>)
  message(STATUS "MLIRToLLVM70: linking against ${LLVM70_MLIR_PYTHON_CAPI}")
  target_link_libraries(MLIRToLLVM70 PRIVATE ${LLVM70_MLIR_PYTHON_CAPI} ${CMAKE_DL_LIBS})
else()
  target_link_libraries(MLIRToLLVM70 PRIVATE LLVM70Translate ${CMAKE_DL_LIBS})
  if(LLVM70_MLIR_PYTHON_CAPI)
    message(STATUS "MLIRToLLVM70: linking against ${LLVM70_MLIR_PYTHON_CAPI}")
    target_link_libraries(MLIRToLLVM70 PRIVATE ${LLVM70_MLIR_PYTHON_CAPI})
  endif()
endif()

install(TARGETS MLIRToLLVM70 DESTINATION lib)
