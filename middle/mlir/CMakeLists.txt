cmake_minimum_required(VERSION 3.20)
project(GaweeMLIR LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-Wno-reserved-user-defined-literal)

# Match LLVM's RTTI setting (LLVM was built with RTTI disabled)
if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()

#===----------------------------------------------------------------------===#
# Find MLIR
#===----------------------------------------------------------------------===#

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

#===----------------------------------------------------------------------===#
# Include paths
#===----------------------------------------------------------------------===#

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/include/Gawee)
include_directories(${CMAKE_SOURCE_DIR}/include/Gawee/generated)

#===----------------------------------------------------------------------===#
# Gawee Dialect Library
#===----------------------------------------------------------------------===#

add_library(GaweeDialect
  lib/Gawee/GaweeDialect.cpp
)

# dependency A -> B. 
target_link_libraries(GaweeDialect
  PUBLIC
  MLIRIR
  MLIRSupport
)

#===----------------------------------------------------------------------===#
# Conversion Library (for later)
#===----------------------------------------------------------------------===#

add_library(GaweeConversion
  lib/Conversion/GaweeToLinalg.cpp
)

target_link_libraries(GaweeConversion
  PUBLIC
  GaweeDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRArithDialect
  MLIRTransforms
)

#===----------------------------------------------------------------------===#
# gawee-opt Tool
#===----------------------------------------------------------------------===#

add_executable(gawee-opt
  tools/gawee-opt.cpp
)

# Get proper LLVM library handling
llvm_map_components_to_libnames(llvm_libs Support Core)

target_link_libraries(gawee-opt
  PRIVATE
  GaweeDialect
  GaweeConversion
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRRewrite
  MLIRTransforms
  MLIRTransformUtils
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRArithDialect
  MLIRArithTransforms
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  # LLVM dialect and conversions (Phase 7)
  MLIRLLVMDialect
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRReconcileUnrealizedCasts
  MLIRIR
  MLIRSupport
  ${llvm_libs}
)

#===----------------------------------------------------------------------===#
# Emitter Library (JSON -> MLIR)
#===----------------------------------------------------------------------===#

add_library(GaweeEmit
  lib/Emit/MLIREmitter.cpp
)

target_include_directories(GaweeEmit
  PUBLIC
  ${CMAKE_SOURCE_DIR}/include
)

target_link_libraries(GaweeEmit
  PUBLIC
  GaweeDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRIR
  MLIRSupport
)

#===----------------------------------------------------------------------===#
# gawee-translate Tool (JSON -> MLIR)
#===----------------------------------------------------------------------===#

add_executable(gawee-translate
  tools/gawee-translate.cpp
)

target_link_libraries(gawee-translate
  PRIVATE
  GaweeEmit
  GaweeDialect
  MLIRFuncDialect
  MLIRArithDialect
  MLIRLinalgDialect
  MLIRTensorDialect
  MLIRIR
  MLIRSupport
  ${llvm_libs}
)
