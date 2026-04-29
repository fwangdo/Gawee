cmake_minimum_required(VERSION 3.20)
project(GaweeMLIRCMakeExercise LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT LLVM_ENABLE_RTTI)
  add_compile_options(-fno-rtti)
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(AddMLIR)

set(EXERCISE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
set(PARENT_MLIR_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/..)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${EXERCISE_ROOT}/include)
include_directories(${PARENT_MLIR_ROOT}/include)
include_directories(${PARENT_MLIR_ROOT}/include/Gawee)
include_directories(${PARENT_MLIR_ROOT}/include/Gawee/generated)

add_library(ExerciseGaweeDialect
  ${PARENT_MLIR_ROOT}/lib/Gawee/GaweeDialect.cpp
)

target_link_libraries(ExerciseGaweeDialect
  PUBLIC
  MLIRIR
  MLIRSupport
)

add_library(ExerciseGaweeConversion
  lib/Conversion/BufferizePrep.cpp
  lib/Conversion/DecomposeAggregatedLinalgOps.cpp
  lib/Conversion/GaweeToLinalg.cpp
  lib/Conversion/LinalgFusion.cpp
  lib/Conversion/LinalgScheduling.cpp
  lib/Conversion/LinalgTransform.cpp
  lib/Conversion/LinalgVectorization.cpp
  lib/Conversion/LinalgVerification.cpp
)

target_link_libraries(ExerciseGaweeConversion
  PUBLIC
  ExerciseGaweeDialect
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRPass
  MLIRRewrite
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
)

add_executable(exercise-opt
  tools/exercise-opt.cpp
)

llvm_map_components_to_libnames(llvm_libs Support Core)

target_link_libraries(exercise-opt
  PRIVATE
  ExerciseGaweeDialect
  ExerciseGaweeConversion
  MLIROptLib
  MLIRAffineToStandard
  MLIRArithDialect
  MLIRArithToLLVM
  MLIRArithTransforms
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRControlFlowDialect
  MLIRControlFlowToLLVM
  MLIRFuncDialect
  MLIRFuncToLLVM
  MLIRIR
  MLIRLLVMIRTransforms
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRMathDialect
  MLIRMathToLLVM
  MLIRMathToLibm
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIROptLib
  MLIRParser
  MLIRPass
  MLIRReconcileUnrealizedCasts
  MLIRRewrite
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRSupport
  MLIRTensorDialect
  MLIRTensorTransforms
  MLIRTransformUtils
  MLIRTransforms
  ${llvm_libs}
)

add_executable(exercise-sandbox
  src/main.cpp
)

target_link_libraries(exercise-sandbox
  PRIVATE
  ExerciseGaweeDialect
  ExerciseGaweeConversion
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRPass
  MLIRParser
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
)

