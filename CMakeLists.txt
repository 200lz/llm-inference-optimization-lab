cmake_minimum_required(VERSION 3.25)

project(
  llm_inference_optimization_lab
  VERSION 0.1.0
  DESCRIPTION "Experiments in LLM inference optimization"
  LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(native_smoke src/native_smoke.cpp)
target_compile_options(native_smoke PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

add_library(q4_hotpath_kernels
  kernels/q4_hotpath/reference/q6_q8_reference.cpp
  kernels/q4_hotpath/baseline/q6_q8_baseline.cpp
  kernels/q4_hotpath/optimized/q6_q8_optimized.cpp
)
target_include_directories(q4_hotpath_kernels PUBLIC kernels/q4_hotpath)
target_compile_options(q4_hotpath_kernels PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-mavx2;-mfma;-mf16c>
)

add_executable(kernel_q4_hotpath_test kernels/q4_hotpath/tests/test_q6_q8.cpp)
target_link_libraries(kernel_q4_hotpath_test PRIVATE q4_hotpath_kernels)
target_compile_options(kernel_q4_hotpath_test PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)

add_executable(kernel_q4_hotpath_bench kernels/q4_hotpath/benchmark/bench_q6_q8.cpp)
target_link_libraries(kernel_q4_hotpath_bench PRIVATE q4_hotpath_kernels)
target_compile_options(kernel_q4_hotpath_bench PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME native_smoke COMMAND native_smoke)
  add_test(NAME kernel_q4_hotpath_correctness COMMAND kernel_q4_hotpath_test)
endif()
