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

add_library(llm_lab_serving
  src/serving/continuous_batching.cpp
  src/serving/fcfs_scheduler.cpp
  src/serving/request.cpp
  src/serving/simulated_backend.cpp
  src/serving/simulation_engine.cpp
)
target_include_directories(llm_lab_serving PUBLIC include)
target_compile_options(llm_lab_serving PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

add_executable(serving-sim-smoke src/serving/serving_sim_smoke.cpp)
target_link_libraries(serving-sim-smoke PRIVATE llm_lab_serving)
target_compile_options(serving-sim-smoke PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
)

add_executable(continuous-batching-smoke
  src/serving/continuous_batching_smoke.cpp
)
target_link_libraries(continuous-batching-smoke PRIVATE llm_lab_serving)
target_compile_options(continuous-batching-smoke PRIVATE
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

add_library(q8_rn2_scale_kernels
  kernels/q8_rn2_scale/q8_rn2_scale.cpp
)
target_include_directories(q8_rn2_scale_kernels PUBLIC kernels/q8_rn2_scale)
target_compile_options(q8_rn2_scale_kernels PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-mavx2;-mfma;-mf16c>
)

add_executable(q8_rn2_scale_test kernels/q8_rn2_scale/tests/test_q8_rn2_scale.cpp)
target_link_libraries(q8_rn2_scale_test PRIVATE q8_rn2_scale_kernels)
target_compile_options(q8_rn2_scale_test PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-mavx2;-mfma;-mf16c>)

add_executable(q8_rn2_scale_bench kernels/q8_rn2_scale/benchmark/bench_q8_rn2_scale.cpp)
target_link_libraries(q8_rn2_scale_bench PRIVATE q8_rn2_scale_kernels)
target_compile_options(q8_rn2_scale_bench PRIVATE $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic;-mavx2;-mfma;-mf16c>)

include(CTest)
if(BUILD_TESTING)
  add_executable(serving_tests tests/serving/test_simulator.cpp)
  target_link_libraries(serving_tests PRIVATE llm_lab_serving)
  target_compile_options(serving_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
  )

  add_executable(serving_scheduler_tests tests/serving/test_scheduler.cpp)
  target_link_libraries(serving_scheduler_tests PRIVATE llm_lab_serving)
  target_compile_options(serving_scheduler_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
  )

  add_executable(continuous_batching_tests
    tests/serving/test_continuous_batching.cpp
  )
  target_link_libraries(continuous_batching_tests PRIVATE llm_lab_serving)
  target_compile_options(continuous_batching_tests PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall;-Wextra;-Wpedantic>
  )

  add_test(NAME native_smoke COMMAND native_smoke)
  add_test(NAME serving_simulator COMMAND serving_tests)
  add_test(NAME serving_scheduler COMMAND serving_scheduler_tests)
  add_test(NAME serving_continuous_batching COMMAND continuous_batching_tests)
  add_test(NAME kernel_q4_hotpath_correctness COMMAND kernel_q4_hotpath_test)
  add_test(NAME q8_rn2_scale_correctness COMMAND q8_rn2_scale_test)
endif()
