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

include(CTest)
if(BUILD_TESTING)
  add_test(NAME native_smoke COMMAND native_smoke)
endif()
