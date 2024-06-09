cmake_minimum_required(VERSION 3.24)

option(FLUID_ENABLE_FAST_MATH "Enable CUDA fast math for visual experiments" OFF)
option(FLUID_BUILD_TESTS "Build native configuration contract tests" ON)
option(FLUID_BUILD_REFERENCE "Build the deterministic C++ reference runtime" ON)
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA compute capabilities to build")

project(gpu_stable_fluids_native LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(fluid_cuda STATIC
  cuda/src/gpu_diagnostics.cpp
  cuda/src/launch_policy.cpp
  cuda/src/solver.cu
  cuda/src/sph_solver.cu
  cuda/src/visualization.cpp
)

target_include_directories(fluid_cuda
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/include
)

target_compile_features(fluid_cuda PUBLIC cxx_std_17)
set_target_properties(fluid_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  POSITION_INDEPENDENT_CODE ON
)

target_compile_options(fluid_cuda PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)

if(MSVC)
  target_compile_options(fluid_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:/W4>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
  )
else()
  target_compile_options(fluid_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
  )
endif()

if(FLUID_ENABLE_FAST_MATH)
  target_compile_options(fluid_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )
endif()

target_link_libraries(fluid_cuda PUBLIC CUDA::cudart)

if(FLUID_BUILD_REFERENCE)
  add_library(fluid_reference STATIC
    cuda/src/benchmark.cpp
    cuda/src/cpu_solver.cpp
    cuda/src/experiment_manifest.cpp
    cuda/src/field_analysis.cpp
    cuda/src/frame_metrics.cpp
    cuda/src/native_runtime.cpp
    cuda/src/numeric_validation.cpp
    cuda/src/performance_model.cpp
    cuda/src/telemetry.cpp
    cuda/src/trace_recorder.cpp
    cuda/src/visualization.cpp
  )
  target_include_directories(fluid_reference
    PUBLIC
      ${CMAKE_CURRENT_SOURCE_DIR}/cuda/include
  )
  target_compile_features(fluid_reference PUBLIC cxx_std_17)
  set_target_properties(fluid_reference PROPERTIES POSITION_INDEPENDENT_CODE ON)

  if(MSVC)
    target_compile_options(fluid_reference PRIVATE /W4)
  else()
    target_compile_options(fluid_reference PRIVATE -Wall -Wextra -Wpedantic)
  endif()

  add_executable(fluid_reference_demo
    cuda/src/reference_main.cpp
  )
  target_link_libraries(fluid_reference_demo PRIVATE fluid_reference)

  if(MSVC)
    target_compile_options(fluid_reference_demo PRIVATE /W4)
  else()
    target_compile_options(fluid_reference_demo PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endif()

add_executable(fluid_cuda_demo
  cuda/src/main.cpp
)
target_link_libraries(fluid_cuda_demo PRIVATE fluid_cuda)

if(MSVC)
  target_compile_options(fluid_cuda_demo PRIVATE /W4)
else()
  target_compile_options(fluid_cuda_demo PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING AND FLUID_BUILD_TESTS)
  add_executable(fluid_native_config_contract
    tests/native/config_contract.cpp
  )
  target_include_directories(fluid_native_config_contract PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/include
  )
  target_compile_features(fluid_native_config_contract PRIVATE cxx_std_17)
  add_test(NAME native_config_contract COMMAND fluid_native_config_contract)

  if(FLUID_BUILD_REFERENCE)
    add_executable(fluid_reference_contract
      tests/native/cpu_solver_contract.cpp
    )
    target_link_libraries(fluid_reference_contract PRIVATE fluid_reference)
    target_compile_features(fluid_reference_contract PRIVATE cxx_std_17)
    add_test(NAME native_reference_contract COMMAND fluid_reference_contract)

    add_executable(fluid_runtime_contract
      tests/native/native_runtime_contract.cpp
    )
    target_link_libraries(fluid_runtime_contract PRIVATE fluid_reference)
    target_compile_features(fluid_runtime_contract PRIVATE cxx_std_17)
    add_test(NAME native_runtime_contract COMMAND fluid_runtime_contract)

    add_executable(fluid_performance_model_contract
      tests/native/performance_model_contract.cpp
    )
    target_link_libraries(fluid_performance_model_contract PRIVATE fluid_reference)
    target_compile_features(fluid_performance_model_contract PRIVATE cxx_std_17)
    add_test(NAME native_performance_model_contract COMMAND fluid_performance_model_contract)

    add_executable(fluid_numeric_validation_contract
      tests/native/numeric_validation_contract.cpp
    )
    target_link_libraries(fluid_numeric_validation_contract PRIVATE fluid_reference)
    target_compile_features(fluid_numeric_validation_contract PRIVATE cxx_std_17)
    add_test(NAME native_numeric_validation_contract COMMAND fluid_numeric_validation_contract)

    add_executable(fluid_experiment_manifest_contract
      tests/native/experiment_manifest_contract.cpp
    )
    target_link_libraries(fluid_experiment_manifest_contract PRIVATE fluid_reference)
    target_compile_features(fluid_experiment_manifest_contract PRIVATE cxx_std_17)
    add_test(NAME native_experiment_manifest_contract COMMAND fluid_experiment_manifest_contract)

    add_executable(fluid_trace_recorder_contract
      tests/native/trace_recorder_contract.cpp
    )
    target_link_libraries(fluid_trace_recorder_contract PRIVATE fluid_reference)
    target_compile_features(fluid_trace_recorder_contract PRIVATE cxx_std_17)
    add_test(NAME native_trace_recorder_contract COMMAND fluid_trace_recorder_contract)

    add_executable(fluid_benchmark_contract
      tests/native/benchmark_contract.cpp
    )
    target_link_libraries(fluid_benchmark_contract PRIVATE fluid_reference)
    target_compile_features(fluid_benchmark_contract PRIVATE cxx_std_17)
    add_test(NAME native_benchmark_contract COMMAND fluid_benchmark_contract)

    add_executable(fluid_field_analysis_contract
      tests/native/field_analysis_contract.cpp
    )
    target_link_libraries(fluid_field_analysis_contract PRIVATE fluid_reference)
    target_compile_features(fluid_field_analysis_contract PRIVATE cxx_std_17)
    add_test(NAME native_field_analysis_contract COMMAND fluid_field_analysis_contract)

    add_executable(fluid_frame_metrics_contract
      tests/native/frame_metrics_contract.cpp
    )
    target_link_libraries(fluid_frame_metrics_contract PRIVATE fluid_reference)
    target_compile_features(fluid_frame_metrics_contract PRIVATE cxx_std_17)
    add_test(NAME native_frame_metrics_contract COMMAND fluid_frame_metrics_contract)
  endif()
endif()

set(FLUID_INSTALL_TARGETS fluid_cuda fluid_cuda_demo)
if(FLUID_BUILD_REFERENCE)
  list(APPEND FLUID_INSTALL_TARGETS fluid_reference fluid_reference_demo)
endif()

install(TARGETS ${FLUID_INSTALL_TARGETS}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(DIRECTORY cuda/include/gpu_fluids DESTINATION include)
