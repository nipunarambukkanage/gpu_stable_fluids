cmake_minimum_required(VERSION 3.24)

option(FLUID_ENABLE_FAST_MATH "Enable CUDA fast math for visual experiments" OFF)
option(FLUID_BUILD_TESTS "Build native configuration contract tests" ON)
set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA compute capabilities to build")

project(gpu_stable_fluids_native LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

add_library(fluid_cuda STATIC
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
endif()

install(TARGETS fluid_cuda_demo RUNTIME DESTINATION bin)
install(DIRECTORY cuda/include/gpu_fluids DESTINATION include)
