#pragma once

#include <cuda_runtime.h>

#include <sstream>
#include <stdexcept>

namespace gpu_fluids {

inline void checkCuda(cudaError_t result, const char* expression, const char* file, int line) {
  if (result == cudaSuccess) {
    return;
  }
  std::ostringstream message;
  message << "CUDA failure at " << file << ":" << line << " for " << expression << ": "
          << cudaGetErrorName(result) << " - " << cudaGetErrorString(result);
  throw std::runtime_error(message.str());
}

inline void checkKernel(const char* kernelName, const char* file, int line) {
  checkCuda(cudaGetLastError(), kernelName, file, line);
}

}  // namespace gpu_fluids

#define FLUID_CUDA_CHECK(expression) ::gpu_fluids::checkCuda((expression), #expression, __FILE__, __LINE__)
#define FLUID_CUDA_KERNEL_CHECK(name) ::gpu_fluids::checkKernel((name), __FILE__, __LINE__)
