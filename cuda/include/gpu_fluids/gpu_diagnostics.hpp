#pragma once

#include "gpu_fluids/config.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu_fluids {

struct GpuDiagnosticsSnapshot {
  int deviceIndex = 0;
  char name[256]{};
  int computeCapabilityMajor = 0;
  int computeCapabilityMinor = 0;
  int multiprocessors = 0;
  int warpSize = 0;
  int maxThreadsPerBlock = 0;
  int maxThreadsPerMultiprocessor = 0;
  int sharedMemoryPerBlock = 0;
  int sharedMemoryPerMultiprocessor = 0;
  int registersPerBlock = 0;
  int l2CacheBytes = 0;
  int memoryBusWidth = 0;
  int memoryClockKHz = 0;
  int memoryCopyEngines = 0;
  int concurrentKernels = 0;
  int unifiedAddressing = 0;
  int cooperativeLaunch = 0;
  int runtimeVersion = 0;
  int driverVersion = 0;
  std::size_t totalMemoryBytes = 0;
  std::size_t freeMemoryBytes = 0;
};

struct LaunchBudget {
  int blockThreads = 0;
  std::size_t dynamicSharedBytes = 0;
  int registersPerThread = 0;
  int blocksPerMultiprocessor = 0;
  int residentThreadsPerMultiprocessor = 0;
  float theoreticalOccupancy = 0.0F;
  bool legal = false;
};

class GpuDiagnostics final {
 public:
  explicit GpuDiagnostics(int deviceIndex = 0);

  void refresh();

  [[nodiscard]] int deviceIndex() const noexcept { return snapshot_.deviceIndex; }
  [[nodiscard]] const GpuDiagnosticsSnapshot& snapshot() const noexcept { return snapshot_; }
  [[nodiscard]] LaunchBudget evaluateLaunch(int blockThreads, std::size_t dynamicSharedBytes,
                                             int registersPerThread) const noexcept;
  [[nodiscard]] std::vector<std::string> validate() const;
  [[nodiscard]] std::string summary() const;
  [[nodiscard]] std::string toJson() const;

  [[nodiscard]] static std::string formatBytes(std::size_t bytes);
  [[nodiscard]] static int encodeVersion(int major, int minor) noexcept;

 private:
  [[nodiscard]] int attribute(cudaDeviceAttr attribute) const;
  [[nodiscard]] static std::string jsonEscape(const char* value);

  GpuDiagnosticsSnapshot snapshot_{};
};

}  // namespace gpu_fluids
