#include "gpu_fluids/gpu_diagnostics.hpp"

#include "gpu_fluids/cuda_utils.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace gpu_fluids {
namespace {

int readAttribute(cudaDeviceAttr attribute, int deviceIndex) {
  int value = 0;
  FLUID_CUDA_CHECK(cudaDeviceGetAttribute(&value, attribute, deviceIndex));
  return value;
}

}  // namespace

GpuDiagnostics::GpuDiagnostics(int deviceIndex) {
  if (deviceIndex < 0) {
    throw std::invalid_argument("CUDA device index cannot be negative");
  }
  snapshot_.deviceIndex = deviceIndex;
  refresh();
}

void GpuDiagnostics::refresh() {
  int deviceCount = 0;
  FLUID_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  if (snapshot_.deviceIndex >= deviceCount) {
    std::ostringstream message;
    message << "CUDA device " << snapshot_.deviceIndex << " is unavailable; detected " << deviceCount << " device(s)";
    throw std::runtime_error(message.str());
  }

  cudaDeviceProp properties{};
  FLUID_CUDA_CHECK(cudaGetDeviceProperties(&properties, snapshot_.deviceIndex));
  std::copy_n(properties.name, sizeof(snapshot_.name) - 1, snapshot_.name);
  snapshot_.name[sizeof(snapshot_.name) - 1] = '\0';
  snapshot_.computeCapabilityMajor = properties.major;
  snapshot_.computeCapabilityMinor = properties.minor;
  snapshot_.multiprocessors = properties.multiProcessorCount;
  snapshot_.warpSize = properties.warpSize;
  snapshot_.maxThreadsPerBlock = properties.maxThreadsPerBlock;
  snapshot_.maxThreadsPerMultiprocessor = properties.maxThreadsPerMultiProcessor;
  snapshot_.sharedMemoryPerBlock = static_cast<int>(properties.sharedMemPerBlock);
  snapshot_.sharedMemoryPerMultiprocessor = static_cast<int>(properties.sharedMemPerMultiprocessor);
  snapshot_.registersPerBlock = properties.regsPerBlock;
  snapshot_.l2CacheBytes = properties.l2CacheSize;
  snapshot_.memoryBusWidth = properties.memoryBusWidth;
  snapshot_.memoryClockKHz = properties.memoryClockRate;
  snapshot_.memoryCopyEngines = properties.asyncEngineCount;
  snapshot_.concurrentKernels = properties.concurrentKernels;
  snapshot_.unifiedAddressing = properties.unifiedAddressing;
  snapshot_.cooperativeLaunch = properties.cooperativeLaunch;
  FLUID_CUDA_CHECK(cudaRuntimeGetVersion(&snapshot_.runtimeVersion));
  FLUID_CUDA_CHECK(cudaDriverGetVersion(&snapshot_.driverVersion));
  FLUID_CUDA_CHECK(cudaMemGetInfo(&snapshot_.freeMemoryBytes, &snapshot_.totalMemoryBytes));
}

int GpuDiagnostics::attribute(cudaDeviceAttr attributeValue) const {
  return readAttribute(attributeValue, snapshot_.deviceIndex);
}

LaunchBudget GpuDiagnostics::evaluateLaunch(int blockThreads, std::size_t dynamicSharedBytes,
                                            int registersPerThread) const noexcept {
  LaunchBudget budget;
  budget.blockThreads = blockThreads;
  budget.dynamicSharedBytes = dynamicSharedBytes;
  budget.registersPerThread = registersPerThread;
  if (blockThreads <= 0 || registersPerThread < 0) {
    return budget;
  }
  const bool threadLimit = blockThreads <= snapshot_.maxThreadsPerBlock;
  const bool sharedLimit = dynamicSharedBytes <= static_cast<std::size_t>(snapshot_.sharedMemoryPerBlock);
  const std::size_t registersPerBlock = static_cast<std::size_t>(blockThreads) *
                                        static_cast<std::size_t>(registersPerThread);
  const bool registerLimit = registersPerBlock <= static_cast<std::size_t>(snapshot_.registersPerBlock);
  budget.legal = threadLimit && sharedLimit && registerLimit;
  if (!budget.legal) {
    return budget;
  }

  const int byThreads = snapshot_.maxThreadsPerMultiprocessor / blockThreads;
  const int byShared = dynamicSharedBytes == 0
                           ? byThreads
                           : snapshot_.sharedMemoryPerMultiprocessor / static_cast<int>(dynamicSharedBytes);
  const int byRegisters = registersPerBlock == 0
                              ? byThreads
                              : snapshot_.registersPerBlock / static_cast<int>(registersPerBlock);
  budget.blocksPerMultiprocessor = std::max(0, std::min({byThreads, byShared, byRegisters}));
  budget.residentThreadsPerMultiprocessor = budget.blocksPerMultiprocessor * blockThreads;
  if (snapshot_.maxThreadsPerMultiprocessor > 0) {
    budget.theoreticalOccupancy = static_cast<float>(budget.residentThreadsPerMultiprocessor) /
                                  static_cast<float>(snapshot_.maxThreadsPerMultiprocessor);
  }
  return budget;
}

std::vector<std::string> GpuDiagnostics::validate() const {
  std::vector<std::string> issues;
  if (snapshot_.computeCapabilityMajor < 7) {
    issues.emplace_back("compute capability below the tested sm_70 baseline");
  }
  if (snapshot_.warpSize != 32) {
    issues.emplace_back("unexpected warp width; tune block mappings before benchmarking");
  }
  if (snapshot_.sharedMemoryPerBlock < 32 * 1024) {
    issues.emplace_back("less than 32 KiB shared memory per block is available");
  }
  if (snapshot_.l2CacheBytes <= 0) {
    issues.emplace_back("L2 cache capacity was not reported by the runtime");
  }
  if (snapshot_.freeMemoryBytes < static_cast<std::size_t>(kCellCount) * sizeof(float) * 8) {
    issues.emplace_back("free VRAM is low for the persistent stable-fluid workspace");
  }
  return issues;
}

std::string GpuDiagnostics::formatBytes(std::size_t bytes) {
  constexpr double kibibyte = 1024.0;
  constexpr double mebibyte = kibibyte * 1024.0;
  constexpr double gibibyte = mebibyte * 1024.0;
  std::ostringstream value;
  value << std::fixed << std::setprecision(2);
  if (bytes >= static_cast<std::size_t>(gibibyte)) {
    value << static_cast<double>(bytes) / gibibyte << " GiB";
  } else if (bytes >= static_cast<std::size_t>(mebibyte)) {
    value << static_cast<double>(bytes) / mebibyte << " MiB";
  } else {
    value << static_cast<double>(bytes) / kibibyte << " KiB";
  }
  return value.str();
}

int GpuDiagnostics::encodeVersion(int major, int minor) noexcept {
  return major * 1000 + minor;
}

std::string GpuDiagnostics::summary() const {
  const auto launch = evaluateLaunch(kBlockSize * kBlockSize,
                                     static_cast<std::size_t>(kTileExtent * kTileExtent) * sizeof(float2),
                                     32);
  std::ostringstream value;
  value << snapshot_.name << " sm_" << snapshot_.computeCapabilityMajor << '.'
        << snapshot_.computeCapabilityMinor << ", " << snapshot_.multiprocessors << " SMs, warp "
        << snapshot_.warpSize << ", VRAM " << formatBytes(snapshot_.freeMemoryBytes) << "/"
        << formatBytes(snapshot_.totalMemoryBytes) << ", tile occupancy "
        << std::fixed << std::setprecision(1) << launch.theoreticalOccupancy * 100.0F << '%';
  return value.str();
}

std::string GpuDiagnostics::jsonEscape(const char* value) {
  std::string escaped;
  for (const char* cursor = value; cursor != nullptr && *cursor != '\0'; ++cursor) {
    if (*cursor == '\\' || *cursor == '"') {
      escaped.push_back('\\');
    }
    escaped.push_back(*cursor);
  }
  return escaped;
}

std::string GpuDiagnostics::toJson() const {
  const LaunchBudget launch = evaluateLaunch(kBlockSize * kBlockSize,
                                             static_cast<std::size_t>(kTileExtent * kTileExtent) * sizeof(float2),
                                             32);
  std::ostringstream value;
  value << std::fixed << std::setprecision(4);
  value << "{\"deviceIndex\": " << snapshot_.deviceIndex
        << ", \"name\": \"" << jsonEscape(snapshot_.name)
        << "\", \"computeCapability\": \"" << snapshot_.computeCapabilityMajor << '.'
        << snapshot_.computeCapabilityMinor << "\", \"multiprocessors\": " << snapshot_.multiprocessors
        << ", \"warpSize\": " << snapshot_.warpSize
        << ", \"sharedMemoryPerBlock\": " << snapshot_.sharedMemoryPerBlock
        << ", \"sharedMemoryPerMultiprocessor\": " << snapshot_.sharedMemoryPerMultiprocessor
        << ", \"l2CacheBytes\": " << snapshot_.l2CacheBytes
        << ", \"freeMemoryBytes\": " << snapshot_.freeMemoryBytes
        << ", \"totalMemoryBytes\": " << snapshot_.totalMemoryBytes
        << ", \"runtimeVersion\": " << snapshot_.runtimeVersion
        << ", \"driverVersion\": " << snapshot_.driverVersion
        << ", \"launch\": {\"legal\": " << (launch.legal ? "true" : "false")
        << ", \"blocksPerMultiprocessor\": " << launch.blocksPerMultiprocessor
        << ", \"occupancy\": " << launch.theoreticalOccupancy << "}}";
  return value.str();
}

}  // namespace gpu_fluids
