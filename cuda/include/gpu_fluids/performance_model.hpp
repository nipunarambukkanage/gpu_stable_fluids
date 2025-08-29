#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gpu_fluids {

enum class MemoryLayout : std::uint8_t {
  StructureOfArrays,
  ArrayOfStructures,
};

enum class Bottleneck : std::uint8_t {
  None,
  MemoryBandwidth,
  ArithmeticThroughput,
  Occupancy,
  LaunchLatency,
};

struct PerformanceDeviceModel {
  int warpSize = 32;
  int maxThreadsPerBlock = 1024;
  int maxThreadsPerMultiprocessor = 2048;
  int sharedMemoryPerBlock = 48 * 1024;
  int sharedMemoryPerMultiprocessor = 100 * 1024;
  int registersPerBlock = 65536;
  int multiprocessors = 1;
  double memoryBandwidthGBPerSecond = 300.0;
  double fp32ThroughputTFLOPS = 8.0;
};

struct KernelWorkload {
  std::string name;
  std::size_t logicalElements = 0;
  std::size_t bytesPerElement = 0;
  std::size_t neighborReads = 0;
  std::size_t floatingPointOperations = 0;
  std::size_t tileWidth = 0;
  std::size_t tileHeight = 0;
  int blockThreads = 128;
  int registersPerThread = 32;
  MemoryLayout layout = MemoryLayout::StructureOfArrays;
  bool usesSharedMemory = false;
};

struct PerformanceEstimate {
  std::string name;
  std::size_t globalBytes = 0;
  std::size_t sharedBytes = 0;
  std::size_t coalescedTransactions = 0;
  double arithmeticIntensity = 0.0;
  double theoreticalOccupancy = 0.0;
  double predictedMemoryMilliseconds = 0.0;
  double predictedComputeMilliseconds = 0.0;
  Bottleneck bottleneck = Bottleneck::None;
  bool legalLaunch = false;
};

struct PipelineEstimate {
  std::vector<PerformanceEstimate> kernels;
  std::size_t totalGlobalBytes = 0;
  double totalPredictedMilliseconds = 0.0;
  double averageOccupancy = 0.0;
  Bottleneck dominantBottleneck = Bottleneck::None;
};

class GpuPerformanceModel final {
 public:
  explicit GpuPerformanceModel(PerformanceDeviceModel device = {});

  [[nodiscard]] const PerformanceDeviceModel& device() const noexcept { return device_; }
  [[nodiscard]] PerformanceEstimate estimate(const KernelWorkload& workload) const;
  [[nodiscard]] PipelineEstimate estimateStableFluidPipeline(std::size_t cellCount) const;
  [[nodiscard]] int recommendedBlockThreads(const KernelWorkload& workload) const noexcept;
  [[nodiscard]] static std::size_t estimateCoalescedTransactions(std::size_t elements,
                                                                  std::size_t bytesPerElement,
                                                                  int warpSize,
                                                                  MemoryLayout layout) noexcept;
  [[nodiscard]] static std::size_t estimateTileBytes(const KernelWorkload& workload) noexcept;
  [[nodiscard]] static const char* bottleneckName(Bottleneck bottleneck) noexcept;
  [[nodiscard]] static const char* layoutName(MemoryLayout layout) noexcept;
  [[nodiscard]] std::string toJson(const PipelineEstimate& pipeline) const;

 private:
  [[nodiscard]] double occupancy(const KernelWorkload& workload) const noexcept;
  [[nodiscard]] Bottleneck classify(const PerformanceEstimate& estimate) const noexcept;

  PerformanceDeviceModel device_{};
};

}  // namespace gpu_fluids
