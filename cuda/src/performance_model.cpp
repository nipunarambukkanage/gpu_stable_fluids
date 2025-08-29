#include "gpu_fluids/performance_model.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace gpu_fluids {
namespace {

constexpr double kBytesPerGiB = 1024.0 * 1024.0 * 1024.0;
constexpr double kBytesPerMegabyte = 1024.0 * 1024.0;

std::size_t ceilDivide(std::size_t numerator, std::size_t denominator) noexcept {
  return denominator == 0 ? 0 : (numerator + denominator - 1) / denominator;
}

double safeRatio(double numerator, double denominator) noexcept {
  return denominator <= 0.0 ? 0.0 : numerator / denominator;
}

}  // namespace

GpuPerformanceModel::GpuPerformanceModel(PerformanceDeviceModel device) : device_(device) {
  if (device_.warpSize <= 0 || device_.maxThreadsPerBlock <= 0 ||
      device_.maxThreadsPerMultiprocessor <= 0 || device_.sharedMemoryPerBlock <= 0 ||
      device_.sharedMemoryPerMultiprocessor <= 0 || device_.registersPerBlock <= 0 ||
      device_.multiprocessors <= 0 || device_.memoryBandwidthGBPerSecond <= 0.0 ||
      device_.fp32ThroughputTFLOPS <= 0.0) {
    throw std::invalid_argument("GPU performance model device limits must be positive");
  }
}

std::size_t GpuPerformanceModel::estimateCoalescedTransactions(
    std::size_t elements, std::size_t bytesPerElement, int warpSize, MemoryLayout layout) noexcept {
  if (elements == 0 || bytesPerElement == 0 || warpSize <= 0) {
    return 0;
  }
  const std::size_t warps = ceilDivide(elements, static_cast<std::size_t>(warpSize));
  const std::size_t cacheLineBytes = 128;
  const std::size_t contiguousBytes = static_cast<std::size_t>(warpSize) * bytesPerElement;
  const std::size_t transactionsPerWarp = std::max<std::size_t>(
      1, ceilDivide(contiguousBytes, cacheLineBytes));
  const std::size_t layoutPenalty = layout == MemoryLayout::StructureOfArrays ? 1 : 2;
  return warps * transactionsPerWarp * layoutPenalty;
}

std::size_t GpuPerformanceModel::estimateTileBytes(const KernelWorkload& workload) noexcept {
  if (!workload.usesSharedMemory || workload.tileWidth == 0 || workload.tileHeight == 0) {
    return 0;
  }
  const std::size_t tileCells = workload.tileWidth * workload.tileHeight;
  return tileCells * workload.bytesPerElement;
}

double GpuPerformanceModel::occupancy(const KernelWorkload& workload) const noexcept {
  if (workload.blockThreads <= 0 || workload.blockThreads > device_.maxThreadsPerBlock ||
      workload.registersPerThread < 0) {
    return 0.0;
  }
  const std::size_t blockRegisters = static_cast<std::size_t>(workload.blockThreads) *
                                     static_cast<std::size_t>(workload.registersPerThread);
  const std::size_t sharedBytes = estimateTileBytes(workload);
  if (blockRegisters > static_cast<std::size_t>(device_.registersPerBlock) ||
      sharedBytes > static_cast<std::size_t>(device_.sharedMemoryPerBlock)) {
    return 0.0;
  }
  const int byThreads = device_.maxThreadsPerMultiprocessor / workload.blockThreads;
  const int byRegisters = blockRegisters == 0
                              ? byThreads
                              : device_.registersPerBlock / static_cast<int>(blockRegisters);
  const int byShared = sharedBytes == 0
                           ? byThreads
                           : device_.sharedMemoryPerMultiprocessor / static_cast<int>(sharedBytes);
  const int residentBlocks = std::max(0, std::min({byThreads, byRegisters, byShared}));
  const int residentThreads = residentBlocks * workload.blockThreads;
  return safeRatio(static_cast<double>(residentThreads),
                   static_cast<double>(device_.maxThreadsPerMultiprocessor));
}

int GpuPerformanceModel::recommendedBlockThreads(const KernelWorkload& workload) const noexcept {
  constexpr int candidates[] = {32, 64, 128, 256, 512, 1024};
  int recommendation = 32;
  double bestScore = -1.0;
  for (const int candidate : candidates) {
    if (candidate > device_.maxThreadsPerBlock) {
      continue;
    }
    KernelWorkload trial = workload;
    trial.blockThreads = candidate;
    const double score = occupancy(trial) * static_cast<double>(candidate);
    if (score > bestScore) {
      bestScore = score;
      recommendation = candidate;
    }
  }
  return recommendation;
}

PerformanceEstimate GpuPerformanceModel::estimate(const KernelWorkload& workload) const {
  if (workload.name.empty()) {
    throw std::invalid_argument("GPU workload must have a name");
  }
  PerformanceEstimate result;
  result.name = workload.name;
  result.globalBytes = workload.logicalElements * workload.bytesPerElement *
                       (1 + workload.neighborReads);
  result.sharedBytes = estimateTileBytes(workload);
  result.coalescedTransactions = estimateCoalescedTransactions(
      workload.logicalElements, workload.bytesPerElement, device_.warpSize, workload.layout);
  result.arithmeticIntensity = safeRatio(
      static_cast<double>(workload.floatingPointOperations), static_cast<double>(result.globalBytes));
  result.theoreticalOccupancy = occupancy(workload);
  result.legalLaunch = result.theoreticalOccupancy > 0.0;

  const double bandwidthBytesPerSecond = device_.memoryBandwidthGBPerSecond * kBytesPerGiB;
  result.predictedMemoryMilliseconds = safeRatio(static_cast<double>(result.globalBytes), bandwidthBytesPerSecond) * 1000.0;
  const double computeOperationsPerSecond = device_.fp32ThroughputTFLOPS * 1.0e12;
  result.predictedComputeMilliseconds = safeRatio(
      static_cast<double>(workload.floatingPointOperations), computeOperationsPerSecond) * 1000.0;
  if (result.theoreticalOccupancy > 0.0) {
    result.predictedComputeMilliseconds /= result.theoreticalOccupancy;
  }
  result.bottleneck = classify(result);
  return result;
}

PipelineEstimate GpuPerformanceModel::estimateStableFluidPipeline(std::size_t cellCount) const {
  if (cellCount == 0) {
    throw std::invalid_argument("stable-fluid pipeline must contain at least one cell");
  }
  const std::size_t stencilTile = 18;
  const std::size_t pressureIterations = 20;
  std::vector<KernelWorkload> workloads;
  workloads.push_back({"splat", cellCount, sizeof(float) * 4, 0, 24, 0, 0, 256, 28,
                       MemoryLayout::StructureOfArrays, false});
  workloads.push_back({"advection", cellCount, sizeof(float) * 2, 4, 42, stencilTile, stencilTile, 256, 38,
                       MemoryLayout::StructureOfArrays, true});
  workloads.push_back({"divergence", cellCount, sizeof(float) * 2, 4, 18, stencilTile, stencilTile, 256, 30,
                       MemoryLayout::StructureOfArrays, true});
  workloads.push_back({"pressure-jacobi", cellCount, sizeof(float), 4, 14, stencilTile, stencilTile, 256, 26,
                       MemoryLayout::StructureOfArrays, true});
  workloads.push_back({"projection", cellCount, sizeof(float) * 2, 4, 22, stencilTile, stencilTile, 256, 34,
                       MemoryLayout::StructureOfArrays, true});
  workloads.push_back({"render", cellCount, sizeof(float) * 4, 0, 16, 0, 0, 256, 24,
                       MemoryLayout::StructureOfArrays, false});

  PipelineEstimate pipeline;
  pipeline.kernels.reserve(workloads.size() + 1);
  for (const KernelWorkload& workload : workloads) {
    PerformanceEstimate estimate = this->estimate(workload);
    if (workload.name == "pressure-jacobi") {
      estimate.globalBytes *= pressureIterations;
      estimate.coalescedTransactions *= pressureIterations;
      estimate.predictedMemoryMilliseconds *= static_cast<double>(pressureIterations);
      estimate.predictedComputeMilliseconds *= static_cast<double>(pressureIterations);
      estimate.bottleneck = classify(estimate);
    }
    pipeline.totalGlobalBytes += estimate.globalBytes;
    pipeline.totalPredictedMilliseconds += std::max(estimate.predictedMemoryMilliseconds,
                                                    estimate.predictedComputeMilliseconds);
    pipeline.kernels.push_back(std::move(estimate));
  }
  if (!pipeline.kernels.empty()) {
    pipeline.averageOccupancy = std::accumulate(
        pipeline.kernels.begin(), pipeline.kernels.end(), 0.0,
        [](double total, const PerformanceEstimate& estimate) {
          return total + estimate.theoreticalOccupancy;
        }) / static_cast<double>(pipeline.kernels.size());
  }

  std::array<int, 5> bottleneckCounts{};
  for (const auto& estimate : pipeline.kernels) {
    ++bottleneckCounts[static_cast<std::size_t>(estimate.bottleneck)];
  }
  pipeline.dominantBottleneck = static_cast<Bottleneck>(std::distance(
      bottleneckCounts.begin(), std::max_element(bottleneckCounts.begin(), bottleneckCounts.end())));
  return pipeline;
}

Bottleneck GpuPerformanceModel::classify(const PerformanceEstimate& result) const noexcept {
  if (!result.legalLaunch || result.theoreticalOccupancy < 0.25) {
    return Bottleneck::Occupancy;
  }
  if (result.predictedMemoryMilliseconds > result.predictedComputeMilliseconds * 1.25) {
    return Bottleneck::MemoryBandwidth;
  }
  if (result.predictedComputeMilliseconds > result.predictedMemoryMilliseconds * 1.25) {
    return Bottleneck::ArithmeticThroughput;
  }
  if (result.globalBytes < 1024 && result.floatingPointOperations == 0) {
    return Bottleneck::LaunchLatency;
  }
  return Bottleneck::None;
}

const char* GpuPerformanceModel::bottleneckName(Bottleneck bottleneck) noexcept {
  switch (bottleneck) {
    case Bottleneck::None:
      return "none";
    case Bottleneck::MemoryBandwidth:
      return "memory-bandwidth";
    case Bottleneck::ArithmeticThroughput:
      return "arithmetic-throughput";
    case Bottleneck::Occupancy:
      return "occupancy";
    case Bottleneck::LaunchLatency:
      return "launch-latency";
  }
  return "unknown";
}

const char* GpuPerformanceModel::layoutName(MemoryLayout layout) noexcept {
  return layout == MemoryLayout::StructureOfArrays ? "soa" : "aos";
}

std::string GpuPerformanceModel::toJson(const PipelineEstimate& pipeline) const {
  std::ostringstream value;
  value << std::fixed << std::setprecision(6);
  value << "{\"device\": {\"warpSize\": " << device_.warpSize
        << ", \"maxThreadsPerBlock\": " << device_.maxThreadsPerBlock
        << ", \"memoryBandwidthGBPerSecond\": " << device_.memoryBandwidthGBPerSecond
        << ", \"fp32ThroughputTFLOPS\": " << device_.fp32ThroughputTFLOPS << "},\n"
        << "\"totalGlobalBytes\": " << pipeline.totalGlobalBytes
        << ", \"totalPredictedMilliseconds\": " << pipeline.totalPredictedMilliseconds
        << ", \"averageOccupancy\": " << pipeline.averageOccupancy
        << ", \"dominantBottleneck\": \"" << bottleneckName(pipeline.dominantBottleneck) << "\",\n"
        << "\"kernels\": [";
  for (std::size_t index = 0; index < pipeline.kernels.size(); ++index) {
    const auto& kernel = pipeline.kernels[index];
    value << "{\"name\": \"" << kernel.name << "\", \"globalBytes\": " << kernel.globalBytes
          << ", \"sharedBytes\": " << kernel.sharedBytes
          << ", \"transactions\": " << kernel.coalescedTransactions
          << ", \"arithmeticIntensity\": " << kernel.arithmeticIntensity
          << ", \"occupancy\": " << kernel.theoreticalOccupancy
          << ", \"bottleneck\": \"" << bottleneckName(kernel.bottleneck)
          << "\", \"legalLaunch\": " << (kernel.legalLaunch ? "true" : "false") << "}";
    if (index + 1 != pipeline.kernels.size()) {
      value << ", ";
    }
  }
  value << "]}";
  return value.str();
}

}  // namespace gpu_fluids
