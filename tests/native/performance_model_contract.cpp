#include "gpu_fluids/performance_model.hpp"

#include <iostream>
#include <string>

namespace {

int fail(const char* message) {
  std::cerr << "native_performance_model_contract: " << message << '\n';
  return 1;
}

}  // namespace

int main() {
  gpu_fluids::PerformanceDeviceModel device;
  device.memoryBandwidthGBPerSecond = 600.0;
  device.fp32ThroughputTFLOPS = 12.0;
  gpu_fluids::GpuPerformanceModel model(device);

  gpu_fluids::KernelWorkload soa;
  soa.name = "coalesced-stencil";
  soa.logicalElements = 4096;
  soa.bytesPerElement = sizeof(float);
  soa.neighborReads = 4;
  soa.floatingPointOperations = 32;
  soa.tileWidth = 18;
  soa.tileHeight = 18;
  soa.blockThreads = 256;
  soa.registersPerThread = 32;
  soa.usesSharedMemory = true;
  const auto soaEstimate = model.estimate(soa);
  if (!soaEstimate.legalLaunch || soaEstimate.sharedBytes == 0 || soaEstimate.theoreticalOccupancy <= 0.0 ||
      soaEstimate.coalescedTransactions == 0) {
    return fail("valid shared-memory workload was rejected or not measured");
  }

  auto aos = soa;
  aos.name = "strided-aos";
  aos.layout = gpu_fluids::MemoryLayout::ArrayOfStructures;
  const auto aosEstimate = model.estimate(aos);
  if (aosEstimate.coalescedTransactions <= soaEstimate.coalescedTransactions) {
    return fail("AoS transaction penalty is not represented");
  }
  if (model.recommendedBlockThreads(soa) <= 0) {
    return fail("block-size recommendation is empty");
  }
  if (gpu_fluids::GpuPerformanceModel::estimateTileBytes(soa) != 18U * 18U * sizeof(float)) {
    return fail("shared tile byte estimate is incorrect");
  }

  const auto pipeline = model.estimateStableFluidPipeline(512U * 512U);
  if (pipeline.kernels.size() != 6 || pipeline.totalGlobalBytes == 0 ||
      pipeline.totalPredictedMilliseconds <= 0.0 || pipeline.averageOccupancy <= 0.0) {
    return fail("stable-fluid pipeline estimate is incomplete");
  }
  const std::string json = model.toJson(pipeline);
  if (json.find("pressure-jacobi") == std::string::npos ||
      json.find("dominantBottleneck") == std::string::npos) {
    return fail("pipeline JSON is missing stage-level performance data");
  }

  gpu_fluids::PerformanceDeviceModel constrained = device;
  constrained.maxThreadsPerBlock = 64;
  gpu_fluids::GpuPerformanceModel constrainedModel(constrained);
  if (constrainedModel.estimate(soa).legalLaunch) {
    return fail("illegal block shape was accepted by the model");
  }
  std::cout << "native_performance_model_contract: passed\n";
  return 0;
}
