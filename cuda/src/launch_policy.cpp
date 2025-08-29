#include "gpu_fluids/launch_policy.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace gpu_fluids {
namespace {

std::string escape(const std::string& value) {
  std::string result;
  result.reserve(value.size() + 4);
  for (const char character : value) {
    if (character == '\\' || character == '"') {
      result.push_back('\\');
    }
    result.push_back(character);
  }
  return result;
}

}  // namespace

int LaunchPolicySelector::warpAlignedDown(int value, int warpSize) noexcept {
  if (value <= 0 || warpSize <= 0) {
    return 0;
  }
  return value - value % warpSize;
}

std::size_t LaunchPolicySelector::stableFluidTileBytes() noexcept {
  return static_cast<std::size_t>(kTileExtent) * static_cast<std::size_t>(kTileExtent) * sizeof(float) * 2U;
}

LaunchPolicy LaunchPolicySelector::select(const GpuDiagnosticsSnapshot& device,
                                          int requestedPressureIterations,
                                          bool particleMode) {
  if (device.warpSize <= 0 || device.maxThreadsPerBlock <= 0 ||
      device.sharedMemoryPerBlock <= 0 || device.freeMemoryBytes == 0) {
    throw std::invalid_argument("device snapshot is incomplete for launch-policy selection");
  }

  LaunchPolicy policy;
  policy.pressureIterations = std::clamp(requestedPressureIterations, 1, 256);
  policy.sharedTileBytes = stableFluidTileBytes();
  const int preferredStencilThreads = std::min(kBlockSize * kBlockSize, device.maxThreadsPerBlock);
  policy.stencilBlockThreads = warpAlignedDown(preferredStencilThreads, device.warpSize);
  if (policy.stencilBlockThreads == 0) {
    policy.stencilBlockThreads = std::min(device.maxThreadsPerBlock, device.warpSize);
  }

  const int preferredParticleThreads = std::min(kParticleBlockSize, device.maxThreadsPerBlock);
  policy.particleBlockThreads = warpAlignedDown(preferredParticleThreads, device.warpSize);
  if (policy.particleBlockThreads == 0) {
    policy.particleBlockThreads = std::min(device.maxThreadsPerBlock, device.warpSize);
  }
  if (particleMode) {
    policy.rationale.emplace_back("particle mode selected a warp-aligned neighbor-search block");
  } else {
    policy.rationale.emplace_back("stencil mode selected a warp-aligned 2D tile block");
  }

  if (policy.sharedTileBytes > static_cast<std::size_t>(device.sharedMemoryPerBlock)) {
    policy.useSharedMemoryTiling = false;
    policy.sharedTileBytes = 0;
    policy.rationale.emplace_back("shared tile exceeds per-block memory; global loads remain legal");
  } else {
    policy.rationale.emplace_back("shared-memory halo tile fits within the device block budget");
  }

  constexpr std::size_t lowMemoryThreshold = 256U * 1024U * 1024U;
  if (device.freeMemoryBytes < lowMemoryThreshold && policy.pressureIterations > 12) {
    policy.pressureIterations = 12;
    policy.reducedPressureBudget = true;
    policy.rationale.emplace_back("low free VRAM reduced the pressure budget to protect residency");
  }
  if (device.computeCapabilityMajor < 7) {
    policy.pressureIterations = std::min(policy.pressureIterations, 16);
    policy.rationale.emplace_back("older SM capability capped the pressure budget conservatively");
  }
  return policy;
}

std::string LaunchPolicySelector::toJson(const LaunchPolicy& policy) {
  std::ostringstream value;
  value << "{\"stencilBlockThreads\": " << policy.stencilBlockThreads
        << ", \"particleBlockThreads\": " << policy.particleBlockThreads
        << ", \"pressureIterations\": " << policy.pressureIterations
        << ", \"sharedTileBytes\": " << policy.sharedTileBytes
        << ", \"useSharedMemoryTiling\": " << (policy.useSharedMemoryTiling ? "true" : "false")
        << ", \"reducedPressureBudget\": " << (policy.reducedPressureBudget ? "true" : "false")
        << ", \"rationale\": [";
  for (std::size_t index = 0; index < policy.rationale.size(); ++index) {
    value << "\"" << escape(policy.rationale[index]) << "\"";
    if (index + 1 != policy.rationale.size()) {
      value << ", ";
    }
  }
  value << "]}";
  return value.str();
}

}  // namespace gpu_fluids
