#pragma once

#include "gpu_fluids/config.hpp"
#include "gpu_fluids/gpu_diagnostics.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace gpu_fluids {

struct LaunchPolicy {
  int stencilBlockThreads = kBlockSize * kBlockSize;
  int particleBlockThreads = kParticleBlockSize;
  int pressureIterations = kPressureIterations;
  std::size_t sharedTileBytes = 0;
  bool useSharedMemoryTiling = true;
  bool reducedPressureBudget = false;
  std::vector<std::string> rationale;
};

class LaunchPolicySelector final {
 public:
  [[nodiscard]] static LaunchPolicy select(const GpuDiagnosticsSnapshot& device,
                                           int requestedPressureIterations,
                                           bool particleMode);
  [[nodiscard]] static std::string toJson(const LaunchPolicy& policy);

 private:
  [[nodiscard]] static int warpAlignedDown(int value, int warpSize) noexcept;
  [[nodiscard]] static std::size_t stableFluidTileBytes() noexcept;
};

}  // namespace gpu_fluids
