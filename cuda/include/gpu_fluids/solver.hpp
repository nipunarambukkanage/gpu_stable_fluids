#pragma once

#include "gpu_fluids/config.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace gpu_fluids {

class StableFluidSolver final {
 public:
  explicit StableFluidSolver(SolverConfig config = {});
  ~StableFluidSolver();

  StableFluidSolver(const StableFluidSolver&) = delete;
  StableFluidSolver& operator=(const StableFluidSolver&) = delete;

  void reset();
  void step(const SimulationParams& params);
  void downloadFrame(std::vector<std::uint8_t>& rgba);

  [[nodiscard]] const FrameStats& lastFrameStats() const noexcept { return frameStats_; }
  [[nodiscard]] const DeviceMetrics& deviceMetrics() const noexcept { return metrics_; }
  [[nodiscard]] static constexpr int width() noexcept { return kGridWidth; }
  [[nodiscard]] static constexpr int height() noexcept { return kGridHeight; }

 private:
  void allocatePersistentResources();
  void releasePersistentResources() noexcept;
  void initializeParticles();
  void launchSplat();
  void launchAdvection();
  void launchVorticity();
  void launchConfinement();
  void launchDivergence();
  void launchPressure();
  void launchGradient();
  void launchParticles();
  void launchRender();
  void refreshMetrics();

  SolverConfig config_;
  DeviceMetrics metrics_{};
  FrameStats frameStats_{};
  SimulationParams params_{};

  cudaStream_t computeStream_ = nullptr;
  cudaStream_t copyStream_ = nullptr;
  cudaEvent_t frameStart_ = nullptr;
  cudaEvent_t frameEnd_ = nullptr;
  cudaEvent_t frameReady_ = nullptr;

  float2* velocity_[2]{};
  float4* density_[2]{};
  float* pressure_[2]{};
  float* divergence_ = nullptr;
  float* vorticity_ = nullptr;
  float4* particles_[2]{};
  std::uint8_t* deviceFrame_ = nullptr;
  std::uint8_t* pinnedHostFrame_ = nullptr;

  int velocityRead_ = 0;
  int densityRead_ = 0;
  int pressureRead_ = 0;
  int particleRead_ = 0;
  std::uint64_t frameIndex_ = 0;
  std::size_t persistentDeviceBytes_ = 0;
};

}  // namespace gpu_fluids
