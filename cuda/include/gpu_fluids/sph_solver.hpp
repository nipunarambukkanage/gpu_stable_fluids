#pragma once

#include "gpu_fluids/config.hpp"

#include <cuda_runtime_api.h>

#include <cstdint>
#include <vector>

namespace gpu_fluids {

class SphSolver final {
 public:
  explicit SphSolver(SphParams params = {}, int deviceIndex = 0);
  ~SphSolver();

  SphSolver(const SphSolver&) = delete;
  SphSolver& operator=(const SphSolver&) = delete;

  void reset();
  void step(const SphParams& params);
  void downloadFrame(std::vector<std::uint8_t>& rgba);

  [[nodiscard]] const FrameStats& lastFrameStats() const noexcept { return frameStats_; }
  [[nodiscard]] const DeviceMetrics& deviceMetrics() const noexcept { return metrics_; }
  [[nodiscard]] static constexpr int width() noexcept { return kGridWidth; }
  [[nodiscard]] static constexpr int height() noexcept { return kGridHeight; }

 private:
  void allocatePersistentResources();
  void releasePersistentResources() noexcept;
  void initializeParticles();
  void refreshMetrics();
  void launchClearGrid();
  void launchBuildGrid();
  void launchDensityPressure();
  void launchForces();
  void launchIntegrate();
  void launchRender();

  DeviceMetrics metrics_{};
  FrameStats frameStats_{};
  SphParams params_{};
  int deviceIndex_ = 0;

  cudaStream_t computeStream_ = nullptr;
  cudaStream_t copyStream_ = nullptr;
  cudaEvent_t frameStart_ = nullptr;
  cudaEvent_t frameEnd_ = nullptr;
  cudaEvent_t frameReady_ = nullptr;

  float* positionX_[2]{};
  float* positionY_[2]{};
  float* velocityX_[2]{};
  float* velocityY_[2]{};
  float* density_ = nullptr;
  float* pressure_ = nullptr;
  float* forceX_ = nullptr;
  float* forceY_ = nullptr;
  unsigned int* cellCounts_ = nullptr;
  unsigned int* cellParticles_ = nullptr;
  unsigned int* neighborOverflow_ = nullptr;
  unsigned int* pinnedNeighborOverflow_ = nullptr;
  std::uint8_t* deviceFrame_ = nullptr;
  std::uint8_t* pinnedHostFrame_ = nullptr;

  int particleRead_ = 0;
  std::uint64_t frameIndex_ = 0;
  std::size_t persistentDeviceBytes_ = 0;
};

}  // namespace gpu_fluids
