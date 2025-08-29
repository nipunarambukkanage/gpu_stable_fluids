#pragma once

#include "gpu_fluids/config.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpu_fluids {

class TelemetryCollector;

// A deterministic host reference implementation used for CI, numerical
// regression tests, and machines without an NVIDIA device.  It intentionally
// follows the same stage ordering as StableFluidSolver while keeping ownership
// and storage completely in standard C++ containers.
class CpuStableFluidSolver final {
 public:
  explicit CpuStableFluidSolver(int width = 128, int height = 128);

  void reset();
  void step(const SimulationParams& params);
  void step(const SimulationParams& params, TelemetryCollector* telemetry);
  void downloadFrame(std::vector<std::uint8_t>& rgba) const;

  void setPressureIterations(int iterations) noexcept;

  [[nodiscard]] int width() const noexcept { return width_; }
  [[nodiscard]] int height() const noexcept { return height_; }
  [[nodiscard]] const FrameStats& lastFrameStats() const noexcept { return frameStats_; }
  [[nodiscard]] float maxSpeed() const noexcept { return maxSpeed_; }
  [[nodiscard]] float dyeEnergy() const noexcept { return dyeEnergy_; }

 private:
  using Field = std::vector<float>;

  [[nodiscard]] std::size_t index(int x, int y) const noexcept;
  [[nodiscard]] float sample(const Field& field, float x, float y) const noexcept;
  [[nodiscard]] float sampleVelocityX(float x, float y) const noexcept;
  [[nodiscard]] float sampleVelocityY(float x, float y) const noexcept;
  [[nodiscard]] float segmentDistanceSquared(float px, float py, float ax, float ay,
                                              float bx, float by) const noexcept;

  void injectStroke(const SimulationParams& params);
  void advectVelocity(float deltaTime);
  void advectDye(float deltaTime);
  void computeDivergence();
  void solvePressure();
  void projectVelocity();
  void applyBoundary();
  void render(std::vector<std::uint8_t>& rgba) const;
  void updateStats(float elapsedMilliseconds);

  int width_ = 0;
  int height_ = 0;
  int pressureIterations_ = kPressureIterations;
  std::size_t cellCount_ = 0;

  Field velocityXRead_;
  Field velocityXWrite_;
  Field velocityYRead_;
  Field velocityYWrite_;
  Field densityRRead_;
  Field densityRWrite_;
  Field densityGRead_;
  Field densityGWrite_;
  Field densityBRead_;
  Field densityBWrite_;
  Field pressureRead_;
  Field pressureWrite_;
  Field divergence_;

  SimulationParams params_{};
  FrameStats frameStats_{};
  std::uint64_t frameIndex_ = 0;
  float maxSpeed_ = 0.0F;
  float dyeEnergy_ = 0.0F;
};

}  // namespace gpu_fluids
