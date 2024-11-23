#pragma once

namespace gpu_fluids {

// Normalized controls for the deterministic input path used by the reference
// runtime. Keeping path generation independent of the solver lets test runs
// reproduce exactly the same injected velocity at any grid resolution.
struct EllipticalStrokeConfig final {
  float centerX = 0.5F;
  float centerY = 0.5F;
  float radiusX = 0.28F;
  float radiusY = 0.22F;
  float angularFrequencyX = 1.7F;
  float angularFrequencyY = 2.1F;
  float phaseX = 0.0F;
  float phaseY = 0.0F;
};

struct StrokeSegment final {
  float startX = 0.0F;
  float startY = 0.0F;
  float endX = 0.0F;
  float endY = 0.0F;
  float velocityX = 0.0F;
  float velocityY = 0.0F;
};

class EllipticalStrokeTrajectory final {
 public:
  explicit EllipticalStrokeTrajectory(EllipticalStrokeConfig config = {});

  [[nodiscard]] const EllipticalStrokeConfig& config() const noexcept { return config_; }
  [[nodiscard]] StrokeSegment sample(float time,
                                      float deltaTime,
                                      float gridWidth,
                                      float gridHeight) const;

 private:
  [[nodiscard]] static bool isValid(const EllipticalStrokeConfig& config) noexcept;

  EllipticalStrokeConfig config_{};
};

}  // namespace gpu_fluids
