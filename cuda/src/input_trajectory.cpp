#include "gpu_fluids/input_trajectory.hpp"

#include <cmath>
#include <stdexcept>

namespace gpu_fluids {
namespace {

bool isFinite(float value) noexcept {
  return std::isfinite(value);
}

}  // namespace

EllipticalStrokeTrajectory::EllipticalStrokeTrajectory(EllipticalStrokeConfig config)
    : config_(config) {
  if (!isValid(config_)) {
    throw std::invalid_argument("elliptical stroke configuration is invalid");
  }
}

bool EllipticalStrokeTrajectory::isValid(const EllipticalStrokeConfig& config) noexcept {
  return isFinite(config.centerX) && isFinite(config.centerY) &&
         isFinite(config.radiusX) && isFinite(config.radiusY) &&
         isFinite(config.angularFrequencyX) && isFinite(config.angularFrequencyY) &&
         isFinite(config.phaseX) && isFinite(config.phaseY) &&
         config.centerX >= 0.0F && config.centerX <= 1.0F &&
         config.centerY >= 0.0F && config.centerY <= 1.0F &&
         config.radiusX >= 0.0F && config.radiusX <= 1.0F &&
         config.radiusY >= 0.0F && config.radiusY <= 1.0F;
}

StrokeSegment EllipticalStrokeTrajectory::sample(float time,
                                                 float deltaTime,
                                                 float gridWidth,
                                                 float gridHeight) const {
  if (!isFinite(time) || !isFinite(deltaTime) || deltaTime <= 0.0F ||
      !isFinite(gridWidth) || !isFinite(gridHeight) || gridWidth <= 0.0F || gridHeight <= 0.0F) {
    throw std::invalid_argument("trajectory sample requires finite positive step and dimensions");
  }

  const float nextTime = time + deltaTime;
  const auto position = [this, gridWidth, gridHeight](float sampleTime) {
    const float x = (config_.centerX + config_.radiusX *
                     std::cos(sampleTime * config_.angularFrequencyX + config_.phaseX)) * gridWidth;
    const float y = (config_.centerY + config_.radiusY *
                     std::sin(sampleTime * config_.angularFrequencyY + config_.phaseY)) * gridHeight;
    return StrokeSegment{x, y};
  };

  StrokeSegment segment = position(time);
  const StrokeSegment end = position(nextTime);
  segment.endX = end.startX;
  segment.endY = end.startY;
  segment.velocityX = (segment.endX - segment.startX) / deltaTime;
  segment.velocityY = (segment.endY - segment.startY) / deltaTime;
  return segment;
}

}  // namespace gpu_fluids
