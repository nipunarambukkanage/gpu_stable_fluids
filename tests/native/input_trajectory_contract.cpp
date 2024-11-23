#include "gpu_fluids/input_trajectory.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

constexpr float kPi = 3.14159265358979323846F;

int fail(const char* message) {
  std::cerr << "native_input_trajectory_contract: " << message << '\n';
  return 1;
}

template <typename Callable>
bool throwsInvalidArgument(Callable&& callable) {
  try {
    callable();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

bool approximatelyEqual(float first, float second) {
  return std::abs(first - second) <= 1.0e-4F;
}

}  // namespace

int main() {
  gpu_fluids::EllipticalStrokeConfig config;
  config.radiusX = 0.25F;
  config.radiusY = 0.5F;
  config.angularFrequencyX = 1.0F;
  config.angularFrequencyY = 2.0F;
  gpu_fluids::EllipticalStrokeTrajectory trajectory(config);
  const gpu_fluids::StrokeSegment segment = trajectory.sample(0.0F, kPi / 2.0F, 100.0F, 80.0F);
  if (!approximatelyEqual(segment.startX, 75.0F) || !approximatelyEqual(segment.startY, 40.0F) ||
      !approximatelyEqual(segment.endX, 50.0F) || !approximatelyEqual(segment.endY, 40.0F) ||
      !approximatelyEqual(segment.velocityX, -50.0F / (kPi / 2.0F)) ||
      !approximatelyEqual(segment.velocityY, 0.0F)) {
    return fail("elliptical trajectory did not produce the expected segment");
  }

  const gpu_fluids::StrokeSegment repeat = trajectory.sample(0.0F, kPi / 2.0F, 100.0F, 80.0F);
  if (!approximatelyEqual(repeat.startX, segment.startX) ||
      !approximatelyEqual(repeat.velocityY, segment.velocityY)) {
    return fail("trajectory samples are not deterministic");
  }

  gpu_fluids::EllipticalStrokeConfig invalidConfig;
  invalidConfig.radiusY = -0.1F;
  if (!throwsInvalidArgument([&] { static_cast<void>(gpu_fluids::EllipticalStrokeTrajectory(invalidConfig)); }) ||
      !throwsInvalidArgument([&] { trajectory.sample(0.0F, 0.0F, 100.0F, 80.0F); }) ||
      !throwsInvalidArgument([&] {
        trajectory.sample(std::numeric_limits<float>::infinity(), 0.1F, 100.0F, 80.0F);
      })) {
    return fail("invalid trajectory input was accepted");
  }

  std::cout << "native_input_trajectory_contract: passed\n";
  return 0;
}
