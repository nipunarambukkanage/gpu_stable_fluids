#include "gpu_fluids/frame_metrics.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

int fail(const char* message) {
  std::cerr << "native_frame_metrics_contract: " << message << '\n';
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

}  // namespace

int main() {
  const std::vector<std::uint8_t> frame = {
      0, 0, 0, 255,       255, 0, 0, 255,     255, 255, 255, 255,
      0, 255, 255, 128,   0, 0, 0, 0,         0, 255, 0, 255,
  };
  const gpu_fluids::FrameMetrics metrics = gpu_fluids::measureRgbaFrame(frame, 3, 2);
  if (metrics.pixels != 6 || metrics.opaquePixels != 4 || metrics.activePixels != 4 ||
      std::abs(metrics.minimumLuminance) > 1.0e-6F ||
      std::abs(metrics.maximumLuminance - 1.0F) > 1.0e-6F ||
      std::abs(metrics.meanLuminance - (2.7152 / 6.0)) > 1.0e-6 ||
      metrics.luminanceVariance <= 0.0 || !metrics.activeBounds.hasActivePixels ||
      metrics.activeBounds.minimumX != 0 || metrics.activeBounds.minimumY != 0 ||
      metrics.activeBounds.maximumX != 2 || metrics.activeBounds.maximumY != 1) {
    return fail("frame metrics did not describe the RGBA frame correctly");
  }

  const gpu_fluids::FrameMetrics sparse = gpu_fluids::measureRgbaFrame(frame, 3, 2, 254U);
  if (sparse.activePixels != 4 || sparse.activeBounds.minimumX != 0 || sparse.activeBounds.maximumX != 2) {
    return fail("frame activity threshold was not respected");
  }

  if (!throwsInvalidArgument([&] { gpu_fluids::measureRgbaFrame(frame, 0, 2); }) ||
      !throwsInvalidArgument([&] { gpu_fluids::measureRgbaFrame({0, 0, 0}, 1, 1); })) {
    return fail("invalid RGBA frame input was accepted");
  }

  std::cout << "native_frame_metrics_contract: passed\n";
  return 0;
}
