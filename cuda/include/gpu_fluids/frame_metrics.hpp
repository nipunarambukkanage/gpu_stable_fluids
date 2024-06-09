#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gpu_fluids {

struct ActivePixelBounds final {
  bool hasActivePixels = false;
  int minimumX = 0;
  int minimumY = 0;
  int maximumX = 0;
  int maximumY = 0;
};

// Lightweight presentation diagnostics for a rendered RGBA8 frame. Luminance
// uses Rec. 709 weights so it can be compared across CPU and CUDA frame exports
// without interpreting a file format.
struct FrameMetrics final {
  std::size_t pixels = 0;
  std::size_t opaquePixels = 0;
  std::size_t activePixels = 0;
  float minimumLuminance = 0.0F;
  float maximumLuminance = 0.0F;
  double meanLuminance = 0.0;
  double luminanceVariance = 0.0;
  ActivePixelBounds activeBounds{};
};

// Measures an in-memory RGBA8 frame. A pixel is active when any RGB channel
// meets activityThreshold; alpha is reported separately and does not hide RGB
// data, which makes corrupt or premultiplied exports observable.
[[nodiscard]] FrameMetrics measureRgbaFrame(const std::vector<std::uint8_t>& rgba,
                                             int width,
                                             int height,
                                             std::uint8_t activityThreshold = 1U);

}  // namespace gpu_fluids
