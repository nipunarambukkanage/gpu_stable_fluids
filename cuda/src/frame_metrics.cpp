#include "gpu_fluids/frame_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gpu_fluids {
namespace {

constexpr float kRedLuminance = 0.2126F;
constexpr float kGreenLuminance = 0.7152F;
constexpr float kBlueLuminance = 0.0722F;
constexpr float kByteScale = 1.0F / 255.0F;

float luminance(std::uint8_t red, std::uint8_t green, std::uint8_t blue) noexcept {
  return (kRedLuminance * static_cast<float>(red) +
          kGreenLuminance * static_cast<float>(green) +
          kBlueLuminance * static_cast<float>(blue)) * kByteScale;
}

}  // namespace

FrameMetrics measureRgbaFrame(const std::vector<std::uint8_t>& rgba,
                              int width,
                              int height,
                              std::uint8_t activityThreshold) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("frame metrics require positive dimensions");
  }
  const std::size_t pixelCount = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
  if (rgba.size() != pixelCount * 4U) {
    throw std::invalid_argument("frame metrics require exactly four bytes per pixel");
  }

  FrameMetrics metrics;
  metrics.pixels = pixelCount;
  double mean = 0.0;
  double sumOfSquaredDifferences = 0.0;

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const std::size_t offset = (static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                  static_cast<std::size_t>(x)) *
                                 4U;
      const std::uint8_t red = rgba[offset];
      const std::uint8_t green = rgba[offset + 1U];
      const std::uint8_t blue = rgba[offset + 2U];
      const std::uint8_t alpha = rgba[offset + 3U];
      const float value = luminance(red, green, blue);
      const std::size_t ordinal = static_cast<std::size_t>(y) * static_cast<std::size_t>(width) +
                                  static_cast<std::size_t>(x) + 1U;

      metrics.minimumLuminance = ordinal == 1U ? value : std::min(metrics.minimumLuminance, value);
      metrics.maximumLuminance = ordinal == 1U ? value : std::max(metrics.maximumLuminance, value);
      const double delta = static_cast<double>(value) - mean;
      mean += delta / static_cast<double>(ordinal);
      sumOfSquaredDifferences += delta * (static_cast<double>(value) - mean);

      if (alpha == 255U) {
        ++metrics.opaquePixels;
      }
      if (std::max({red, green, blue}) >= activityThreshold) {
        ++metrics.activePixels;
        if (!metrics.activeBounds.hasActivePixels) {
          metrics.activeBounds = {true, x, y, x, y};
        } else {
          metrics.activeBounds.minimumX = std::min(metrics.activeBounds.minimumX, x);
          metrics.activeBounds.minimumY = std::min(metrics.activeBounds.minimumY, y);
          metrics.activeBounds.maximumX = std::max(metrics.activeBounds.maximumX, x);
          metrics.activeBounds.maximumY = std::max(metrics.activeBounds.maximumY, y);
        }
      }
    }
  }

  metrics.meanLuminance = mean;
  metrics.luminanceVariance = sumOfSquaredDifferences / static_cast<double>(pixelCount);
  return metrics;
}

}  // namespace gpu_fluids
