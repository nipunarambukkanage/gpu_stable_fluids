#include "gpu_fluids/color_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double color_linear_to_srgb(double first, double second, double third) noexcept {
  return first <= 0.0031308 ? first * 12.92 : 1.055 * std::pow(first, 1.0 / 2.4) - 0.055;
}

}  // namespace gpu_fluids

