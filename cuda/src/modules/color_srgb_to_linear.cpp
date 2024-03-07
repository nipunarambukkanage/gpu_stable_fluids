#include "gpu_fluids/color_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double color_srgb_to_linear(double first, double second, double third) noexcept {
  return first <= 0.04045 ? first / 12.92 : std::pow((first + 0.055) / 1.055, 2.4);
}

}  // namespace gpu_fluids

