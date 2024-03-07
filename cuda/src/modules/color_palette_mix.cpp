#include "gpu_fluids/color_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double color_palette_mix(double first, double second, double third) noexcept {
  return first * (1.0 - third) + second * third;
}

}  // namespace gpu_fluids

