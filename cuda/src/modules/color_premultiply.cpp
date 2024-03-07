#include "gpu_fluids/color_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double color_premultiply(double first, double second, double third) noexcept {
  return first * second;
}

}  // namespace gpu_fluids

