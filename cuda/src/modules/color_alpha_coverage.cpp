#include "gpu_fluids/color_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double color_alpha_coverage(double first, double second, double third) noexcept {
  return std::max(0.0, std::min(first, 1.0));
}

}  // namespace gpu_fluids

