#include "gpu_fluids/sampling_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double sampling_clamp_coordinate(double first, double second, double third) noexcept {
  return std::max(0.0, std::min(first, second - 1.0));
}

}  // namespace gpu_fluids

