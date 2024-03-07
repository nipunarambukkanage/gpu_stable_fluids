#include "gpu_fluids/sampling_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double sampling_smoothstep(double first, double second, double third) noexcept {
  return std::max(0.0, std::min(1.0, first)) * std::max(0.0, std::min(1.0, first)) * (3.0 - 2.0 * std::max(0.0, std::min(1.0, first)));
}

}  // namespace gpu_fluids

