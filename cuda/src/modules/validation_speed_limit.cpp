#include "gpu_fluids/validation_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double validation_speed_limit(double first, double second, double third) noexcept {
  return std::max(0.0, third - std::fabs(first));
}

}  // namespace gpu_fluids

