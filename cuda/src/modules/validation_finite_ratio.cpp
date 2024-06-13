#include "gpu_fluids/validation_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double validation_finite_ratio(double first, double second, double third) noexcept {
  return second > 0.0 ? std::max(0.0, std::min(1.0, first / second)) : 1.0;
}

}  // namespace gpu_fluids

