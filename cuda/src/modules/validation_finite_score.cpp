#include "gpu_fluids/validation_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double validation_finite_score(double first, double second, double third) noexcept {
  return std::isfinite(first) ? 1.0 : 0.0;
}

}  // namespace gpu_fluids

