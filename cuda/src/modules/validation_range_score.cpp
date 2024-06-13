#include "gpu_fluids/validation_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double validation_range_score(double first, double second, double third) noexcept {
  return first >= second && first <= third ? 1.0 : 0.0;
}

}  // namespace gpu_fluids

