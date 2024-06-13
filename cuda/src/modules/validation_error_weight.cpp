#include "gpu_fluids/validation_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double validation_error_weight(double first, double second, double third) noexcept {
  return 1.0 / (1.0 + first + 0.25 * second);
}

}  // namespace gpu_fluids

