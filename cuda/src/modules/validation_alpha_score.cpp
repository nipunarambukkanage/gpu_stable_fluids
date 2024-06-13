#include "gpu_fluids/validation_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double validation_alpha_score(double first, double second, double third) noexcept {
  return first >= 255.0 ? 1.0 : 0.0;
}

}  // namespace gpu_fluids

