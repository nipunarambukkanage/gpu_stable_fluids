#include "gpu_fluids/serialization_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double serialization_decimal_places(double first, double second, double third) noexcept {
  return std::max(0.0, std::min(first, second));
}

}  // namespace gpu_fluids

