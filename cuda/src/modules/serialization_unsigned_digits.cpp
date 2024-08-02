#include "gpu_fluids/serialization_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double serialization_unsigned_digits(double first, double second, double third) noexcept {
  return first >= 0.0 ? std::floor(std::log10(first + 1.0)) + 1.0 : 1.0;
}

}  // namespace gpu_fluids

