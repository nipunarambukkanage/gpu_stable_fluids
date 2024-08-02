#include "gpu_fluids/serialization_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double serialization_precision_scale(double first, double second, double third) noexcept {
  return second > 0.0 ? std::round(first * std::pow(10.0, second)) / std::pow(10.0, second) : first;
}

}  // namespace gpu_fluids

