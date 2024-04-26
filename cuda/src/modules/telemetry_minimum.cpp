#include "gpu_fluids/telemetry_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double telemetry_minimum(double first, double second, double third) noexcept {
  return std::min(first, std::min(second, third));
}

}  // namespace gpu_fluids

