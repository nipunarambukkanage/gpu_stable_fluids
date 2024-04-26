#include "gpu_fluids/telemetry_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double telemetry_percentile_rank(double first, double second, double third) noexcept {
  return std::max(first, std::max(second, third));
}

}  // namespace gpu_fluids

