#include "gpu_fluids/telemetry_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double telemetry_sanitize_millis(double first, double second, double third) noexcept {
  return std::isfinite(first) && first >= 0.0 ? first : 0.0;
}

}  // namespace gpu_fluids

