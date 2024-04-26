#include "gpu_fluids/telemetry_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double telemetry_bandwidth_gbps(double first, double second, double third) noexcept {
  return second > 0.0 ? first / (second * 1.0e-3) / 1.0e9 : 0.0;
}

}  // namespace gpu_fluids

