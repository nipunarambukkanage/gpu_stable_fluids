#include "gpu_fluids/telemetry_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double telemetry_samples_per_second(double first, double second, double third) noexcept {
  return second > 0.0 ? first / (second * 1.0e-3) : 0.0;
}

}  // namespace gpu_fluids

