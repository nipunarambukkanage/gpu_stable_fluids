#include "gpu_fluids/telemetry_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double telemetry_mean(double first, double second, double third) noexcept {
  return (first + second + third) / 3.0;
}

}  // namespace gpu_fluids

