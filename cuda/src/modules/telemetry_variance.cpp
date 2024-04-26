#include "gpu_fluids/telemetry_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double telemetry_variance(double first, double second, double third) noexcept {
  return ((first - second) * (first - second) + (third - second) * (third - second)) / 2.0;
}

}  // namespace gpu_fluids

