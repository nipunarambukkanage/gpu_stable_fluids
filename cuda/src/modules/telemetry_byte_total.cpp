#include "gpu_fluids/telemetry_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double telemetry_byte_total(double first, double second, double third) noexcept {
  return first + second + third;
}

}  // namespace gpu_fluids

