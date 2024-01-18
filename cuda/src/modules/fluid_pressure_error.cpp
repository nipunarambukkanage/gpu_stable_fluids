#include "gpu_fluids/fluid_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double fluid_pressure_error(double first, double second, double third) noexcept {
  return std::fabs(first - second);
}

}  // namespace gpu_fluids

