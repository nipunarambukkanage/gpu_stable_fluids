#include "gpu_fluids/fluid_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double fluid_speed_squared(double first, double second, double third) noexcept {
  return first * first + second * second;
}

}  // namespace gpu_fluids

