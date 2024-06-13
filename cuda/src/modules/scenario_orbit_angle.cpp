#include "gpu_fluids/scenario_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double scenario_orbit_angle(double first, double second, double third) noexcept {
  return first * second + third;
}

}  // namespace gpu_fluids

