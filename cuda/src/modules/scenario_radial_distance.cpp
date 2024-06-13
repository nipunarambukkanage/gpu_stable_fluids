#include "gpu_fluids/scenario_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double scenario_radial_distance(double first, double second, double third) noexcept {
  return std::hypot(first, second);
}

}  // namespace gpu_fluids

