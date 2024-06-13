#include "gpu_fluids/scenario_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double scenario_ease_in_out(double first, double second, double third) noexcept {
  return first * first * (3.0 - 2.0 * first);
}

}  // namespace gpu_fluids

