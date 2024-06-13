#include "gpu_fluids/scenario_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double scenario_smooth_pulse(double first, double second, double third) noexcept {
  return std::max(0.0, std::min(1.0, (first - second) / (third - second)));
}

}  // namespace gpu_fluids

