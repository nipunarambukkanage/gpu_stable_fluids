#include "gpu_fluids/scenario_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double scenario_periodic_phase(double first, double second, double third) noexcept {
  return second > 0.0 ? std::fmod(std::fabs(first), second) / second : 0.0;
}

}  // namespace gpu_fluids

