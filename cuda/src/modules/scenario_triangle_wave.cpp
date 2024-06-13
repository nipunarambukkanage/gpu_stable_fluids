#include "gpu_fluids/scenario_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double scenario_triangle_wave(double first, double second, double third) noexcept {
  return second > 0.0 ? 1.0 - std::fabs(2.0 * (std::fmod(first, second) / second) - 1.0) : 0.0;
}

}  // namespace gpu_fluids

