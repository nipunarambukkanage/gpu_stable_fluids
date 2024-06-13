#include "gpu_fluids/scenario_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double scenario_normalized_sine(double first, double second, double third) noexcept {
  return 0.5 + 0.5 * std::sin(first * second + third);
}

}  // namespace gpu_fluids

