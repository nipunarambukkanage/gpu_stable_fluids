#include "gpu_fluids/fluid_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double fluid_dissipation(double first, double second, double third) noexcept {
  return std::exp(-std::max(0.0, first) * std::max(0.0, second));
}

}  // namespace gpu_fluids

