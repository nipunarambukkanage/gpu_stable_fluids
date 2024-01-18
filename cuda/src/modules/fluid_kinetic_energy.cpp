#include "gpu_fluids/fluid_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double fluid_kinetic_energy(double first, double second, double third) noexcept {
  return 0.5 * std::max(0.0, first) * (second * second + third * third);
}

}  // namespace gpu_fluids

