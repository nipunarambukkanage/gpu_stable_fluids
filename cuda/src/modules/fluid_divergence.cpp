#include "gpu_fluids/fluid_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double fluid_divergence(double first, double second, double third) noexcept {
  return 0.5 * ((first - second) + (third - first));
}

}  // namespace gpu_fluids

