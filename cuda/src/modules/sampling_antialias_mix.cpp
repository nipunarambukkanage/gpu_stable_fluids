#include "gpu_fluids/sampling_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double sampling_antialias_mix(double first, double second, double third) noexcept {
  return first * third + second * (1.0 - third);
}

}  // namespace gpu_fluids

