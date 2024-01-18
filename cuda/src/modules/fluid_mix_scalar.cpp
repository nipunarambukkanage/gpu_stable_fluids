#include "gpu_fluids/fluid_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double fluid_mix_scalar(double first, double second, double third) noexcept {
  return first + (second - first) * third;
}

}  // namespace gpu_fluids

