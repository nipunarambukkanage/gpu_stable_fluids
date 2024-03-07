#include "gpu_fluids/sampling_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double sampling_bilinear_value(double first, double second, double third) noexcept {
  return first * (1.0 - third) + second * third;
}

}  // namespace gpu_fluids

