#include "gpu_fluids/sampling_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double sampling_lerp(double first, double second, double third) noexcept {
  return first + (second - first) * third;
}

}  // namespace gpu_fluids

