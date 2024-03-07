#include "gpu_fluids/sampling_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double sampling_cubic_weight(double first, double second, double third) noexcept {
  return first * first * first;
}

}  // namespace gpu_fluids

