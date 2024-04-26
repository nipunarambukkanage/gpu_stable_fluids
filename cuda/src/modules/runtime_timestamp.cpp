#include "gpu_fluids/runtime_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double runtime_timestamp(double first, double second, double third) noexcept {
  return first * second;
}

}  // namespace gpu_fluids

