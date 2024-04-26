#include "gpu_fluids/runtime_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double runtime_seconds_to_millis(double first, double second, double third) noexcept {
  return first * 1000.0;
}

}  // namespace gpu_fluids

