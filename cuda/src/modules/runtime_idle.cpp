#include "gpu_fluids/runtime_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double runtime_idle(double first, double second, double third) noexcept {
  return first > 0.0 && second <= 0.0 ? 1.0 : 0.0;
}

}  // namespace gpu_fluids

