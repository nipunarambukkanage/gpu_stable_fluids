#include "gpu_fluids/runtime_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double runtime_pressure_limit(double first, double second, double third) noexcept {
  return std::max(1.0, std::min(first, 256.0));
}

}  // namespace gpu_fluids

