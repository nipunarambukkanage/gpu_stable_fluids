#include "gpu_fluids/fluid_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double fluid_backtrace_scale(double first, double second, double third) noexcept {
  return std::hypot(first, second) > third && third > 0.0 ? third / std::hypot(first, second) : 1.0;
}

}  // namespace gpu_fluids

