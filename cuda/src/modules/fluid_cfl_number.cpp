#include "gpu_fluids/fluid_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double fluid_cfl_number(double first, double second, double third) noexcept {
  return third > 0.0 ? std::fabs(first) * std::max(0.0, second) / third : 0.0;
}

}  // namespace gpu_fluids

