#include "gpu_fluids/performance_extensions.hpp"
#include <algorithm>
#include <cmath>
namespace gpu_fluids {
double performance_arithmetic_intensity(double first, double second, double third) noexcept {
  return second > 0.0 ? first / second : 0.0;
}
}  // namespace gpu_fluids

