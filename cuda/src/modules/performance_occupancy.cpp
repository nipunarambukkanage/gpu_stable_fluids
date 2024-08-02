#include "gpu_fluids/performance_extensions.hpp"
#include <algorithm>
#include <cmath>
namespace gpu_fluids {
double performance_occupancy(double first, double second, double third) noexcept {
  return second > 0.0 ? std::max(0.0, std::min(1.0, first / second)) : 0.0;
}
}  // namespace gpu_fluids

