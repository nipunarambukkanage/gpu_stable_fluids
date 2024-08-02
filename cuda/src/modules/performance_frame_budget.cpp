#include "gpu_fluids/performance_extensions.hpp"
#include <algorithm>
#include <cmath>
namespace gpu_fluids {
double performance_frame_budget(double first, double second, double third) noexcept {
  return first > 0.0 ? 1000.0 / first : 0.0;
}
}  // namespace gpu_fluids

