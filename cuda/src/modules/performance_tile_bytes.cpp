#include "gpu_fluids/performance_extensions.hpp"
#include <algorithm>
#include <cmath>
namespace gpu_fluids {
double performance_tile_bytes(double first, double second, double third) noexcept {
  return std::max(0.0, first * second * third);
}
}  // namespace gpu_fluids

