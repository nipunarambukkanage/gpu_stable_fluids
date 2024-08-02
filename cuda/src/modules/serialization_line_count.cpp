#include "gpu_fluids/serialization_extensions.hpp"
#include <algorithm>
#include <cmath>
namespace gpu_fluids {
double serialization_line_count(double first, double second, double third) noexcept {
  return std::max(0.0, first) + (second > 0.0 ? 1.0 : 0.0);
}
}  // namespace gpu_fluids

