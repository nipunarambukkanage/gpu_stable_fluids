#include "gpu_fluids/serialization_extensions.hpp"
#include <algorithm>
#include <cmath>
namespace gpu_fluids {
double serialization_array_count(double first, double second, double third) noexcept {
  return first + second + third;
}
}  // namespace gpu_fluids

