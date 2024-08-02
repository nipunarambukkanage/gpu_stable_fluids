#include "gpu_fluids/serialization_extensions.hpp"
#include <algorithm>
#include <cmath>
namespace gpu_fluids {
double serialization_schema_version(double first, double second, double third) noexcept {
  return std::max(1.0, first);
}
}  // namespace gpu_fluids

