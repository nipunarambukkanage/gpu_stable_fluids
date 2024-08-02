#include "gpu_fluids/serialization_extensions.hpp"
#include <algorithm>
#include <cmath>
namespace gpu_fluids {
double serialization_checksum(double first, double second, double third) noexcept {
  return std::fmod(std::fabs(first + second * 257.0 + third * 65537.0), 4294967291.0);
}
}  // namespace gpu_fluids

