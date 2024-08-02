#include "gpu_fluids/serialization_extensions.hpp"
#include <algorithm>
#include <cmath>
namespace gpu_fluids {
double serialization_hash_mix(double first, double second, double third) noexcept {
  return std::fmod(std::fabs(first * 31.0 + second * 17.0 + third), 4294967296.0);
}
}  // namespace gpu_fluids

