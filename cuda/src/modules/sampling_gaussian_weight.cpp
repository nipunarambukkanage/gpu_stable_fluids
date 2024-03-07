#include "gpu_fluids/sampling_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double sampling_gaussian_weight(double first, double second, double third) noexcept {
  return std::exp(-std::max(0.0, first) / std::max(1.0e-9, second * second));
}

}  // namespace gpu_fluids

