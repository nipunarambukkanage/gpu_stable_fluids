#include "gpu_fluids/sampling_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double sampling_nearest_coordinate(double first, double second, double third) noexcept {
  return std::round(first);
}

}  // namespace gpu_fluids

