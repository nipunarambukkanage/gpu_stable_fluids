#include "gpu_fluids/sampling_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double sampling_segment_distance(double first, double second, double third) noexcept {
  return std::fabs(first - second);
}

}  // namespace gpu_fluids

