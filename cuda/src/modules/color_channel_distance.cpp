#include "gpu_fluids/color_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double color_channel_distance(double first, double second, double third) noexcept {
  return std::hypot(first - second, third);
}

}  // namespace gpu_fluids

