#include "gpu_fluids/color_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double color_luminance709(double first, double second, double third) noexcept {
  return 0.2126 * first + 0.7152 * second + 0.0722 * third;
}

}  // namespace gpu_fluids

