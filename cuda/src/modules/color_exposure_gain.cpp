#include "gpu_fluids/color_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double color_exposure_gain(double first, double second, double third) noexcept {
  return std::pow(2.0, first);
}

}  // namespace gpu_fluids

