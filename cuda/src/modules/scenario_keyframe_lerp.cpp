#include "gpu_fluids/scenario_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double scenario_keyframe_lerp(double first, double second, double third) noexcept {
  return first + (second - first) * third;
}

}  // namespace gpu_fluids

