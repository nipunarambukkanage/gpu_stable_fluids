#include "gpu_fluids/runtime_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double runtime_remaining_frames(double first, double second, double third) noexcept {
  return std::max(0.0, second - first);
}

}  // namespace gpu_fluids

