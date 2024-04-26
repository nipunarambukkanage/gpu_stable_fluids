#include "gpu_fluids/runtime_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double runtime_frame_due(double first, double second, double third) noexcept {
  return second > 0.0 && std::fmod(first, second) == 0.0 ? 1.0 : 0.0;
}

}  // namespace gpu_fluids

