#include "gpu_fluids/runtime_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double runtime_frame_complete(double first, double second, double third) noexcept {
  return first >= second ? 1.0 : 0.0;
}

}  // namespace gpu_fluids

