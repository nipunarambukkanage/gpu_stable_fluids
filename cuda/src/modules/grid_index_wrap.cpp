#include "gpu_fluids/grid_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double grid_index_wrap(double first, double second, double third) noexcept {
  return second > 0.0 ? std::fmod(std::fabs(first), second) : 0.0;
}

}  // namespace gpu_fluids

