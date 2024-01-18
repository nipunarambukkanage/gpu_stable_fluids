#include "gpu_fluids/grid_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double grid_normalized_coordinate(double first, double second, double third) noexcept {
  return second > 1.0 ? std::max(0.0, std::min(first, second - 1.0)) / (second - 1.0) : 0.0;
}

}  // namespace gpu_fluids

