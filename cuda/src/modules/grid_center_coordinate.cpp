#include "gpu_fluids/grid_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double grid_center_coordinate(double first, double second, double third) noexcept {
  return first + (second - first) * 0.5;
}

}  // namespace gpu_fluids

