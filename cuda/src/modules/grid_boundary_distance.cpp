#include "gpu_fluids/grid_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double grid_boundary_distance(double first, double second, double third) noexcept {
  return std::min(std::min(first, second), third);
}

}  // namespace gpu_fluids

