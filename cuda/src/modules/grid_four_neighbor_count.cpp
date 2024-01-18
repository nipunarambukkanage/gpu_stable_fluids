#include "gpu_fluids/grid_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double grid_four_neighbor_count(double first, double second, double third) noexcept {
  return (first > 0.0 ? 1.0 : 0.0) + (second > 0.0 ? 1.0 : 0.0) + (third > 0.0 ? 1.0 : 0.0);
}

}  // namespace gpu_fluids

