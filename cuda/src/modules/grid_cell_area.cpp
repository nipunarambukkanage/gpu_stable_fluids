#include "gpu_fluids/grid_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double grid_cell_area(double first, double second, double third) noexcept {
  return first > 0.0 && second > 0.0 ? first * second : 0.0;
}

}  // namespace gpu_fluids

