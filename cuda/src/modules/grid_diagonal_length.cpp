#include "gpu_fluids/grid_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double grid_diagonal_length(double first, double second, double third) noexcept {
  return std::hypot(first, second);
}

}  // namespace gpu_fluids

