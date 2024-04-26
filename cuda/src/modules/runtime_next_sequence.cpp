#include "gpu_fluids/runtime_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double runtime_next_sequence(double first, double second, double third) noexcept {
  return first >= third ? 1.0 : first + 1.0;
}

}  // namespace gpu_fluids

