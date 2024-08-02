#include "gpu_fluids/serialization_extensions.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_fluids {

double serialization_indent_width(double first, double second, double third) noexcept {
  return std::max(0.0, first) * 2.0;
}

}  // namespace gpu_fluids

