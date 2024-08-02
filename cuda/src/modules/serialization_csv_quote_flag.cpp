#include "gpu_fluids/serialization_extensions.hpp"
#include <algorithm>
#include <cmath>
namespace gpu_fluids {
double serialization_csv_quote_flag(double first, double second, double third) noexcept {
  return first > 0.0 || second > 0.0 || third > 0.0 ? 1.0 : 0.0;
}
}  // namespace gpu_fluids

