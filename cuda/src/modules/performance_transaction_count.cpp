#include "gpu_fluids/performance_extensions.hpp"
#include <algorithm>
#include <cmath>
namespace gpu_fluids {
double performance_transaction_count(double first, double second, double third) noexcept {
  return third > 0.0 ? std::ceil(first * second / third) : 0.0;
}
}  // namespace gpu_fluids

