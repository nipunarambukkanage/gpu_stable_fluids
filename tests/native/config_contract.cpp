#include "gpu_fluids/config.hpp"

int main() {
  static_assert(gpu_fluids::kGridWidth == 512);
  static_assert(gpu_fluids::kGridHeight == 512);
  static_assert(gpu_fluids::kPressureIterations == 20);
  static_assert(sizeof(gpu_fluids::SimulationParams) == 112);
  static_assert(sizeof(gpu_fluids::SphParams) == 64);

  return gpu_fluids::kMaxBacktraceDistance > 0.0F &&
                 gpu_fluids::kTileExtent == gpu_fluids::kBlockSize + 2
             ? 0
             : 1;
}
