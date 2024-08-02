#pragma once

namespace gpu_fluids {

double performance_rgba_bytes(double first, double second, double third) noexcept;
double performance_tile_bytes(double first, double second, double third) noexcept;
double performance_transaction_count(double first, double second, double third) noexcept;
double performance_occupancy(double first, double second, double third) noexcept;
double performance_work_group_count(double first, double second, double third) noexcept;
double performance_memory_gbps(double first, double second, double third) noexcept;
double performance_compute_gflops(double first, double second, double third) noexcept;
double performance_arithmetic_intensity(double first, double second, double third) noexcept;
double performance_frame_budget(double first, double second, double third) noexcept;
double performance_latency_headroom(double first, double second, double third) noexcept;

}  // namespace gpu_fluids

