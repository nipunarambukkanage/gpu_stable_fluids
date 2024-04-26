#pragma once

namespace gpu_fluids {

double telemetry_mean(double first, double second, double third) noexcept;
double telemetry_percentile_rank(double first, double second, double third) noexcept;
double telemetry_minimum(double first, double second, double third) noexcept;
double telemetry_maximum(double first, double second, double third) noexcept;
double telemetry_byte_total(double first, double second, double third) noexcept;
double telemetry_bandwidth_gbps(double first, double second, double third) noexcept;
double telemetry_variance(double first, double second, double third) noexcept;
double telemetry_stage_share(double first, double second, double third) noexcept;
double telemetry_sanitize_millis(double first, double second, double third) noexcept;
double telemetry_samples_per_second(double first, double second, double third) noexcept;

}  // namespace gpu_fluids

