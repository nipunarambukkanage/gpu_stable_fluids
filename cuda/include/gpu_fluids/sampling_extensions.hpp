#pragma once

namespace gpu_fluids {

double sampling_lerp(double first, double second, double third) noexcept;
double sampling_smoothstep(double first, double second, double third) noexcept;
double sampling_bilinear_value(double first, double second, double third) noexcept;
double sampling_gaussian_weight(double first, double second, double third) noexcept;
double sampling_segment_projection(double first, double second, double third) noexcept;
double sampling_segment_distance(double first, double second, double third) noexcept;
double sampling_nearest_coordinate(double first, double second, double third) noexcept;
double sampling_cubic_weight(double first, double second, double third) noexcept;
double sampling_antialias_mix(double first, double second, double third) noexcept;
double sampling_clamp_coordinate(double first, double second, double third) noexcept;

}  // namespace gpu_fluids

