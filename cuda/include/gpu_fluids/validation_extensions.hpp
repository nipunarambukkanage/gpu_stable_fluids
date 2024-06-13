#pragma once

namespace gpu_fluids {

double validation_finite_score(double first, double second, double third) noexcept;
double validation_nonnegative_score(double first, double second, double third) noexcept;
double validation_range_score(double first, double second, double third) noexcept;
double validation_alpha_score(double first, double second, double third) noexcept;
double validation_finite_ratio(double first, double second, double third) noexcept;
double validation_error_weight(double first, double second, double third) noexcept;
double validation_dimension_score(double first, double second, double third) noexcept;
double validation_byte_ratio(double first, double second, double third) noexcept;
double validation_speed_limit(double first, double second, double third) noexcept;
double validation_energy_limit(double first, double second, double third) noexcept;

}  // namespace gpu_fluids

