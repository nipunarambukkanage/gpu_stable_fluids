#pragma once

namespace gpu_fluids {

double fluid_speed_squared(double first, double second, double third) noexcept;
double fluid_kinetic_energy(double first, double second, double third) noexcept;
double fluid_cfl_number(double first, double second, double third) noexcept;
double fluid_pressure_error(double first, double second, double third) noexcept;
double fluid_vorticity(double first, double second, double third) noexcept;
double fluid_divergence(double first, double second, double third) noexcept;
double fluid_dissipation(double first, double second, double third) noexcept;
double fluid_backtrace_scale(double first, double second, double third) noexcept;
double fluid_mix_scalar(double first, double second, double third) noexcept;
double fluid_density_clamp(double first, double second, double third) noexcept;

}  // namespace gpu_fluids

