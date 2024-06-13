#pragma once

namespace gpu_fluids {

double scenario_normalized_sine(double first, double second, double third) noexcept;
double scenario_triangle_wave(double first, double second, double third) noexcept;
double scenario_sawtooth_wave(double first, double second, double third) noexcept;
double scenario_ease_in_out(double first, double second, double third) noexcept;
double scenario_keyframe_lerp(double first, double second, double third) noexcept;
double scenario_periodic_phase(double first, double second, double third) noexcept;
double scenario_radial_distance(double first, double second, double third) noexcept;
double scenario_orbit_angle(double first, double second, double third) noexcept;
double scenario_pulse_gate(double first, double second, double third) noexcept;
double scenario_smooth_pulse(double first, double second, double third) noexcept;

}  // namespace gpu_fluids

