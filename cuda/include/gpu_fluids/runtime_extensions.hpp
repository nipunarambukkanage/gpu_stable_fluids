#pragma once

namespace gpu_fluids {

double runtime_remaining_frames(double first, double second, double third) noexcept;
double runtime_frame_due(double first, double second, double third) noexcept;
double runtime_frame_complete(double first, double second, double third) noexcept;
double runtime_pressure_limit(double first, double second, double third) noexcept;
double runtime_next_sequence(double first, double second, double third) noexcept;
double runtime_timestamp(double first, double second, double third) noexcept;
double runtime_seconds_to_millis(double first, double second, double third) noexcept;
double runtime_export_boundary(double first, double second, double third) noexcept;
double runtime_command_terminal(double first, double second, double third) noexcept;
double runtime_idle(double first, double second, double third) noexcept;

}  // namespace gpu_fluids

