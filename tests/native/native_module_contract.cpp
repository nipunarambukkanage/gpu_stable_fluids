#include "gpu_fluids/grid_extensions.hpp"
#include "gpu_fluids/fluid_extensions.hpp"
#include "gpu_fluids/color_extensions.hpp"
#include "gpu_fluids/sampling_extensions.hpp"
#include "gpu_fluids/runtime_extensions.hpp"
#include "gpu_fluids/telemetry_extensions.hpp"
#include "gpu_fluids/validation_extensions.hpp"
#include "gpu_fluids/scenario_extensions.hpp"
#include "gpu_fluids/serialization_extensions.hpp"
#include "gpu_fluids/performance_extensions.hpp"

#include <array>
#include <cmath>
#include <iostream>

namespace {
using Module = double (*)(double, double, double) noexcept;
int fail(const char* message) {
  std::cerr << "native_module_contract: " << message << '\n';
  return 1;
}
}  // namespace

int main() {
  const std::array<Module, 100> modules = {
      gpu_fluids::grid_cell_area,
      gpu_fluids::grid_center_coordinate,
      gpu_fluids::grid_spacing,
      gpu_fluids::grid_index_wrap,
      gpu_fluids::grid_index_clamp,
      gpu_fluids::grid_boundary_distance,
      gpu_fluids::grid_diagonal_length,
      gpu_fluids::grid_aspect_ratio,
      gpu_fluids::grid_normalized_coordinate,
      gpu_fluids::grid_four_neighbor_count,
      gpu_fluids::fluid_speed_squared,
      gpu_fluids::fluid_kinetic_energy,
      gpu_fluids::fluid_cfl_number,
      gpu_fluids::fluid_pressure_error,
      gpu_fluids::fluid_vorticity,
      gpu_fluids::fluid_divergence,
      gpu_fluids::fluid_dissipation,
      gpu_fluids::fluid_backtrace_scale,
      gpu_fluids::fluid_mix_scalar,
      gpu_fluids::fluid_density_clamp,
      gpu_fluids::color_luminance709,
      gpu_fluids::color_srgb_to_linear,
      gpu_fluids::color_linear_to_srgb,
      gpu_fluids::color_alpha_coverage,
      gpu_fluids::color_premultiply,
      gpu_fluids::color_unpremultiply,
      gpu_fluids::color_channel_distance,
      gpu_fluids::color_palette_mix,
      gpu_fluids::color_exposure_gain,
      gpu_fluids::color_contrast_gain,
      gpu_fluids::sampling_lerp,
      gpu_fluids::sampling_smoothstep,
      gpu_fluids::sampling_bilinear_value,
      gpu_fluids::sampling_gaussian_weight,
      gpu_fluids::sampling_segment_projection,
      gpu_fluids::sampling_segment_distance,
      gpu_fluids::sampling_nearest_coordinate,
      gpu_fluids::sampling_cubic_weight,
      gpu_fluids::sampling_antialias_mix,
      gpu_fluids::sampling_clamp_coordinate,
      gpu_fluids::runtime_remaining_frames,
      gpu_fluids::runtime_frame_due,
      gpu_fluids::runtime_frame_complete,
      gpu_fluids::runtime_pressure_limit,
      gpu_fluids::runtime_next_sequence,
      gpu_fluids::runtime_timestamp,
      gpu_fluids::runtime_seconds_to_millis,
      gpu_fluids::runtime_export_boundary,
      gpu_fluids::runtime_command_terminal,
      gpu_fluids::runtime_idle,
      gpu_fluids::telemetry_mean,
      gpu_fluids::telemetry_percentile_rank,
      gpu_fluids::telemetry_minimum,
      gpu_fluids::telemetry_maximum,
      gpu_fluids::telemetry_byte_total,
      gpu_fluids::telemetry_bandwidth_gbps,
      gpu_fluids::telemetry_variance,
      gpu_fluids::telemetry_stage_share,
      gpu_fluids::telemetry_sanitize_millis,
      gpu_fluids::telemetry_samples_per_second,
      gpu_fluids::validation_finite_score,
      gpu_fluids::validation_nonnegative_score,
      gpu_fluids::validation_range_score,
      gpu_fluids::validation_alpha_score,
      gpu_fluids::validation_finite_ratio,
      gpu_fluids::validation_error_weight,
      gpu_fluids::validation_dimension_score,
      gpu_fluids::validation_byte_ratio,
      gpu_fluids::validation_speed_limit,
      gpu_fluids::validation_energy_limit,
      gpu_fluids::scenario_normalized_sine,
      gpu_fluids::scenario_triangle_wave,
      gpu_fluids::scenario_sawtooth_wave,
      gpu_fluids::scenario_ease_in_out,
      gpu_fluids::scenario_keyframe_lerp,
      gpu_fluids::scenario_periodic_phase,
      gpu_fluids::scenario_radial_distance,
      gpu_fluids::scenario_orbit_angle,
      gpu_fluids::scenario_pulse_gate,
      gpu_fluids::scenario_smooth_pulse,
      gpu_fluids::serialization_precision_scale,
      gpu_fluids::serialization_decimal_places,
      gpu_fluids::serialization_unsigned_digits,
      gpu_fluids::serialization_indent_width,
      gpu_fluids::serialization_array_count,
      gpu_fluids::serialization_csv_quote_flag,
      gpu_fluids::serialization_hash_mix,
      gpu_fluids::serialization_schema_version,
      gpu_fluids::serialization_checksum,
      gpu_fluids::serialization_line_count,
      gpu_fluids::performance_rgba_bytes,
      gpu_fluids::performance_tile_bytes,
      gpu_fluids::performance_transaction_count,
      gpu_fluids::performance_occupancy,
      gpu_fluids::performance_work_group_count,
      gpu_fluids::performance_memory_gbps,
      gpu_fluids::performance_compute_gflops,
      gpu_fluids::performance_arithmetic_intensity,
      gpu_fluids::performance_frame_budget,
      gpu_fluids::performance_latency_headroom,
  };
  for (std::size_t index = 0; index < modules.size(); ++index) {
    const double value = modules[index](1.25, 2.0, 0.5);
    if (!std::isfinite(value)) return fail("module returned a non-finite result");
  }
  if (std::abs(gpu_fluids::grid_cell_area(4.0, 5.0, 0.0) - 20.0) > 1.0e-9 ||
      std::abs(gpu_fluids::fluid_speed_squared(3.0, 4.0, 0.0) - 25.0) > 1.0e-9 ||
      std::abs(gpu_fluids::color_luminance709(0.0, 1.0, 0.0) - 0.7152) > 1.0e-9 ||
      std::abs(gpu_fluids::performance_occupancy(3.0, 4.0, 0.0) - 0.75) > 1.0e-9) {
    return fail("representative module responsibilities are incorrect");
  }
  std::cout << "native_module_contract: passed " << modules.size() << " modules\n";
  return 0;
}

