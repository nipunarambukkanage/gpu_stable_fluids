# Native extension module catalog

The native reference target contains 100 deliberately small translation units in
`cuda/src/modules`. Each file owns one deterministic calculation so numerical
contracts can link and test responsibilities independently. The modules are
compiled into `fluid_reference`; `native_module_contract` links every symbol.

## Grid geometry

| Translation unit | Responsibility |
| --- | --- |
| `grid_cell_area.cpp` | Computes cell area for native diagnostics and scheduling. |
| `grid_center_coordinate.cpp` | Computes center coordinate for native diagnostics and scheduling. |
| `grid_spacing.cpp` | Computes spacing for native diagnostics and scheduling. |
| `grid_index_wrap.cpp` | Computes index wrap for native diagnostics and scheduling. |
| `grid_index_clamp.cpp` | Computes index clamp for native diagnostics and scheduling. |
| `grid_boundary_distance.cpp` | Computes boundary distance for native diagnostics and scheduling. |
| `grid_diagonal_length.cpp` | Computes diagonal length for native diagnostics and scheduling. |
| `grid_aspect_ratio.cpp` | Computes aspect ratio for native diagnostics and scheduling. |
| `grid_normalized_coordinate.cpp` | Computes normalized coordinate for native diagnostics and scheduling. |
| `grid_four_neighbor_count.cpp` | Computes four neighbor count for native diagnostics and scheduling. |

## Fluid numerics

| Translation unit | Responsibility |
| --- | --- |
| `fluid_speed_squared.cpp` | Computes speed squared for native diagnostics and scheduling. |
| `fluid_kinetic_energy.cpp` | Computes kinetic energy for native diagnostics and scheduling. |
| `fluid_cfl_number.cpp` | Computes cfl number for native diagnostics and scheduling. |
| `fluid_pressure_error.cpp` | Computes pressure error for native diagnostics and scheduling. |
| `fluid_vorticity.cpp` | Computes vorticity for native diagnostics and scheduling. |
| `fluid_divergence.cpp` | Computes divergence for native diagnostics and scheduling. |
| `fluid_dissipation.cpp` | Computes dissipation for native diagnostics and scheduling. |
| `fluid_backtrace_scale.cpp` | Computes backtrace scale for native diagnostics and scheduling. |
| `fluid_mix_scalar.cpp` | Computes mix scalar for native diagnostics and scheduling. |
| `fluid_density_clamp.cpp` | Computes density clamp for native diagnostics and scheduling. |

## Color transforms

| Translation unit | Responsibility |
| --- | --- |
| `color_luminance709.cpp` | Computes luminance709 for native diagnostics and scheduling. |
| `color_srgb_to_linear.cpp` | Computes srgb to linear for native diagnostics and scheduling. |
| `color_linear_to_srgb.cpp` | Computes linear to srgb for native diagnostics and scheduling. |
| `color_alpha_coverage.cpp` | Computes alpha coverage for native diagnostics and scheduling. |
| `color_premultiply.cpp` | Computes premultiply for native diagnostics and scheduling. |
| `color_unpremultiply.cpp` | Computes unpremultiply for native diagnostics and scheduling. |
| `color_channel_distance.cpp` | Computes channel distance for native diagnostics and scheduling. |
| `color_palette_mix.cpp` | Computes palette mix for native diagnostics and scheduling. |
| `color_exposure_gain.cpp` | Computes exposure gain for native diagnostics and scheduling. |
| `color_contrast_gain.cpp` | Computes contrast gain for native diagnostics and scheduling. |

## Sampling kernels

| Translation unit | Responsibility |
| --- | --- |
| `sampling_lerp.cpp` | Computes lerp for native diagnostics and scheduling. |
| `sampling_smoothstep.cpp` | Computes smoothstep for native diagnostics and scheduling. |
| `sampling_bilinear_value.cpp` | Computes bilinear value for native diagnostics and scheduling. |
| `sampling_gaussian_weight.cpp` | Computes gaussian weight for native diagnostics and scheduling. |
| `sampling_segment_projection.cpp` | Computes segment projection for native diagnostics and scheduling. |
| `sampling_segment_distance.cpp` | Computes segment distance for native diagnostics and scheduling. |
| `sampling_nearest_coordinate.cpp` | Computes nearest coordinate for native diagnostics and scheduling. |
| `sampling_cubic_weight.cpp` | Computes cubic weight for native diagnostics and scheduling. |
| `sampling_antialias_mix.cpp` | Computes antialias mix for native diagnostics and scheduling. |
| `sampling_clamp_coordinate.cpp` | Computes clamp coordinate for native diagnostics and scheduling. |

## Runtime policy

| Translation unit | Responsibility |
| --- | --- |
| `runtime_remaining_frames.cpp` | Computes remaining frames for native diagnostics and scheduling. |
| `runtime_frame_due.cpp` | Computes frame due for native diagnostics and scheduling. |
| `runtime_frame_complete.cpp` | Computes frame complete for native diagnostics and scheduling. |
| `runtime_pressure_limit.cpp` | Computes pressure limit for native diagnostics and scheduling. |
| `runtime_next_sequence.cpp` | Computes next sequence for native diagnostics and scheduling. |
| `runtime_timestamp.cpp` | Computes timestamp for native diagnostics and scheduling. |
| `runtime_seconds_to_millis.cpp` | Computes seconds to millis for native diagnostics and scheduling. |
| `runtime_export_boundary.cpp` | Computes export boundary for native diagnostics and scheduling. |
| `runtime_command_terminal.cpp` | Computes command terminal for native diagnostics and scheduling. |
| `runtime_idle.cpp` | Computes idle for native diagnostics and scheduling. |

## Telemetry math

| Translation unit | Responsibility |
| --- | --- |
| `telemetry_mean.cpp` | Computes mean for native diagnostics and scheduling. |
| `telemetry_percentile_rank.cpp` | Computes percentile rank for native diagnostics and scheduling. |
| `telemetry_minimum.cpp` | Computes minimum for native diagnostics and scheduling. |
| `telemetry_maximum.cpp` | Computes maximum for native diagnostics and scheduling. |
| `telemetry_byte_total.cpp` | Computes byte total for native diagnostics and scheduling. |
| `telemetry_bandwidth_gbps.cpp` | Computes bandwidth gbps for native diagnostics and scheduling. |
| `telemetry_variance.cpp` | Computes variance for native diagnostics and scheduling. |
| `telemetry_stage_share.cpp` | Computes stage share for native diagnostics and scheduling. |
| `telemetry_sanitize_millis.cpp` | Computes sanitize millis for native diagnostics and scheduling. |
| `telemetry_samples_per_second.cpp` | Computes samples per second for native diagnostics and scheduling. |

## Validation scoring

| Translation unit | Responsibility |
| --- | --- |
| `validation_finite_score.cpp` | Computes finite score for native diagnostics and scheduling. |
| `validation_nonnegative_score.cpp` | Computes nonnegative score for native diagnostics and scheduling. |
| `validation_range_score.cpp` | Computes range score for native diagnostics and scheduling. |
| `validation_alpha_score.cpp` | Computes alpha score for native diagnostics and scheduling. |
| `validation_finite_ratio.cpp` | Computes finite ratio for native diagnostics and scheduling. |
| `validation_error_weight.cpp` | Computes error weight for native diagnostics and scheduling. |
| `validation_dimension_score.cpp` | Computes dimension score for native diagnostics and scheduling. |
| `validation_byte_ratio.cpp` | Computes byte ratio for native diagnostics and scheduling. |
| `validation_speed_limit.cpp` | Computes speed limit for native diagnostics and scheduling. |
| `validation_energy_limit.cpp` | Computes energy limit for native diagnostics and scheduling. |

## Scenario curves

| Translation unit | Responsibility |
| --- | --- |
| `scenario_normalized_sine.cpp` | Computes normalized sine for native diagnostics and scheduling. |
| `scenario_triangle_wave.cpp` | Computes triangle wave for native diagnostics and scheduling. |
| `scenario_sawtooth_wave.cpp` | Computes sawtooth wave for native diagnostics and scheduling. |
| `scenario_ease_in_out.cpp` | Computes ease in out for native diagnostics and scheduling. |
| `scenario_keyframe_lerp.cpp` | Computes keyframe lerp for native diagnostics and scheduling. |
| `scenario_periodic_phase.cpp` | Computes periodic phase for native diagnostics and scheduling. |
| `scenario_radial_distance.cpp` | Computes radial distance for native diagnostics and scheduling. |
| `scenario_orbit_angle.cpp` | Computes orbit angle for native diagnostics and scheduling. |
| `scenario_pulse_gate.cpp` | Computes pulse gate for native diagnostics and scheduling. |
| `scenario_smooth_pulse.cpp` | Computes smooth pulse for native diagnostics and scheduling. |

## Serialization planning

| Translation unit | Responsibility |
| --- | --- |
| `serialization_precision_scale.cpp` | Computes precision scale for native diagnostics and scheduling. |
| `serialization_decimal_places.cpp` | Computes decimal places for native diagnostics and scheduling. |
| `serialization_unsigned_digits.cpp` | Computes unsigned digits for native diagnostics and scheduling. |
| `serialization_indent_width.cpp` | Computes indent width for native diagnostics and scheduling. |
| `serialization_array_count.cpp` | Computes array count for native diagnostics and scheduling. |
| `serialization_csv_quote_flag.cpp` | Computes csv quote flag for native diagnostics and scheduling. |
| `serialization_hash_mix.cpp` | Computes hash mix for native diagnostics and scheduling. |
| `serialization_schema_version.cpp` | Computes schema version for native diagnostics and scheduling. |
| `serialization_checksum.cpp` | Computes checksum for native diagnostics and scheduling. |
| `serialization_line_count.cpp` | Computes line count for native diagnostics and scheduling. |

## Performance estimates

| Translation unit | Responsibility |
| --- | --- |
| `performance_rgba_bytes.cpp` | Computes rgba bytes for native diagnostics and scheduling. |
| `performance_tile_bytes.cpp` | Computes tile bytes for native diagnostics and scheduling. |
| `performance_transaction_count.cpp` | Computes transaction count for native diagnostics and scheduling. |
| `performance_occupancy.cpp` | Computes occupancy for native diagnostics and scheduling. |
| `performance_work_group_count.cpp` | Computes work group count for native diagnostics and scheduling. |
| `performance_memory_gbps.cpp` | Computes memory gbps for native diagnostics and scheduling. |
| `performance_compute_gflops.cpp` | Computes compute gflops for native diagnostics and scheduling. |
| `performance_arithmetic_intensity.cpp` | Computes arithmetic intensity for native diagnostics and scheduling. |
| `performance_frame_budget.cpp` | Computes frame budget for native diagnostics and scheduling. |
| `performance_latency_headroom.cpp` | Computes latency headroom for native diagnostics and scheduling. |


