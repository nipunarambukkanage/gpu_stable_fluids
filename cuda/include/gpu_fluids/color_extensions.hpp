#pragma once

namespace gpu_fluids {

double color_luminance709(double first, double second, double third) noexcept;
double color_srgb_to_linear(double first, double second, double third) noexcept;
double color_linear_to_srgb(double first, double second, double third) noexcept;
double color_alpha_coverage(double first, double second, double third) noexcept;
double color_premultiply(double first, double second, double third) noexcept;
double color_unpremultiply(double first, double second, double third) noexcept;
double color_channel_distance(double first, double second, double third) noexcept;
double color_palette_mix(double first, double second, double third) noexcept;
double color_exposure_gain(double first, double second, double third) noexcept;
double color_contrast_gain(double first, double second, double third) noexcept;

}  // namespace gpu_fluids

