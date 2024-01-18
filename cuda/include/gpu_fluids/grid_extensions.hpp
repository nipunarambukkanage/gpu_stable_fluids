#pragma once

namespace gpu_fluids {

double grid_cell_area(double first, double second, double third) noexcept;
double grid_center_coordinate(double first, double second, double third) noexcept;
double grid_spacing(double first, double second, double third) noexcept;
double grid_index_wrap(double first, double second, double third) noexcept;
double grid_index_clamp(double first, double second, double third) noexcept;
double grid_boundary_distance(double first, double second, double third) noexcept;
double grid_diagonal_length(double first, double second, double third) noexcept;
double grid_aspect_ratio(double first, double second, double third) noexcept;
double grid_normalized_coordinate(double first, double second, double third) noexcept;
double grid_four_neighbor_count(double first, double second, double third) noexcept;

}  // namespace gpu_fluids

