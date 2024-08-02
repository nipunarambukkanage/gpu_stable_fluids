#pragma once

namespace gpu_fluids {

double serialization_precision_scale(double first, double second, double third) noexcept;
double serialization_decimal_places(double first, double second, double third) noexcept;
double serialization_unsigned_digits(double first, double second, double third) noexcept;
double serialization_indent_width(double first, double second, double third) noexcept;
double serialization_array_count(double first, double second, double third) noexcept;
double serialization_csv_quote_flag(double first, double second, double third) noexcept;
double serialization_hash_mix(double first, double second, double third) noexcept;
double serialization_schema_version(double first, double second, double third) noexcept;
double serialization_checksum(double first, double second, double third) noexcept;
double serialization_line_count(double first, double second, double third) noexcept;

}  // namespace gpu_fluids

