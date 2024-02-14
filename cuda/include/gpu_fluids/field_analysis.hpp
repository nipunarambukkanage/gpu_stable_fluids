#pragma once

#include <cstddef>
#include <vector>

namespace gpu_fluids {

// Summary statistics for a scalar grid. Non-finite values are counted but do
// not participate in the numerical aggregates, so callers can report a useful
// diagnostic without allowing a single bad cell to poison every measurement.
struct ScalarFieldSummary final {
  std::size_t samples = 0;
  std::size_t finiteSamples = 0;
  std::size_t nonFiniteSamples = 0;
  float minimum = 0.0F;
  float maximum = 0.0F;
  double mean = 0.0;
  double variance = 0.0;
  double rootMeanSquare = 0.0;
  double absoluteSum = 0.0;
  double totalVariation = 0.0;
};

// Computes stable summary statistics in one pass over a scalar field.
[[nodiscard]] ScalarFieldSummary analyzeScalarField(const std::vector<float>& field) noexcept;

// Divides the closed interval [minimum, maximum] into equally sized buckets.
// Values outside the interval and non-finite values are ignored. A value equal
// to maximum belongs to the final bucket.
[[nodiscard]] std::vector<std::size_t> buildScalarHistogram(const std::vector<float>& field,
                                                             std::size_t bucketCount,
                                                             float minimum,
                                                             float maximum);

}  // namespace gpu_fluids
