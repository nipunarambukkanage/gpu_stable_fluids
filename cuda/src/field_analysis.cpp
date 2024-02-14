#include "gpu_fluids/field_analysis.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace gpu_fluids {

ScalarFieldSummary analyzeScalarField(const std::vector<float>& field) noexcept {
  ScalarFieldSummary summary;
  summary.samples = field.size();

  double mean = 0.0;
  double sumOfSquares = 0.0;
  double sumOfSquaredDifferences = 0.0;
  bool hasPrevious = false;
  float previous = 0.0F;

  for (const float value : field) {
    if (!std::isfinite(value)) {
      ++summary.nonFiniteSamples;
      hasPrevious = false;
      continue;
    }

    ++summary.finiteSamples;
    summary.minimum = summary.finiteSamples == 1 ? value : std::fmin(summary.minimum, value);
    summary.maximum = summary.finiteSamples == 1 ? value : std::fmax(summary.maximum, value);
    summary.absoluteSum += std::abs(static_cast<double>(value));
    sumOfSquares += static_cast<double>(value) * static_cast<double>(value);

    const double delta = static_cast<double>(value) - mean;
    mean += delta / static_cast<double>(summary.finiteSamples);
    sumOfSquaredDifferences += delta * (static_cast<double>(value) - mean);

    if (hasPrevious) {
      summary.totalVariation += std::abs(static_cast<double>(value) - static_cast<double>(previous));
    }
    previous = value;
    hasPrevious = true;
  }

  if (summary.finiteSamples == 0) {
    return summary;
  }

  summary.mean = mean;
  summary.variance = sumOfSquaredDifferences / static_cast<double>(summary.finiteSamples);
  summary.rootMeanSquare = std::sqrt(sumOfSquares / static_cast<double>(summary.finiteSamples));
  return summary;
}

std::vector<std::size_t> buildScalarHistogram(const std::vector<float>& field,
                                               std::size_t bucketCount,
                                               float minimum,
                                               float maximum) {
  if (bucketCount == 0) {
    throw std::invalid_argument("scalar histogram requires at least one bucket");
  }
  if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum >= maximum) {
    throw std::invalid_argument("scalar histogram requires a finite increasing range");
  }

  std::vector<std::size_t> histogram(bucketCount, 0);
  const double reciprocalRange = 1.0 / (static_cast<double>(maximum) - static_cast<double>(minimum));
  for (const float value : field) {
    if (!std::isfinite(value) || value < minimum || value > maximum) {
      continue;
    }
    const double normalized = (static_cast<double>(value) - static_cast<double>(minimum)) * reciprocalRange;
    const std::size_t bucket = std::min(
        bucketCount - 1,
        static_cast<std::size_t>(normalized * static_cast<double>(bucketCount)));
    ++histogram[bucket];
  }
  return histogram;
}

}  // namespace gpu_fluids
