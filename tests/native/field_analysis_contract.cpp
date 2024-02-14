#include "gpu_fluids/field_analysis.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

int fail(const char* message) {
  std::cerr << "native_field_analysis_contract: " << message << '\n';
  return 1;
}

template <typename Callable>
bool throwsInvalidArgument(Callable&& callable) {
  try {
    callable();
  } catch (const std::invalid_argument&) {
    return true;
  }
  return false;
}

}  // namespace

int main() {
  const gpu_fluids::ScalarFieldSummary summary =
      gpu_fluids::analyzeScalarField({0.0F, 1.0F, 2.0F, 3.0F});
  if (summary.samples != 4 || summary.finiteSamples != 4 || summary.nonFiniteSamples != 0 ||
      std::abs(summary.minimum - 0.0F) > 1.0e-6F || std::abs(summary.maximum - 3.0F) > 1.0e-6F ||
      std::abs(summary.mean - 1.5) > 1.0e-9 || std::abs(summary.variance - 1.25) > 1.0e-9 ||
      std::abs(summary.rootMeanSquare - std::sqrt(3.5)) > 1.0e-9 ||
      std::abs(summary.absoluteSum - 6.0) > 1.0e-9 || std::abs(summary.totalVariation - 3.0) > 1.0e-9) {
    return fail("scalar summary statistics are incorrect");
  }

  const gpu_fluids::ScalarFieldSummary nonFinite = gpu_fluids::analyzeScalarField(
      {1.0F, std::numeric_limits<float>::quiet_NaN(), -1.0F, std::numeric_limits<float>::infinity()});
  if (nonFinite.samples != 4 || nonFinite.finiteSamples != 2 || nonFinite.nonFiniteSamples != 2 ||
      std::abs(nonFinite.mean) > 1.0e-9 || std::abs(nonFinite.totalVariation) > 1.0e-9) {
    return fail("non-finite values were not isolated from field statistics");
  }

  const std::vector<std::size_t> histogram =
      gpu_fluids::buildScalarHistogram({-1.0F, 0.0F, 0.25F, 0.75F, 1.0F, 2.0F}, 4, 0.0F, 1.0F);
  if (histogram != std::vector<std::size_t>{1, 1, 0, 2}) {
    return fail("scalar histogram bucket assignment is incorrect");
  }

  if (!throwsInvalidArgument([] { gpu_fluids::buildScalarHistogram({}, 0, 0.0F, 1.0F); }) ||
      !throwsInvalidArgument([] { gpu_fluids::buildScalarHistogram({}, 2, 1.0F, 1.0F); })) {
    return fail("invalid histogram configuration was accepted");
  }

  std::cout << "native_field_analysis_contract: passed\n";
  return 0;
}
