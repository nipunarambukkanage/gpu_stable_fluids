#include "gpu_fluids/numeric_validation.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

int fail(const char* message) {
  std::cerr << "native_numeric_validation_contract: " << message << '\n';
  return 1;
}

}  // namespace

int main() {
  const auto finite = gpu_fluids::validateFiniteField({0.0F, 0.5F, -0.25F}, "velocity", 10.0F);
  if (!finite.valid() || finite.checkCount() == 0 || finite.errorCount() != 0) {
    return fail("finite field was incorrectly rejected");
  }
  const auto nonFinite = gpu_fluids::validateFiniteField(
      {0.0F, std::numeric_limits<float>::quiet_NaN()}, "velocity", 10.0F);
  if (nonFinite.valid() || nonFinite.errorCount() == 0 ||
      nonFinite.toJson().find("non-finite") == std::string::npos) {
    return fail("non-finite field was not reported");
  }

  std::vector<std::uint8_t> frame(4U * 3U * 4U, 0U);
  for (std::size_t pixel = 0; pixel < frame.size(); pixel += 4) {
    frame[pixel + 3] = 255;
  }
  const auto validFrame = gpu_fluids::validateRgbaFrame(frame, 4, 3);
  if (!validFrame.valid() || validFrame.errorCount() != 0) {
    return fail("valid RGBA frame was incorrectly rejected");
  }
  frame.pop_back();
  const auto invalidFrame = gpu_fluids::validateRgbaFrame(frame, 4, 3);
  if (invalidFrame.valid() || invalidFrame.errorCount() == 0) {
    return fail("invalid RGBA byte count was not reported");
  }

  const auto diagnostics = gpu_fluids::validateSimulationDiagnostics(2.0F, 4.0F, 12);
  if (!diagnostics.valid() || diagnostics.checkCount() != 3) {
    return fail("finite simulation diagnostics were incorrectly rejected");
  }
  const auto invalidDiagnostics = gpu_fluids::validateSimulationDiagnostics(
      -1.0F, std::numeric_limits<float>::infinity(), 0);
  if (invalidDiagnostics.valid() || invalidDiagnostics.errorCount() != 3 ||
      invalidDiagnostics.summary().find("invalid") == std::string::npos) {
    return fail("invalid simulation diagnostics were not fully reported");
  }
  std::cout << "native_numeric_validation_contract: passed\n";
  return 0;
}
