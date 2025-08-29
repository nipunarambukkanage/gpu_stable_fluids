#include "gpu_fluids/experiment_manifest.hpp"

#include <filesystem>
#include <iostream>
#include <string>

namespace {

int fail(const char* message) {
  std::cerr << "native_experiment_manifest_contract: " << message << '\n';
  return 1;
}

}  // namespace

int main() {
  gpu_fluids::ExperimentManifest manifest;
  manifest.setString("name", "stencil \"baseline\"");
  manifest.setInteger("frames", 4);
  manifest.setNumber("dt", 1.0 / 60.0);
  manifest.setBoolean("deterministic", true);
  manifest.setStringArray("artifacts", {"frame-0001.ppm", "telemetry.json"});
  const std::string json = manifest.toJson();
  if (manifest.empty() || json.find("stencil \\\"baseline\\\"") == std::string::npos ||
      json.find("\"frames\": 4") == std::string::npos ||
      json.find("\"deterministic\": true") == std::string::npos) {
    return fail("manifest serialization did not preserve typed values");
  }

  gpu_fluids::RuntimeConfig config;
  config.width = 32;
  config.height = 24;
  config.frameLimit = 4;
  config.exportEvery = 2;
  config.pressureIterations = 8;
  gpu_fluids::RuntimeReport report;
  report.framesSimulated = 4;
  report.framesExported = 2;
  report.validationChecks = 20;
  report.telemetry.frames = 4;
  report.telemetry.averageFrameMilliseconds = 2.5;
  report.telemetry.p95FrameMilliseconds = 3.5;
  report.exports.push_back({2, "frame-0002.ppm", 128});
  const auto runtimeManifest = gpu_fluids::makeRuntimeManifest(config, report);
  const std::filesystem::path output =
      std::filesystem::temp_directory_path() / "gpu-stable-fluids-manifest-contract.json";
  runtimeManifest.write(output);
  if (!std::filesystem::exists(output)) {
    return fail("runtime manifest was not written");
  }
  std::filesystem::remove(output);
  std::cout << "native_experiment_manifest_contract: passed\n";
  return 0;
}
