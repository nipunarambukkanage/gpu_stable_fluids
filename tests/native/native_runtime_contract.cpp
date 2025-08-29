#include "gpu_fluids/native_runtime.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>

namespace {

int fail(const char* message) {
  std::cerr << "native_runtime_contract: " << message << '\n';
  return 1;
}

gpu_fluids::RuntimeConfig testConfig(int frameLimit) {
  gpu_fluids::RuntimeConfig config;
  config.width = 24;
  config.height = 20;
  config.frameLimit = frameLimit;
  config.exportFrames = false;
  config.writeTelemetry = false;
  config.pressureIterations = 6;
  return config;
}

}  // namespace

int main() {
  gpu_fluids::NativeReferenceRuntime runtime(testConfig(3));
  runtime.run();
  const auto& firstReport = runtime.report();
  if (firstReport.framesSimulated != 3 || firstReport.telemetry.frames != 3 ||
      runtime.paused() || runtime.stopped() || !runtime.finished()) {
    return fail("runtime did not complete its configured frame budget");
  }
  if (!std::isfinite(static_cast<double>(firstReport.finalDyeEnergy)) ||
      firstReport.maximumObservedSpeed <= 0.0F) {
    return fail("runtime report did not expose finite simulation diagnostics");
  }

  gpu_fluids::NativeReferenceRuntime commands(testConfig(10));
  commands.enqueue(gpu_fluids::RuntimeCommandType::Pause);
  commands.run();
  if (!commands.paused() || commands.report().framesSimulated != 0) {
    return fail("pause command was not applied before simulation");
  }
  commands.enqueue(gpu_fluids::RuntimeCommandType::SingleStep);
  commands.run();
  if (!commands.paused() || commands.report().framesSimulated != 1) {
    return fail("single-step command did not advance exactly one frame");
  }
  commands.enqueue(gpu_fluids::RuntimeCommandType::Resume);
  commands.run();
  if (commands.report().framesSimulated != 10 || commands.paused()) {
    return fail("resume command did not consume the remaining frame budget");
  }

  commands.enqueue(gpu_fluids::RuntimeCommandType::Reset);
  commands.enqueue(gpu_fluids::RuntimeCommandType::Stop);
  commands.stepOnce();
  if (!commands.stopped() || commands.report().framesSimulated != 0 ||
      commands.report().commandsApplied < 5) {
    return fail("reset and stop command ordering was not deterministic");
  }

  const std::filesystem::path outputDirectory =
      std::filesystem::temp_directory_path() / "gpu-stable-fluids-runtime-contract";
  std::filesystem::create_directories(outputDirectory);
  commands.writeReport(outputDirectory / "report.json");
  if (!std::filesystem::exists(outputDirectory / "report.json")) {
    return fail("runtime report was not written");
  }
  std::filesystem::remove(outputDirectory / "report.json");
  std::filesystem::remove(outputDirectory);

  std::cout << "native_runtime_contract: passed\n";
  return 0;
}
