#include "gpu_fluids/native_runtime.hpp"
#include "gpu_fluids/experiment_manifest.hpp"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct ReferenceOptions {
  gpu_fluids::RuntimeConfig runtime{};
};

void printUsage() {
  std::cout
      << "fluid_reference_demo [options]\n"
      << "  --frames N                 Number of deterministic frames (default: 60)\n"
      << "  --export-every N           Export every Nth frame as PPM (default: 30)\n"
      << "  --width N                  Reference grid width (default: 128)\n"
      << "  --height N                 Reference grid height (default: 128)\n"
      << "  --pressure-iterations N    Jacobi iterations per frame (default: 20)\n"
      << "  --dt SECONDS              Fixed simulation timestep (default: 0.016667)\n"
      << "  --output DIR               Frame, report, and telemetry directory\n"
      << "  --no-export                Run simulation without PPM output\n"
      << "  --no-telemetry             Disable JSON and CSV telemetry output\n"
      << "  --quiet                    Print only fatal errors\n"
      << "  --help                     Show this message\n";
}

int parsePositive(const std::string& value, const char* option) {
  std::size_t consumed = 0;
  const int parsed = std::stoi(value, &consumed);
  if (consumed != value.size() || parsed <= 0) {
    throw std::invalid_argument(std::string(option) + " expects a positive integer");
  }
  return parsed;
}

float parseDeltaTime(const std::string& value) {
  std::size_t consumed = 0;
  const float parsed = std::stof(value, &consumed);
  if (consumed != value.size() || parsed < 1.0e-4F || parsed > 0.05F) {
    throw std::invalid_argument("--dt expects a value between 0.0001 and 0.05");
  }
  return parsed;
}

ReferenceOptions parseArguments(int argc, char** argv) {
  ReferenceOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help") {
      printUsage();
      std::exit(0);
    }
    if (argument == "--quiet") {
      options.runtime.quiet = true;
      continue;
    }
    if (argument == "--no-export") {
      options.runtime.exportFrames = false;
      continue;
    }
    if (argument == "--no-telemetry") {
      options.runtime.writeTelemetry = false;
      continue;
    }
    if (argument == "--frames" || argument == "--export-every" || argument == "--width" ||
        argument == "--height" || argument == "--pressure-iterations" || argument == "--dt" ||
        argument == "--output") {
      if (index + 1 >= argc) {
        throw std::invalid_argument(argument + " requires a value");
      }
      const std::string value = argv[++index];
      if (argument == "--frames") {
        options.runtime.frameLimit = parsePositive(value, "--frames");
      } else if (argument == "--export-every") {
        options.runtime.exportEvery = parsePositive(value, "--export-every");
      } else if (argument == "--width") {
        options.runtime.width = parsePositive(value, "--width");
      } else if (argument == "--height") {
        options.runtime.height = parsePositive(value, "--height");
      } else if (argument == "--pressure-iterations") {
        options.runtime.pressureIterations = parsePositive(value, "--pressure-iterations");
      } else if (argument == "--dt") {
        options.runtime.fixedDeltaTime = parseDeltaTime(value);
      } else {
        options.runtime.outputDirectory = value;
      }
      continue;
    }
    throw std::invalid_argument("Unknown option: " + argument);
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const ReferenceOptions options = parseArguments(argc, argv);
    gpu_fluids::NativeReferenceRuntime runtime(options.runtime);
    runtime.run();
    runtime.writeReport(options.runtime.outputDirectory / "runtime-report.json");
    gpu_fluids::makeRuntimeManifest(options.runtime, runtime.report())
        .write(options.runtime.outputDirectory / "experiment-manifest.json");

    if (!options.runtime.quiet) {
      const auto& report = runtime.report();
      std::cout << "C++ deterministic stable-fluids reference\n"
                << "  grid: " << options.runtime.width << " x " << options.runtime.height << "\n"
                << "  frames: " << report.framesSimulated << "\n"
                << "  exports: " << report.framesExported << "\n"
                << "  average frame: " << std::fixed << std::setprecision(3)
                << report.telemetry.averageFrameMilliseconds << " ms\n"
                << "  p95 frame: " << report.telemetry.p95FrameMilliseconds << " ms\n"
                << "  maximum speed: " << report.maximumObservedSpeed << "\n"
                << "  report: " << (options.runtime.outputDirectory / "runtime-report.json").string() << "\n"
                << "  manifest: " << (options.runtime.outputDirectory / "experiment-manifest.json").string() << "\n";
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fluid_reference_demo: " << error.what() << "\n";
    return 1;
  }
}
