#include "gpu_fluids/config.hpp"
#include "gpu_fluids/solver.hpp"
#include "gpu_fluids/sph_solver.hpp"
#include "gpu_fluids/visualization.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct CliOptions {
  std::string mode = "stable";
  int frames = 120;
  int exportEvery = 30;
  int pressureIterations = gpu_fluids::kPressureIterations;
  int deviceIndex = 0;
  bool enableVorticity = true;
  bool quiet = false;
  std::filesystem::path outputDirectory = "artifacts/native-frames";
};

void printUsage() {
  std::cout
      << "fluid_cuda_demo [options]\n"
      << "  --mode stable|sph          Select Eulerian grid or particle SPH simulation\n"
      << "  --frames N                 Number of GPU frames to run (default: 120)\n"
      << "  --export-every N           Export every Nth frame as PPM (default: 30)\n"
      << "  --pressure-iterations N    Jacobi iterations per frame (default: 20)\n"
      << "  --device N                 CUDA device index (default: 0)\n"
      << "  --no-vorticity             Disable curl/confinement stages\n"
      << "  --output DIR               PPM output directory\n"
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

int parseNonNegative(const std::string& value, const char* option) {
  std::size_t consumed = 0;
  const int parsed = std::stoi(value, &consumed);
  if (consumed != value.size() || parsed < 0) {
    throw std::invalid_argument(std::string(option) + " expects a non-negative integer");
  }
  return parsed;
}

CliOptions parseArguments(int argc, char** argv) {
  CliOptions options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--help") {
      printUsage();
      std::exit(0);
    }
    if (argument == "--no-vorticity") {
      options.enableVorticity = false;
      continue;
    }
    if (argument == "--quiet") {
      options.quiet = true;
      continue;
    }
    if (argument == "--mode" || argument == "--frames" || argument == "--export-every" || argument == "--pressure-iterations" || argument == "--device" || argument == "--output") {
      if (index + 1 >= argc) {
        throw std::invalid_argument(argument + " requires a value");
      }
      const std::string value = argv[++index];
      if (argument == "--mode") {
        options.mode = value;
      } else if (argument == "--frames") {
        options.frames = parsePositive(value, "--frames");
      } else if (argument == "--export-every") {
        options.exportEvery = parsePositive(value, "--export-every");
      } else if (argument == "--pressure-iterations") {
        options.pressureIterations = parsePositive(value, "--pressure-iterations");
      } else if (argument == "--device") {
        options.deviceIndex = parseNonNegative(value, "--device");
      } else {
        options.outputDirectory = value;
      }
      continue;
    }
    throw std::invalid_argument("Unknown option: " + argument);
  }
  if (options.mode != "stable" && options.mode != "sph") {
    throw std::invalid_argument("--mode must be stable or sph");
  }
  return options;
}

gpu_fluids::SimulationParams makeDemoParams(int frame, bool enableVorticity) {
  using namespace gpu_fluids;
  const float time = static_cast<float>(frame) / 60.0F;
  const float nextTime = time + 1.0F / 60.0F;
  SimulationParams params;
  params.time = nextTime;
  params.deltaTime = 1.0F / 60.0F;
  params.pointerActive = 1.0F;
  params.brushRadius = 20.0F;
  params.strokeStartX = 256.0F + std::cos(time * 1.7F) * 150.0F;
  params.strokeStartY = 256.0F + std::sin(time * 2.1F) * 120.0F;
  params.strokeEndX = 256.0F + std::cos(nextTime * 1.7F) * 150.0F;
  params.strokeEndY = 256.0F + std::sin(nextTime * 2.1F) * 120.0F;
  params.injectedVelocityX = (params.strokeEndX - params.strokeStartX) / params.deltaTime;
  params.injectedVelocityY = (params.strokeEndY - params.strokeStartY) / params.deltaTime;
  params.velocityForce = 1.2F;
  params.inkAmount = 1.45F;
  params.inkColorR = 0.25F + 0.35F * (0.5F + 0.5F * std::sin(time * 0.7F));
  params.inkColorG = 0.55F;
  params.inkColorB = 1.0F;
  params.exposure = 1.15F;
  params.vorticityConfinement = enableVorticity ? 0.75F : 0.0F;
  return params;
}

gpu_fluids::SphParams makeSphParams(int frame) {
  gpu_fluids::SphParams params;
  params.deltaTime = 1.0F / 120.0F;
  params.pressureStiffness = 2.6F;
  params.viscosity = 0.18F;
  params.gravityY = 140.0F;
  params.colorPhase = static_cast<float>(frame) * 0.025F;
  return params;
}

std::string formatMilliseconds(float milliseconds) {
  std::ostringstream value;
  value << std::fixed << std::setprecision(3) << milliseconds << " ms";
  return value.str();
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const CliOptions options = parseArguments(argc, argv);
    std::filesystem::create_directories(options.outputDirectory);

    if (options.mode == "sph") {
      gpu_fluids::SphSolver solver({}, options.deviceIndex);
      const auto& metrics = solver.deviceMetrics();
      if (!options.quiet) {
        std::cout << "CUDA SPH fluids\n"
                  << "  device: " << metrics.name << " (sm " << metrics.computeCapabilityMajor << "."
                  << metrics.computeCapabilityMinor << ")\n"
                  << "  particles: " << gpu_fluids::kSphParticleCount
                  << ", uniform grid: " << gpu_fluids::kSphGridWidth << " x " << gpu_fluids::kSphGridHeight << "\n"
                  << "  persistent device memory: " << solver.lastFrameStats().persistentDeviceBytes << " bytes\n";
      }
      std::vector<std::uint8_t> rgba(static_cast<std::size_t>(solver.width()) * solver.height() * 4);
      for (int frame = 0; frame < options.frames; ++frame) {
        solver.step(makeSphParams(frame));
        if ((frame + 1) % options.exportEvery == 0 || frame + 1 == options.frames) {
          solver.downloadFrame(rgba);
          std::ostringstream filename;
          filename << "sph-frame-" << std::setw(5) << std::setfill('0') << (frame + 1) << ".ppm";
          gpu_fluids::writePpm(options.outputDirectory / filename.str(), solver.width(), solver.height(), rgba.data());
          if (!options.quiet) {
            const auto& stats = solver.lastFrameStats();
            std::cout << "  frame " << stats.frameIndex << ": " << formatMilliseconds(stats.gpuMilliseconds)
                      << ", neighbor overflow " << stats.neighborOverflow
                      << ", exported " << filename.str() << "\n";
          }
        }
      }
      return 0;
    }

    gpu_fluids::SolverConfig config;
    config.pressureIterations = options.pressureIterations;
    config.deviceIndex = options.deviceIndex;
    config.enableVorticity = options.enableVorticity;
    gpu_fluids::StableFluidSolver solver(config);
    const auto& metrics = solver.deviceMetrics();

    if (!options.quiet) {
      std::cout << "CUDA stable fluids\n"
                << "  device: " << metrics.name << " (sm " << metrics.computeCapabilityMajor << "."
                << metrics.computeCapabilityMinor << ")\n"
                << "  grid: " << solver.width() << " x " << solver.height()
                << ", block: " << gpu_fluids::kBlockSize << " x " << gpu_fluids::kBlockSize << "\n"
                << "  persistent device memory: " << solver.lastFrameStats().persistentDeviceBytes << " bytes\n"
                << "  divergence registers/thread: " << metrics.divergenceRegistersPerThread << "\n";
    }

    std::vector<std::uint8_t> rgba(static_cast<std::size_t>(solver.width()) * solver.height() * 4);
    for (int frame = 0; frame < options.frames; ++frame) {
      solver.step(makeDemoParams(frame, options.enableVorticity));
      if ((frame + 1) % options.exportEvery == 0 || frame + 1 == options.frames) {
        solver.downloadFrame(rgba);
        std::ostringstream filename;
        filename << "frame-" << std::setw(5) << std::setfill('0') << (frame + 1) << ".ppm";
        gpu_fluids::writePpm(options.outputDirectory / filename.str(), solver.width(), solver.height(), rgba.data());
        if (!options.quiet) {
          const auto& stats = solver.lastFrameStats();
          std::cout << "  frame " << stats.frameIndex << ": " << formatMilliseconds(stats.gpuMilliseconds)
                    << ", exported " << filename.str() << "\n";
        }
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "fluid_cuda_demo: " << error.what() << "\n";
    return 1;
  }
}
