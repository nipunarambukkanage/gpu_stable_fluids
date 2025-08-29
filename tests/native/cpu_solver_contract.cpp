#include "gpu_fluids/cpu_solver.hpp"
#include "gpu_fluids/telemetry.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

bool finite(float value) {
  return std::isfinite(static_cast<double>(value));
}

gpu_fluids::SimulationParams makeInput(int width, int height) {
  gpu_fluids::SimulationParams params;
  params.gridWidth = static_cast<float>(width);
  params.gridHeight = static_cast<float>(height);
  params.pointerActive = 1.0F;
  params.brushRadius = 6.0F;
  params.strokeStartX = static_cast<float>(width) * 0.25F;
  params.strokeStartY = static_cast<float>(height) * 0.5F;
  params.strokeEndX = static_cast<float>(width) * 0.75F;
  params.strokeEndY = static_cast<float>(height) * 0.5F;
  params.injectedVelocityX = 90.0F;
  params.injectedVelocityY = -12.0F;
  params.inkAmount = 2.0F;
  params.inkColorR = 0.2F;
  params.inkColorG = 0.8F;
  params.inkColorB = 1.0F;
  params.vorticityConfinement = 0.4F;
  return params;
}

int fail(const char* message) {
  std::cerr << "native_reference_contract: " << message << '\n';
  return 1;
}

}  // namespace

int main() {
  constexpr int width = 32;
  constexpr int height = 24;
  gpu_fluids::CpuStableFluidSolver solver(width, height);
  solver.setPressureIterations(8);
  gpu_fluids::TelemetryCollector collector;
  gpu_fluids::TelemetryJournal journal;

  for (std::uint64_t frame = 1; frame <= 4; ++frame) {
    collector.beginFrame(frame);
    gpu_fluids::SimulationParams params = makeInput(width, height);
    params.time = static_cast<float>(frame) / 60.0F;
    solver.step(params, &collector);
    std::vector<std::uint8_t> pixels;
    {
      auto renderStage = collector.stage(gpu_fluids::StageId::Render);
      solver.downloadFrame(pixels);
    }
    collector.finishFrame(solver.maxSpeed(), solver.dyeEnergy());
    journal.append(collector.currentFrame());

    if (pixels.size() != static_cast<std::size_t>(width * height * 4)) {
      return fail("rendered frame has an unexpected byte count");
    }
    for (int pixel = 0; pixel < width * height; ++pixel) {
      if (pixels[static_cast<std::size_t>(pixel) * 4 + 3] != 255) {
        return fail("rendered frame alpha channel is not opaque");
      }
    }
    if (!finite(solver.maxSpeed()) || !finite(solver.dyeEnergy())) {
      return fail("solver produced a non-finite diagnostic");
    }
  }

  const gpu_fluids::TelemetrySummary summary = journal.summarize();
  if (summary.frames != 4 || summary.totalMilliseconds <= 0.0 ||
      summary.stages[static_cast<std::size_t>(gpu_fluids::StageId::Pressure)].samples != 4) {
    return fail("telemetry summary did not record all frames and pressure stages");
  }
  if (journal.percentileFrame(95.0) < journal.percentileFrame(0.0)) {
    return fail("frame percentile ordering is invalid");
  }
  if (std::string(gpu_fluids::stageName(gpu_fluids::StageId::Projection)) != "projection") {
    return fail("stage naming contract changed");
  }

  solver.reset();
  if (solver.lastFrameStats().frameIndex != 0 || solver.dyeEnergy() != 0.0F) {
    return fail("reset did not clear deterministic solver state");
  }
  std::cout << "native_reference_contract: passed\n";
  return 0;
}
