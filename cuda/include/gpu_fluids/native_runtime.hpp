#pragma once

#include "gpu_fluids/cpu_solver.hpp"
#include "gpu_fluids/frame_metrics.hpp"
#include "gpu_fluids/input_trajectory.hpp"
#include "gpu_fluids/numeric_validation.hpp"
#include "gpu_fluids/telemetry.hpp"
#include "gpu_fluids/trace_recorder.hpp"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <string>
#include <vector>

namespace gpu_fluids {

enum class RuntimeCommandType : std::uint8_t {
  Pause,
  Resume,
  SingleStep,
  Reset,
  Stop,
};

struct RuntimeCommand {
  RuntimeCommandType type = RuntimeCommandType::Resume;
  std::uint64_t sequence = 0;
};

struct RuntimeConfig {
  int width = 128;
  int height = 128;
  int frameLimit = 60;
  int exportEvery = 30;
  int pressureIterations = 20;
  float fixedDeltaTime = 1.0F / 60.0F;
  EllipticalStrokeConfig strokeTrajectory{};
  bool exportFrames = true;
  bool writeTelemetry = true;
  bool quiet = false;
  std::filesystem::path outputDirectory = "artifacts/native-reference";
};

struct ExportedFrame {
  std::uint64_t frameIndex = 0;
  std::filesystem::path path;
  std::size_t byteCount = 0;
};

struct RuntimeReport {
  std::string backend = "cpu-reference";
  std::uint64_t framesSimulated = 0;
  std::uint64_t framesExported = 0;
  std::uint64_t commandsApplied = 0;
  bool stoppedByCommand = false;
  bool pausedByCommand = false;
  float maximumObservedSpeed = 0.0F;
  float finalDyeEnergy = 0.0F;
  float finalFrameMeanLuminance = 0.0F;
  std::size_t finalFrameActivePixels = 0;
  std::size_t finalFrameOpaquePixels = 0;
  std::uint64_t validationChecks = 0;
  std::uint64_t validationFailures = 0;
  TelemetrySummary telemetry{};
  std::vector<ExportedFrame> exports;
};

// Production-style CPU orchestration for the deterministic host reference.
// The class models the control boundary used by the CUDA executable: commands
// are serialized, simulation state is persistent, telemetry is frame-scoped,
// and output artifacts are written only at explicit presentation boundaries.
class NativeReferenceRuntime final {
 public:
  explicit NativeReferenceRuntime(RuntimeConfig config = {});

  NativeReferenceRuntime(const NativeReferenceRuntime&) = delete;
  NativeReferenceRuntime& operator=(const NativeReferenceRuntime&) = delete;

  void enqueue(RuntimeCommandType type);
  void run();
  void stepOnce();
  void reset();

  [[nodiscard]] bool paused() const noexcept { return paused_; }
  [[nodiscard]] bool stopped() const noexcept { return stopped_; }
  [[nodiscard]] bool finished() const noexcept;
  [[nodiscard]] const RuntimeConfig& config() const noexcept { return config_; }
  [[nodiscard]] const RuntimeReport& report() const noexcept { return report_; }
  [[nodiscard]] const TelemetryJournal& telemetry() const noexcept { return journal_; }
  [[nodiscard]] const TraceRecorder& trace() const noexcept { return trace_; }

  void writeReport(const std::filesystem::path& path) const;
  void writeTrace(const std::filesystem::path& path) const;

 private:
  void validateConfig() const;
  void applyCommands();
  void simulateFrame();
  void exportFrameIfNeeded();
  void writeTelemetryArtifacts() const;
  [[nodiscard]] SimulationParams makeParameters(std::uint64_t frame) const;
  [[nodiscard]] std::filesystem::path framePath(std::uint64_t frame) const;
  [[nodiscard]] static std::string commandName(RuntimeCommandType type);

  RuntimeConfig config_{};
  CpuStableFluidSolver solver_;
  EllipticalStrokeTrajectory trajectory_;
  TelemetryCollector collector_;
  TelemetryJournal journal_;
  TraceRecorder trace_;
  std::vector<std::uint8_t> frameBuffer_;
  std::deque<RuntimeCommand> commands_;
  RuntimeReport report_{};
  std::uint64_t nextCommandSequence_ = 1;
  std::uint64_t currentFrame_ = 0;
  bool paused_ = false;
  bool stopped_ = false;
  bool singleStepRequested_ = false;
};

}  // namespace gpu_fluids
