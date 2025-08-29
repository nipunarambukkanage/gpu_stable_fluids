#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace gpu_fluids {

enum class StageId : std::uint8_t {
  Input = 0,
  Advection,
  Divergence,
  Pressure,
  Projection,
  Render,
  Transfer,
  Count,
};

[[nodiscard]] const char* stageName(StageId stage) noexcept;

struct StageTiming {
  StageId stage = StageId::Input;
  double milliseconds = 0.0;
};

struct FrameTelemetry {
  std::uint64_t frameIndex = 0;
  double totalMilliseconds = 0.0;
  float maxSpeed = 0.0F;
  float dyeEnergy = 0.0F;
  std::vector<StageTiming> stages;
};

class TelemetryCollector;

class ScopedStage final {
 public:
  ScopedStage() = default;
  ScopedStage(TelemetryCollector* owner, StageId stage) noexcept;
  ~ScopedStage();

  ScopedStage(const ScopedStage&) = delete;
  ScopedStage& operator=(const ScopedStage&) = delete;
  ScopedStage(ScopedStage&& other) noexcept;
  ScopedStage& operator=(ScopedStage&& other) noexcept;

 private:
  TelemetryCollector* owner_ = nullptr;
  StageId stage_ = StageId::Input;
};

class TelemetryCollector final {
 public:
  using Clock = std::chrono::steady_clock;

  void beginFrame(std::uint64_t frameIndex) noexcept;
  void finishFrame(float maxSpeed, float dyeEnergy) noexcept;
  [[nodiscard]] ScopedStage stage(StageId stage) noexcept;
  void record(StageId stage, double milliseconds);

  [[nodiscard]] const FrameTelemetry& currentFrame() const noexcept { return currentFrame_; }
  [[nodiscard]] bool frameActive() const noexcept { return frameActive_; }

 private:
  friend class ScopedStage;
  void finishStage(StageId stage) noexcept;

  FrameTelemetry currentFrame_{};
  std::array<Clock::time_point, static_cast<std::size_t>(StageId::Count)> stageStarts_{};
  std::array<bool, static_cast<std::size_t>(StageId::Count)> stageActive_{};
  Clock::time_point frameStart_{};
  bool frameActive_ = false;
};

struct StageAggregate {
  std::uint64_t samples = 0;
  double totalMilliseconds = 0.0;
  double minimumMilliseconds = 0.0;
  double maximumMilliseconds = 0.0;
  double averageMilliseconds = 0.0;
  double p95Milliseconds = 0.0;
};

struct TelemetrySummary {
  std::uint64_t frames = 0;
  double totalMilliseconds = 0.0;
  double averageFrameMilliseconds = 0.0;
  double p95FrameMilliseconds = 0.0;
  double minimumFrameMilliseconds = 0.0;
  double maximumFrameMilliseconds = 0.0;
  std::array<StageAggregate, static_cast<std::size_t>(StageId::Count)> stages{};
};

class TelemetryJournal final {
 public:
  void append(FrameTelemetry frame);
  void clear() noexcept;

  [[nodiscard]] const std::vector<FrameTelemetry>& frames() const noexcept { return frames_; }
  [[nodiscard]] TelemetrySummary summarize() const;
  [[nodiscard]] double percentileFrame(double percentile) const;
  [[nodiscard]] double percentileStage(StageId stage, double percentile) const;

  void writeJson(const std::filesystem::path& path) const;
  void writeCsv(const std::filesystem::path& path) const;

 private:
  std::vector<double> frameDurations() const;
  std::vector<double> stageDurations(StageId stage) const;

  std::vector<FrameTelemetry> frames_;
};

}  // namespace gpu_fluids
