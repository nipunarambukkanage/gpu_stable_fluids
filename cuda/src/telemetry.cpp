#include "gpu_fluids/telemetry.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace gpu_fluids {
namespace {

constexpr std::size_t stageIndex(StageId stage) noexcept {
  return static_cast<std::size_t>(stage);
}

double percentileSorted(const std::vector<double>& sortedValues, double percentile) {
  if (sortedValues.empty()) {
    return 0.0;
  }
  const double clamped = std::clamp(percentile, 0.0, 100.0) / 100.0;
  const double position = clamped * static_cast<double>(sortedValues.size() - 1);
  const auto lower = static_cast<std::size_t>(position);
  const auto upper = std::min(lower + 1, sortedValues.size() - 1);
  const double fraction = position - static_cast<double>(lower);
  return sortedValues[lower] + (sortedValues[upper] - sortedValues[lower]) * fraction;
}

std::string jsonEscape(const char* value) {
  std::string escaped;
  for (const char* cursor = value; cursor != nullptr && *cursor != '\0'; ++cursor) {
    switch (*cursor) {
      case '"':
        escaped += "\\\"";
        break;
      case '\\':
        escaped += "\\\\";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped += *cursor;
        break;
    }
  }
  return escaped;
}

}  // namespace

const char* stageName(StageId stage) noexcept {
  switch (stage) {
    case StageId::Input:
      return "input";
    case StageId::Advection:
      return "advection";
    case StageId::Divergence:
      return "divergence";
    case StageId::Pressure:
      return "pressure";
    case StageId::Projection:
      return "projection";
    case StageId::Render:
      return "render";
    case StageId::Transfer:
      return "transfer";
    case StageId::Count:
      break;
  }
  return "unknown";
}

ScopedStage::ScopedStage(TelemetryCollector* owner, StageId stage) noexcept
    : owner_(owner), stage_(stage) {}

ScopedStage::~ScopedStage() {
  if (owner_ != nullptr) {
    owner_->finishStage(stage_);
  }
}

ScopedStage::ScopedStage(ScopedStage&& other) noexcept
    : owner_(other.owner_), stage_(other.stage_) {
  other.owner_ = nullptr;
}

ScopedStage& ScopedStage::operator=(ScopedStage&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (owner_ != nullptr) {
    owner_->finishStage(stage_);
  }
  owner_ = other.owner_;
  stage_ = other.stage_;
  other.owner_ = nullptr;
  return *this;
}

void TelemetryCollector::beginFrame(std::uint64_t frameIndex) noexcept {
  currentFrame_ = {};
  currentFrame_.frameIndex = frameIndex;
  currentFrame_.stages.reserve(stageIndex(StageId::Count));
  stageActive_.fill(false);
  frameStart_ = Clock::now();
  frameActive_ = true;
}

void TelemetryCollector::finishFrame(float maxSpeed, float dyeEnergy) noexcept {
  if (!frameActive_) {
    return;
  }
  for (std::size_t index = 0; index < stageActive_.size(); ++index) {
    if (stageActive_[index]) {
      finishStage(static_cast<StageId>(index));
    }
  }
  const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - frameStart_);
  currentFrame_.totalMilliseconds = elapsed.count();
  currentFrame_.maxSpeed = maxSpeed;
  currentFrame_.dyeEnergy = dyeEnergy;
  frameActive_ = false;
}

ScopedStage TelemetryCollector::stage(StageId stageValue) noexcept {
  const auto index = stageIndex(stageValue);
  if (stageValue == StageId::Count || !frameActive_ || stageActive_[index]) {
    return {};
  }
  stageStarts_[index] = Clock::now();
  stageActive_[index] = true;
  return ScopedStage(this, stageValue);
}

void TelemetryCollector::record(StageId stageValue, double milliseconds) {
  if (stageValue == StageId::Count || milliseconds < 0.0) {
    throw std::invalid_argument("telemetry stage samples must be non-negative and named");
  }
  currentFrame_.stages.push_back({stageValue, milliseconds});
}

void TelemetryCollector::finishStage(StageId stageValue) noexcept {
  const auto index = stageIndex(stageValue);
  if (!frameActive_ || stageValue == StageId::Count || !stageActive_[index]) {
    return;
  }
  const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - stageStarts_[index]);
  stageActive_[index] = false;
  currentFrame_.stages.push_back({stageValue, elapsed.count()});
}

void TelemetryJournal::append(FrameTelemetry frame) {
  if (frame.totalMilliseconds < 0.0) {
    throw std::invalid_argument("frame telemetry duration must be non-negative");
  }
  frames_.push_back(std::move(frame));
}

void TelemetryJournal::clear() noexcept {
  frames_.clear();
}

std::vector<double> TelemetryJournal::frameDurations() const {
  std::vector<double> values;
  values.reserve(frames_.size());
  for (const FrameTelemetry& frame : frames_) {
    values.push_back(frame.totalMilliseconds);
  }
  return values;
}

std::vector<double> TelemetryJournal::stageDurations(StageId stageValue) const {
  std::vector<double> values;
  for (const FrameTelemetry& frame : frames_) {
    for (const StageTiming& stage : frame.stages) {
      if (stage.stage == stageValue) {
        values.push_back(stage.milliseconds);
      }
    }
  }
  return values;
}

double TelemetryJournal::percentileFrame(double percentile) const {
  std::vector<double> values = frameDurations();
  std::sort(values.begin(), values.end());
  return percentileSorted(values, percentile);
}

double TelemetryJournal::percentileStage(StageId stageValue, double percentile) const {
  std::vector<double> values = stageDurations(stageValue);
  std::sort(values.begin(), values.end());
  return percentileSorted(values, percentile);
}

TelemetrySummary TelemetryJournal::summarize() const {
  TelemetrySummary summary;
  summary.frames = static_cast<std::uint64_t>(frames_.size());
  if (frames_.empty()) {
    return summary;
  }

  const std::vector<double> frameValues = frameDurations();
  const auto frameBounds = std::minmax_element(frameValues.begin(), frameValues.end());
  summary.totalMilliseconds = std::accumulate(frameValues.begin(), frameValues.end(), 0.0);
  summary.averageFrameMilliseconds = summary.totalMilliseconds / static_cast<double>(summary.frames);
  summary.minimumFrameMilliseconds = *frameBounds.first;
  summary.maximumFrameMilliseconds = *frameBounds.second;
  summary.p95FrameMilliseconds = percentileFrame(95.0);

  for (std::size_t index = 0; index < stageIndex(StageId::Count); ++index) {
    const StageId stageValue = static_cast<StageId>(index);
    std::vector<double> values = stageDurations(stageValue);
    if (values.empty()) {
      continue;
    }
    auto& aggregate = summary.stages[index];
    aggregate.samples = static_cast<std::uint64_t>(values.size());
    aggregate.totalMilliseconds = std::accumulate(values.begin(), values.end(), 0.0);
    const auto bounds = std::minmax_element(values.begin(), values.end());
    aggregate.minimumMilliseconds = *bounds.first;
    aggregate.maximumMilliseconds = *bounds.second;
    aggregate.averageMilliseconds = aggregate.totalMilliseconds / static_cast<double>(aggregate.samples);
    std::sort(values.begin(), values.end());
    aggregate.p95Milliseconds = percentileSorted(values, 95.0);
  }
  return summary;
}

void TelemetryJournal::writeJson(const std::filesystem::path& path) const {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Could not open telemetry JSON: " + path.string());
  }

  const TelemetrySummary summary = summarize();
  output << std::fixed << std::setprecision(4);
  output << "{\n  \"frames\": " << summary.frames << ",\n";
  output << "  \"totalMilliseconds\": " << summary.totalMilliseconds << ",\n";
  output << "  \"averageFrameMilliseconds\": " << summary.averageFrameMilliseconds << ",\n";
  output << "  \"p95FrameMilliseconds\": " << summary.p95FrameMilliseconds << ",\n";
  output << "  \"stages\": {\n";
  for (std::size_t index = 0; index < stageIndex(StageId::Count); ++index) {
    const auto& aggregate = summary.stages[index];
    output << "    \"" << jsonEscape(stageName(static_cast<StageId>(index))) << "\": {\"samples\": "
            << aggregate.samples << ", \"averageMilliseconds\": " << aggregate.averageMilliseconds
            << ", \"p95Milliseconds\": " << aggregate.p95Milliseconds << "}";
    output << (index + 1 == stageIndex(StageId::Count) ? "\n" : ",\n");
  }
  output << "  },\n  \"framesDetail\": [\n";
  for (std::size_t frameIndex = 0; frameIndex < frames_.size(); ++frameIndex) {
    const FrameTelemetry& frame = frames_[frameIndex];
    output << "    {\"frame\": " << frame.frameIndex << ", \"milliseconds\": "
            << frame.totalMilliseconds << ", \"maxSpeed\": " << frame.maxSpeed
            << ", \"dyeEnergy\": " << frame.dyeEnergy << ", \"stages\": {";
    for (std::size_t stageIndexValue = 0; stageIndexValue < frame.stages.size(); ++stageIndexValue) {
      const StageTiming& stage = frame.stages[stageIndexValue];
      output << "\"" << jsonEscape(stageName(stage.stage)) << "\": " << stage.milliseconds;
      if (stageIndexValue + 1 != frame.stages.size()) {
        output << ", ";
      }
    }
    output << "}}" << (frameIndex + 1 == frames_.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  if (!output) {
    throw std::runtime_error("Could not write telemetry JSON: " + path.string());
  }
}

void TelemetryJournal::writeCsv(const std::filesystem::path& path) const {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Could not open telemetry CSV: " + path.string());
  }
  output << "frame,total_ms,max_speed,dye_energy";
  for (std::size_t index = 0; index < stageIndex(StageId::Count); ++index) {
    output << ',' << stageName(static_cast<StageId>(index)) << "_ms";
  }
  output << '\n' << std::fixed << std::setprecision(4);
  for (const FrameTelemetry& frame : frames_) {
    output << frame.frameIndex << ',' << frame.totalMilliseconds << ',' << frame.maxSpeed << ',' << frame.dyeEnergy;
    std::array<double, stageIndex(StageId::Count)> stageValues{};
    for (const StageTiming& stage : frame.stages) {
      stageValues[stageIndex(stage.stage)] += stage.milliseconds;
    }
    for (double value : stageValues) {
      output << ',' << value;
    }
    output << '\n';
  }
  if (!output) {
    throw std::runtime_error("Could not write telemetry CSV: " + path.string());
  }
}

}  // namespace gpu_fluids
