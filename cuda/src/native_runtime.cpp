#include "gpu_fluids/native_runtime.hpp"

#include "gpu_fluids/visualization.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace gpu_fluids {
namespace {

constexpr float kMinimumRuntimeDeltaTime = 1.0e-4F;
constexpr float kMaximumRuntimeDeltaTime = 0.05F;

std::string jsonPath(const std::filesystem::path& path) {
  std::string value = path.generic_string();
  std::string escaped;
  escaped.reserve(value.size() + 8);
  for (const char character : value) {
    if (character == '\\' || character == '"') {
      escaped.push_back('\\');
    }
    escaped.push_back(character);
  }
  return escaped;
}

}  // namespace

NativeReferenceRuntime::NativeReferenceRuntime(RuntimeConfig config)
    : config_(std::move(config)), solver_(config_.width, config_.height), trajectory_(config_.strokeTrajectory) {
  validateConfig();
  solver_.setPressureIterations(config_.pressureIterations);
  report_.backend = "cpu-reference";
  if (config_.exportFrames || config_.writeTelemetry) {
    std::filesystem::create_directories(config_.outputDirectory);
  }
}

void NativeReferenceRuntime::validateConfig() const {
  if (config_.width < 4 || config_.height < 4 || config_.width > 2048 || config_.height > 2048) {
    throw std::invalid_argument("runtime dimensions must be between 4 and 2048");
  }
  if (config_.frameLimit < 0) {
    throw std::invalid_argument("runtime frame limit cannot be negative");
  }
  if (config_.exportEvery <= 0) {
    throw std::invalid_argument("runtime export interval must be positive");
  }
  if (config_.pressureIterations < 1 || config_.pressureIterations > 256) {
    throw std::invalid_argument("runtime pressure iterations must be between 1 and 256");
  }
  if (!std::isfinite(config_.fixedDeltaTime) || config_.fixedDeltaTime < kMinimumRuntimeDeltaTime ||
      config_.fixedDeltaTime > kMaximumRuntimeDeltaTime) {
    throw std::invalid_argument("runtime fixed delta time is outside the stable range");
  }
}

void NativeReferenceRuntime::enqueue(RuntimeCommandType type) {
  commands_.push_back({type, nextCommandSequence_++});
}

void NativeReferenceRuntime::applyCommands() {
  while (!commands_.empty()) {
    const RuntimeCommand command = commands_.front();
    commands_.pop_front();
    ++report_.commandsApplied;
    switch (command.type) {
      case RuntimeCommandType::Pause:
        paused_ = true;
        report_.pausedByCommand = true;
        break;
      case RuntimeCommandType::Resume:
        paused_ = false;
        report_.pausedByCommand = false;
        break;
      case RuntimeCommandType::SingleStep:
        paused_ = true;
        singleStepRequested_ = true;
        report_.pausedByCommand = true;
        break;
      case RuntimeCommandType::Reset:
        reset();
        break;
      case RuntimeCommandType::Stop:
        stopped_ = true;
        report_.stoppedByCommand = true;
        break;
    }
  }
}

void NativeReferenceRuntime::run() {
  stopped_ = false;
  while (!finished()) {
    applyCommands();
    if (stopped_) {
      break;
    }
    if (paused_ && !singleStepRequested_) {
      report_.pausedByCommand = true;
      break;
    }
    simulateFrame();
    if (singleStepRequested_) {
      singleStepRequested_ = false;
      paused_ = true;
      report_.pausedByCommand = true;
    }
  }
  report_.telemetry = journal_.summarize();
  writeTelemetryArtifacts();
}

void NativeReferenceRuntime::stepOnce() {
  if (stopped_ || finished()) {
    return;
  }
  applyCommands();
  if (!stopped_) {
    simulateFrame();
  }
  report_.telemetry = journal_.summarize();
}

void NativeReferenceRuntime::reset() {
  const std::uint64_t commandCount = report_.commandsApplied;
  solver_.reset();
  solver_.setPressureIterations(config_.pressureIterations);
  journal_.clear();
  trace_.clear();
  frameBuffer_.clear();
  currentFrame_ = 0;
  paused_ = false;
  stopped_ = false;
  singleStepRequested_ = false;
  report_ = {};
  report_.backend = "cpu-reference";
  report_.commandsApplied = commandCount;
}

bool NativeReferenceRuntime::finished() const noexcept {
  return stopped_ || currentFrame_ >= static_cast<std::uint64_t>(config_.frameLimit);
}

void NativeReferenceRuntime::simulateFrame() {
  const std::uint64_t frame = currentFrame_ + 1;
  trace_.beginFrame(frame);
  trace_.begin("simulation", "native-runtime");
  collector_.beginFrame(frame);
  solver_.step(makeParameters(frame), &collector_);
  {
    auto renderStage = collector_.stage(StageId::Render);
    solver_.downloadFrame(frameBuffer_);
  }
  const ValidationReport frameValidation = validateRgbaFrame(frameBuffer_, solver_.width(), solver_.height());
  const ValidationReport diagnosticValidation = validateSimulationDiagnostics(
      solver_.maxSpeed(), solver_.dyeEnergy(), static_cast<std::size_t>(solver_.width()) * solver_.height());
  report_.validationChecks += static_cast<std::uint64_t>(frameValidation.checkCount() + diagnosticValidation.checkCount());
  report_.validationFailures += static_cast<std::uint64_t>(frameValidation.errorCount() + diagnosticValidation.errorCount());
  if (!frameValidation.valid() || !diagnosticValidation.valid()) {
    throw std::runtime_error("native reference validation failed: " + frameValidation.summary() + "; " + diagnosticValidation.summary());
  }
  trace_.end("simulation", "native-runtime");
  exportFrameIfNeeded();
  trace_.instant("presentation-boundary", "native-runtime");
  trace_.endFrame();
  collector_.finishFrame(solver_.maxSpeed(), solver_.dyeEnergy());
  journal_.append(collector_.currentFrame());
  currentFrame_ = frame;
  report_.framesSimulated = currentFrame_;
  report_.framesExported = static_cast<std::uint64_t>(report_.exports.size());
  report_.maximumObservedSpeed = std::max(report_.maximumObservedSpeed, solver_.maxSpeed());
  report_.finalDyeEnergy = solver_.dyeEnergy();
  report_.telemetry = journal_.summarize();
}

void NativeReferenceRuntime::exportFrameIfNeeded() {
  if (!config_.exportFrames || currentFrame_ + 1 == 0) {
    return;
  }
  const std::uint64_t frame = currentFrame_ + 1;
  if (frame % static_cast<std::uint64_t>(config_.exportEvery) != 0 &&
      frame != static_cast<std::uint64_t>(config_.frameLimit)) {
    return;
  }

  auto transferStage = collector_.stage(StageId::Transfer);
  const std::filesystem::path path = framePath(frame);
  writePpm(path, solver_.width(), solver_.height(), frameBuffer_.data());
  report_.exports.push_back({frame, path, frameBuffer_.size()});
}

SimulationParams NativeReferenceRuntime::makeParameters(std::uint64_t frame) const {
  const float time = static_cast<float>(frame - 1) * config_.fixedDeltaTime;
  const float nextTime = time + config_.fixedDeltaTime;
  const StrokeSegment stroke = trajectory_.sample(
      time, config_.fixedDeltaTime, static_cast<float>(config_.width), static_cast<float>(config_.height));
  SimulationParams params;
  params.time = nextTime;
  params.deltaTime = config_.fixedDeltaTime;
  params.gridWidth = static_cast<float>(config_.width);
  params.gridHeight = static_cast<float>(config_.height);
  params.pointerActive = 1.0F;
  params.brushRadius = std::max(3.0F, static_cast<float>(config_.width) * 0.04F);
  params.strokeStartX = stroke.startX;
  params.strokeStartY = stroke.startY;
  params.strokeEndX = stroke.endX;
  params.strokeEndY = stroke.endY;
  params.injectedVelocityX = stroke.velocityX;
  params.injectedVelocityY = stroke.velocityY;
  params.velocityForce = 0.9F;
  params.inkAmount = 1.7F;
  params.inkColorR = 0.22F + 0.35F * (0.5F + 0.5F * std::sin(time * 0.7F));
  params.inkColorG = 0.52F + 0.18F * (0.5F + 0.5F * std::cos(time * 0.4F));
  params.inkColorB = 1.0F;
  params.exposure = 1.15F;
  params.vorticityConfinement = 0.35F;
  return params;
}

std::filesystem::path NativeReferenceRuntime::framePath(std::uint64_t frame) const {
  std::ostringstream filename;
  filename << "reference-frame-" << std::setw(6) << std::setfill('0') << frame << ".ppm";
  return config_.outputDirectory / filename.str();
}

std::string NativeReferenceRuntime::commandName(RuntimeCommandType type) {
  switch (type) {
    case RuntimeCommandType::Pause:
      return "pause";
    case RuntimeCommandType::Resume:
      return "resume";
    case RuntimeCommandType::SingleStep:
      return "single-step";
    case RuntimeCommandType::Reset:
      return "reset";
    case RuntimeCommandType::Stop:
      return "stop";
  }
  return "unknown";
}

void NativeReferenceRuntime::writeTelemetryArtifacts() const {
  if (!config_.writeTelemetry) {
    return;
  }
  journal_.writeJson(config_.outputDirectory / "telemetry.json");
  journal_.writeCsv(config_.outputDirectory / "telemetry.csv");
  trace_.writeChromeTrace(config_.outputDirectory / "trace.json");
}

void NativeReferenceRuntime::writeReport(const std::filesystem::path& path) const {
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("Could not open runtime report: " + path.string());
  }
  const RuntimeReport& report = report_;
  output << std::fixed << std::setprecision(4);
  output << "{\n"
         << "  \"backend\": \"" << report.backend << "\",\n"
         << "  \"grid\": {\"width\": " << config_.width << ", \"height\": " << config_.height << "},\n"
         << "  \"framesSimulated\": " << report.framesSimulated << ",\n"
         << "  \"framesExported\": " << report.framesExported << ",\n"
         << "  \"commandsApplied\": " << report.commandsApplied << ",\n"
         << "  \"stoppedByCommand\": " << (report.stoppedByCommand ? "true" : "false") << ",\n"
         << "  \"pausedByCommand\": " << (report.pausedByCommand ? "true" : "false") << ",\n"
         << "  \"maximumObservedSpeed\": " << report.maximumObservedSpeed << ",\n"
         << "  \"finalDyeEnergy\": " << report.finalDyeEnergy << ",\n"
         << "  \"validationChecks\": " << report.validationChecks << ",\n"
         << "  \"validationFailures\": " << report.validationFailures << ",\n"
         << "  \"averageFrameMilliseconds\": " << report.telemetry.averageFrameMilliseconds << ",\n"
         << "  \"p95FrameMilliseconds\": " << report.telemetry.p95FrameMilliseconds << ",\n"
         << "  \"exports\": [\n";
  for (std::size_t index = 0; index < report.exports.size(); ++index) {
    const ExportedFrame& frame = report.exports[index];
    output << "    {\"frame\": " << frame.frameIndex << ", \"path\": \""
            << jsonPath(frame.path) << "\", \"bytes\": " << frame.byteCount << "}";
    output << (index + 1 == report.exports.size() ? "\n" : ",\n");
  }
  output << "  ],\n  \"commands\": [";
  bool firstCommand = true;
  for (const RuntimeCommand& command : commands_) {
    if (!firstCommand) {
      output << ", ";
    }
    firstCommand = false;
    output << "{\"sequence\": " << command.sequence << ", \"type\": \""
            << commandName(command.type) << "\"}";
  }
  output << "]\n}\n";
  if (!output) {
    throw std::runtime_error("Could not write runtime report: " + path.string());
  }
}

void NativeReferenceRuntime::writeTrace(const std::filesystem::path& path) const {
  trace_.writeChromeTrace(path);
}

}  // namespace gpu_fluids
