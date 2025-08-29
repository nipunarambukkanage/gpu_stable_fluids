#include "gpu_fluids/cpu_solver.hpp"
#include "gpu_fluids/telemetry.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace gpu_fluids {
namespace {

constexpr float kMinimumDeltaTime = 1.0e-4F;
constexpr float kMaximumDeltaTime = 0.05F;
constexpr float kPressureScale = 0.5F;
constexpr float kDissipationFloor = 0.0F;

float saturate(float value) {
  return std::clamp(value, 0.0F, 1.0F);
}

std::uint8_t toByte(float value) {
  return static_cast<std::uint8_t>(std::lround(saturate(value) * 255.0F));
}

float mix(float first, float second, float amount) {
  return first + (second - first) * amount;
}

void swapFields(std::vector<float>& first, std::vector<float>& second) {
  first.swap(second);
}

}  // namespace

CpuStableFluidSolver::CpuStableFluidSolver(int width, int height)
    : width_(width), height_(height), cellCount_(static_cast<std::size_t>(width) * height) {
  if (width_ < 4 || height_ < 4) {
    throw std::invalid_argument("CPU reference solver requires a grid of at least 4 x 4 cells");
  }

  const auto allocate = [this](Field& field) { field.assign(cellCount_, 0.0F); };
  allocate(velocityXRead_);
  allocate(velocityXWrite_);
  allocate(velocityYRead_);
  allocate(velocityYWrite_);
  allocate(densityRRead_);
  allocate(densityRWrite_);
  allocate(densityGRead_);
  allocate(densityGWrite_);
  allocate(densityBRead_);
  allocate(densityBWrite_);
  allocate(pressureRead_);
  allocate(pressureWrite_);
  allocate(divergence_);
  reset();
}

void CpuStableFluidSolver::reset() {
  const auto clear = [](Field& field) { std::fill(field.begin(), field.end(), 0.0F); };
  clear(velocityXRead_);
  clear(velocityXWrite_);
  clear(velocityYRead_);
  clear(velocityYWrite_);
  clear(densityRRead_);
  clear(densityRWrite_);
  clear(densityGRead_);
  clear(densityGWrite_);
  clear(densityBRead_);
  clear(densityBWrite_);
  clear(pressureRead_);
  clear(pressureWrite_);
  clear(divergence_);
  params_ = {};
  frameStats_ = {};
  frameStats_.pressureIterations = pressureIterations_;
  frameIndex_ = 0;
  maxSpeed_ = 0.0F;
  dyeEnergy_ = 0.0F;
}

void CpuStableFluidSolver::setPressureIterations(int iterations) noexcept {
  pressureIterations_ = std::clamp(iterations, 1, 256);
  frameStats_.pressureIterations = pressureIterations_;
}

std::size_t CpuStableFluidSolver::index(int x, int y) const noexcept {
  return static_cast<std::size_t>(y) * static_cast<std::size_t>(width_) +
         static_cast<std::size_t>(x);
}

float CpuStableFluidSolver::sample(const Field& field, float x, float y) const noexcept {
  const float clampedX = std::clamp(x, 0.0F, static_cast<float>(width_ - 1));
  const float clampedY = std::clamp(y, 0.0F, static_cast<float>(height_ - 1));
  const int x0 = std::clamp(static_cast<int>(std::floor(clampedX)), 0, width_ - 1);
  const int y0 = std::clamp(static_cast<int>(std::floor(clampedY)), 0, height_ - 1);
  const int x1 = std::min(x0 + 1, width_ - 1);
  const int y1 = std::min(y0 + 1, height_ - 1);
  const float tx = clampedX - static_cast<float>(x0);
  const float ty = clampedY - static_cast<float>(y0);
  const float top = mix(field[index(x0, y0)], field[index(x1, y0)], tx);
  const float bottom = mix(field[index(x0, y1)], field[index(x1, y1)], tx);
  return mix(top, bottom, ty);
}

float CpuStableFluidSolver::sampleVelocityX(float x, float y) const noexcept {
  return sample(velocityXRead_, x, y);
}

float CpuStableFluidSolver::sampleVelocityY(float x, float y) const noexcept {
  return sample(velocityYRead_, x, y);
}

float CpuStableFluidSolver::segmentDistanceSquared(float px, float py, float ax, float ay,
                                                    float bx, float by) const noexcept {
  const float abx = bx - ax;
  const float aby = by - ay;
  const float lengthSquared = abx * abx + aby * aby;
  if (lengthSquared <= 1.0e-6F) {
    const float dx = px - ax;
    const float dy = py - ay;
    return dx * dx + dy * dy;
  }
  const float projection = std::clamp(((px - ax) * abx + (py - ay) * aby) / lengthSquared,
                                      0.0F, 1.0F);
  const float closestX = ax + projection * abx;
  const float closestY = ay + projection * aby;
  const float dx = px - closestX;
  const float dy = py - closestY;
  return dx * dx + dy * dy;
}

void CpuStableFluidSolver::injectStroke(const SimulationParams& params) {
  if (params.pointerActive <= 0.0F || params.inkAmount <= 0.0F) {
    return;
  }

  const float radius = std::max(1.0F, params.brushRadius *
                                        static_cast<float>(width_) / kGridWidth);
  const float radiusSquared = radius * radius;
  const float scaleX = static_cast<float>(width_) / std::max(params.gridWidth, 1.0F);
  const float scaleY = static_cast<float>(height_) / std::max(params.gridHeight, 1.0F);
  const float startX = params.strokeStartX * scaleX;
  const float startY = params.strokeStartY * scaleY;
  const float endX = params.strokeEndX * scaleX;
  const float endY = params.strokeEndY * scaleY;
  const float velocityScale = params.velocityForce * params.deltaTime * 0.08F;

  for (int y = 1; y < height_ - 1; ++y) {
    for (int x = 1; x < width_ - 1; ++x) {
      const float distanceSquared = segmentDistanceSquared(
          static_cast<float>(x), static_cast<float>(y), startX, startY, endX, endY);
      if (distanceSquared > radiusSquared * 4.0F) {
        continue;
      }
      const float falloff = std::exp(-distanceSquared / (radiusSquared * 0.72F));
      const std::size_t cell = index(x, y);
      velocityXRead_[cell] += params.injectedVelocityX * velocityScale * falloff;
      velocityYRead_[cell] += params.injectedVelocityY * velocityScale * falloff;
      densityRRead_[cell] = std::min(8.0F, densityRRead_[cell] + params.inkAmount * params.inkColorR * falloff);
      densityGRead_[cell] = std::min(8.0F, densityGRead_[cell] + params.inkAmount * params.inkColorG * falloff);
      densityBRead_[cell] = std::min(8.0F, densityBRead_[cell] + params.inkAmount * params.inkColorB * falloff);
    }
  }
}

void CpuStableFluidSolver::advectVelocity(float deltaTime) {
  const float maxBacktrace = kMaxBacktraceDistance *
                             static_cast<float>(width_) / static_cast<float>(kGridWidth);
  const float velocityDissipation = std::exp(-std::max(0.0F, params_.velocityDissipation) * deltaTime);

  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const std::size_t cell = index(x, y);
      const float velocityX = velocityXRead_[cell];
      const float velocityY = velocityYRead_[cell];
      float backtraceX = velocityX * deltaTime;
      float backtraceY = velocityY * deltaTime;
      const float backtraceLength = std::hypot(backtraceX, backtraceY);
      if (backtraceLength > maxBacktrace) {
        const float scale = maxBacktrace / backtraceLength;
        backtraceX *= scale;
        backtraceY *= scale;
      }
      const float sourceX = static_cast<float>(x) - backtraceX;
      const float sourceY = static_cast<float>(y) - backtraceY;
      velocityXWrite_[cell] = sample(velocityXRead_, sourceX, sourceY) * velocityDissipation;
      velocityYWrite_[cell] = sample(velocityYRead_, sourceX, sourceY) * velocityDissipation;
    }
  }
  swapFields(velocityXRead_, velocityXWrite_);
  swapFields(velocityYRead_, velocityYWrite_);
}

void CpuStableFluidSolver::advectDye(float deltaTime) {
  const float densityDissipation = std::exp(-std::max(kDissipationFloor, params_.densityDissipation) * deltaTime);
  const float maxBacktrace = kMaxBacktraceDistance *
                             static_cast<float>(width_) / static_cast<float>(kGridWidth);

  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const std::size_t cell = index(x, y);
      float backtraceX = sampleVelocityX(static_cast<float>(x), static_cast<float>(y)) * deltaTime;
      float backtraceY = sampleVelocityY(static_cast<float>(x), static_cast<float>(y)) * deltaTime;
      const float backtraceLength = std::hypot(backtraceX, backtraceY);
      if (backtraceLength > maxBacktrace) {
        const float scale = maxBacktrace / backtraceLength;
        backtraceX *= scale;
        backtraceY *= scale;
      }
      const float sourceX = static_cast<float>(x) - backtraceX;
      const float sourceY = static_cast<float>(y) - backtraceY;
      densityRWrite_[cell] = sample(densityRRead_, sourceX, sourceY) * densityDissipation;
      densityGWrite_[cell] = sample(densityGRead_, sourceX, sourceY) * densityDissipation;
      densityBWrite_[cell] = sample(densityBRead_, sourceX, sourceY) * densityDissipation;
    }
  }
  swapFields(densityRRead_, densityRWrite_);
  swapFields(densityGRead_, densityGWrite_);
  swapFields(densityBRead_, densityBWrite_);
}

void CpuStableFluidSolver::computeDivergence() {
  const float inverseCellSize = 0.5F;
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const int left = std::max(0, x - 1);
      const int right = std::min(width_ - 1, x + 1);
      const int down = std::max(0, y - 1);
      const int up = std::min(height_ - 1, y + 1);
      divergence_[index(x, y)] = inverseCellSize *
                                  ((velocityXRead_[index(right, y)] - velocityXRead_[index(left, y)]) +
                                   (velocityYRead_[index(x, up)] - velocityYRead_[index(x, down)]));
    }
  }
}

void CpuStableFluidSolver::solvePressure() {
  std::fill(pressureRead_.begin(), pressureRead_.end(), 0.0F);
  std::fill(pressureWrite_.begin(), pressureWrite_.end(), 0.0F);
  for (int iteration = 0; iteration < pressureIterations_; ++iteration) {
    for (int y = 0; y < height_; ++y) {
      for (int x = 0; x < width_; ++x) {
        const int left = std::max(0, x - 1);
        const int right = std::min(width_ - 1, x + 1);
        const int down = std::max(0, y - 1);
        const int up = std::min(height_ - 1, y + 1);
        const float neighborSum = pressureRead_[index(left, y)] + pressureRead_[index(right, y)] +
                                  pressureRead_[index(x, down)] + pressureRead_[index(x, up)];
        pressureWrite_[index(x, y)] = (neighborSum - divergence_[index(x, y)]) * 0.25F;
      }
    }
    swapFields(pressureRead_, pressureWrite_);
  }
}

void CpuStableFluidSolver::projectVelocity() {
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const int left = std::max(0, x - 1);
      const int right = std::min(width_ - 1, x + 1);
      const int down = std::max(0, y - 1);
      const int up = std::min(height_ - 1, y + 1);
      const std::size_t cell = index(x, y);
      const float pressureGradientX = (pressureRead_[index(right, y)] - pressureRead_[index(left, y)]) * kPressureScale;
      const float pressureGradientY = (pressureRead_[index(x, up)] - pressureRead_[index(x, down)]) * kPressureScale;
      float velocityX = velocityXRead_[cell] - pressureGradientX;
      float velocityY = velocityYRead_[cell] - pressureGradientY;

      if (params_.vorticityConfinement > 0.0F) {
        const float curl = (velocityYRead_[index(right, y)] - velocityYRead_[index(left, y)]) * kPressureScale -
                           (velocityXRead_[index(x, up)] - velocityXRead_[index(x, down)]) * kPressureScale;
        const float curlRight = std::abs((velocityYRead_[index(std::min(width_ - 1, x + 1), y)] -
                                          velocityYRead_[index(std::max(0, x - 1), y)]) * kPressureScale);
        const float curlUp = std::abs((velocityXRead_[index(x, std::min(height_ - 1, y + 1))] -
                                       velocityXRead_[index(x, std::max(0, y - 1))]) * kPressureScale);
        const float gradientLength = std::max(std::hypot(curlRight, curlUp), 1.0e-4F);
        const float normalX = curlRight / gradientLength;
        const float normalY = curlUp / gradientLength;
        velocityX += normalY * curl * params_.vorticityConfinement * params_.deltaTime;
        velocityY -= normalX * curl * params_.vorticityConfinement * params_.deltaTime;
      }

      velocityXRead_[cell] = velocityX;
      velocityYRead_[cell] = velocityY;
    }
  }
}

void CpuStableFluidSolver::applyBoundary() {
  for (int x = 0; x < width_; ++x) {
    const std::size_t top = index(x, 0);
    const std::size_t bottom = index(x, height_ - 1);
    velocityXRead_[top] = 0.0F;
    velocityYRead_[top] = 0.0F;
    velocityXRead_[bottom] = 0.0F;
    velocityYRead_[bottom] = 0.0F;
  }
  for (int y = 0; y < height_; ++y) {
    const std::size_t left = index(0, y);
    const std::size_t right = index(width_ - 1, y);
    velocityXRead_[left] = 0.0F;
    velocityYRead_[left] = 0.0F;
    velocityXRead_[right] = 0.0F;
    velocityYRead_[right] = 0.0F;
  }
}

void CpuStableFluidSolver::updateStats(float elapsedMilliseconds) {
  maxSpeed_ = 0.0F;
  dyeEnergy_ = 0.0F;
  for (std::size_t cell = 0; cell < cellCount_; ++cell) {
    maxSpeed_ = std::max(maxSpeed_, std::hypot(velocityXRead_[cell], velocityYRead_[cell]));
    dyeEnergy_ += densityRRead_[cell] + densityGRead_[cell] + densityBRead_[cell];
  }
  ++frameIndex_;
  frameStats_.frameIndex = frameIndex_;
  frameStats_.gpuMilliseconds = elapsedMilliseconds;
  frameStats_.persistentDeviceBytes = 0;
  frameStats_.pressureIterations = pressureIterations_;
  frameStats_.activeParticles = 0;
  frameStats_.neighborOverflow = 0;
}

void CpuStableFluidSolver::step(const SimulationParams& params) {
  step(params, nullptr);
}

void CpuStableFluidSolver::step(const SimulationParams& params, TelemetryCollector* telemetry) {
  params_ = params;
  params_.gridWidth = static_cast<float>(width_);
  params_.gridHeight = static_cast<float>(height_);
  params_.deltaTime = std::clamp(params.deltaTime, kMinimumDeltaTime, kMaximumDeltaTime);
  const auto start = std::chrono::steady_clock::now();

  {
    auto stage = telemetry == nullptr ? ScopedStage{} : telemetry->stage(StageId::Input);
    injectStroke(params_);
  }
  {
    auto stage = telemetry == nullptr ? ScopedStage{} : telemetry->stage(StageId::Advection);
    advectVelocity(params_.deltaTime);
    advectDye(params_.deltaTime);
  }
  {
    auto stage = telemetry == nullptr ? ScopedStage{} : telemetry->stage(StageId::Divergence);
    computeDivergence();
  }
  {
    auto stage = telemetry == nullptr ? ScopedStage{} : telemetry->stage(StageId::Pressure);
    solvePressure();
  }
  {
    auto stage = telemetry == nullptr ? ScopedStage{} : telemetry->stage(StageId::Projection);
    projectVelocity();
    applyBoundary();
  }

  const auto end = std::chrono::steady_clock::now();
  const float elapsedMilliseconds = std::chrono::duration<float, std::milli>(end - start).count();
  updateStats(elapsedMilliseconds);
}

void CpuStableFluidSolver::render(std::vector<std::uint8_t>& rgba) const {
  rgba.resize(cellCount_ * 4);
  const float exposure = std::max(0.01F, params_.exposure);
  for (int y = 0; y < height_; ++y) {
    for (int x = 0; x < width_; ++x) {
      const std::size_t cell = index(x, y);
      const int left = std::max(0, x - 1);
      const int right = std::min(width_ - 1, x + 1);
      const int down = std::max(0, y - 1);
      const int up = std::min(height_ - 1, y + 1);
      const float curl = (velocityYRead_[index(right, y)] - velocityYRead_[index(left, y)]) * 0.5F -
                         (velocityXRead_[index(x, up)] - velocityXRead_[index(x, down)]) * 0.5F;
      const float glow = std::abs(curl) * 0.016F;
      const float red = (densityRRead_[cell] + glow * 0.65F) * exposure;
      const float green = (densityGRead_[cell] + glow * 0.35F) * exposure;
      const float blue = (densityBRead_[cell] + glow) * exposure;
      const std::size_t output = cell * 4;
      rgba[output] = toByte(std::sqrt(saturate(red)));
      rgba[output + 1] = toByte(std::sqrt(saturate(green)));
      rgba[output + 2] = toByte(std::sqrt(saturate(blue)));
      rgba[output + 3] = 255;
    }
  }
}

void CpuStableFluidSolver::downloadFrame(std::vector<std::uint8_t>& rgba) const {
  render(rgba);
}

}  // namespace gpu_fluids
