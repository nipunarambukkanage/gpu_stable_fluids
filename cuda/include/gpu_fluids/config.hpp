#pragma once

#include <cstddef>
#include <cstdint>

namespace gpu_fluids {

constexpr int kGridWidth = 512;
constexpr int kGridHeight = 512;
constexpr int kCellCount = kGridWidth * kGridHeight;
constexpr int kBlockSize = 16;
constexpr int kBlocksX = kGridWidth / kBlockSize;
constexpr int kBlocksY = kGridHeight / kBlockSize;
constexpr int kPressureIterations = 20;
constexpr int kParticleCount = 8192;
constexpr int kParticleBlockSize = 128;
constexpr int kParticleBlocks = (kParticleCount + kParticleBlockSize - 1) / kParticleBlockSize;
constexpr int kTileExtent = kBlockSize + 2;

struct alignas(16) SimulationParams {
  float time = 0.0F;
  float deltaTime = 1.0F / 60.0F;
  float velocityDissipation = 0.08F;
  float densityDissipation = 0.025F;

  float gridWidth = static_cast<float>(kGridWidth);
  float gridHeight = static_cast<float>(kGridHeight);
  float pointerActive = 0.0F;
  float brushRadius = 18.0F;

  float strokeStartX = 0.0F;
  float strokeStartY = 0.0F;
  float reserved0 = 0.0F;
  float reserved1 = 0.0F;

  float strokeEndX = 0.0F;
  float strokeEndY = 0.0F;
  float reserved2 = 0.0F;
  float reserved3 = 0.0F;

  float injectedVelocityX = 0.0F;
  float injectedVelocityY = 0.0F;
  float velocityForce = 1.0F;
  float inkAmount = 1.8F;

  float inkColorR = 0.28235295F;
  float inkColorG = 0.71764708F;
  float inkColorB = 1.0F;
  float exposure = 1.15F;

  float vorticityConfinement = 0.0F;
  float reserved4 = 0.0F;
  float reserved5 = 0.0F;
  float reserved6 = 0.0F;
};

static_assert(sizeof(SimulationParams) == 112, "SimulationParams must remain seven 16-byte constant-memory slots");

struct SolverConfig {
  int pressureIterations = kPressureIterations;
  bool enableVorticity = true;
  int deviceIndex = 0;
};

struct DeviceMetrics {
  char name[256]{};
  int computeCapabilityMajor = 0;
  int computeCapabilityMinor = 0;
  int multiprocessors = 0;
  int warpSize = 0;
  int maxThreadsPerBlock = 0;
  std::size_t globalMemoryBytes = 0;
  int divergenceRegistersPerThread = 0;
  std::size_t divergenceStaticSharedBytes = 0;
};

struct FrameStats {
  std::uint64_t frameIndex = 0;
  float gpuMilliseconds = 0.0F;
  std::size_t persistentDeviceBytes = 0;
  int pressureIterations = kPressureIterations;
  int activeParticles = kParticleCount;
};

}  // namespace gpu_fluids
