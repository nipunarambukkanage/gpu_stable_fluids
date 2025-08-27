#include "gpu_fluids/sph_solver.hpp"

#include "gpu_fluids/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace gpu_fluids {
namespace {

__constant__ SphParams cSphParams;

__device__ __forceinline__ int sphCellX(float x) {
  return max(0, min(kSphGridWidth - 1, static_cast<int>(x / cSphParams.cellSize)));
}

__device__ __forceinline__ int sphCellY(float y) {
  return max(0, min(kSphGridHeight - 1, static_cast<int>(y / cSphParams.cellSize)));
}

__device__ __forceinline__ int sphCellIndex(int x, int y) {
  return y * kSphGridWidth + x;
}

__device__ __forceinline__ float safeDensity(float value) {
  return fmaxf(value, 1.0e-4F);
}

__device__ __forceinline__ void neighborRange(int cell, const unsigned int* __restrict__ cellCounts, int& count) {
  const unsigned int rawCount = cellCounts[cell];
  count = static_cast<int>(rawCount > static_cast<unsigned int>(kSphMaxParticlesPerCell)
                               ? kSphMaxParticlesPerCell
                               : rawCount);
}

__global__ void clearSpatialGridKernel(unsigned int* __restrict__ cellCounts, unsigned int* __restrict__ overflow) {
  const int cell = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (cell < kSphCellCount) {
    cellCounts[cell] = 0;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    overflow[0] = 0;
  }
}

__global__ void buildSpatialGridKernel(
    const float* __restrict__ positionX,
    const float* __restrict__ positionY,
    unsigned int* __restrict__ cellCounts,
    unsigned int* __restrict__ cellParticles,
    unsigned int* __restrict__ overflow) {
  const int particle = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (particle >= kSphParticleCount) {
    return;
  }
  const int cell = sphCellIndex(sphCellX(positionX[particle]), sphCellY(positionY[particle]));
  const unsigned int slot = atomicAdd(&cellCounts[cell], 1u);
  if (slot < static_cast<unsigned int>(kSphMaxParticlesPerCell)) {
    cellParticles[cell * kSphMaxParticlesPerCell + slot] = static_cast<unsigned int>(particle);
  } else {
    atomicAdd(overflow, 1u);
  }
}

__global__ void densityPressureKernel(
    const float* __restrict__ positionX,
    const float* __restrict__ positionY,
    const unsigned int* __restrict__ cellCounts,
    const unsigned int* __restrict__ cellParticles,
    float* __restrict__ density,
    float* __restrict__ pressure) {
  const int particle = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (particle >= kSphParticleCount) {
    return;
  }
  const float x = positionX[particle];
  const float y = positionY[particle];
  const float radius = cSphParams.smoothingRadius;
  const float inverseRadius = 1.0F / radius;
  float localDensity = 0.0F;
  const int centerX = sphCellX(x);
  const int centerY = sphCellY(y);
  for (int cellY = max(0, centerY - 1); cellY <= min(kSphGridHeight - 1, centerY + 1); ++cellY) {
    for (int cellX = max(0, centerX - 1); cellX <= min(kSphGridWidth - 1, centerX + 1); ++cellX) {
      const int cell = sphCellIndex(cellX, cellY);
      int count = 0;
      neighborRange(cell, cellCounts, count);
      for (int entry = 0; entry < count; ++entry) {
        const int neighbor = static_cast<int>(cellParticles[cell * kSphMaxParticlesPerCell + entry]);
        const float dx = x - positionX[neighbor];
        const float dy = y - positionY[neighbor];
        const float distanceSquared = dx * dx + dy * dy;
        if (distanceSquared < radius * radius) {
          const float q = 1.0F - sqrtf(distanceSquared) * inverseRadius;
          localDensity += cSphParams.particleMass * q * q;
        }
      }
    }
  }
  density[particle] = fmaxf(localDensity, 1.0e-4F);
  pressure[particle] = cSphParams.pressureStiffness * (density[particle] - cSphParams.restDensity);
}

__global__ void forceKernel(
    const float* __restrict__ positionX,
    const float* __restrict__ positionY,
    const float* __restrict__ velocityX,
    const float* __restrict__ velocityY,
    const float* __restrict__ density,
    const float* __restrict__ pressure,
    const unsigned int* __restrict__ cellCounts,
    const unsigned int* __restrict__ cellParticles,
    float* __restrict__ forceX,
    float* __restrict__ forceY) {
  const int particle = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (particle >= kSphParticleCount) {
    return;
  }
  const float x = positionX[particle];
  const float y = positionY[particle];
  const float radius = cSphParams.smoothingRadius;
  const int centerX = sphCellX(x);
  const int centerY = sphCellY(y);
  float pressureForceX = 0.0F;
  float pressureForceY = 0.0F;
  float viscosityForceX = 0.0F;
  float viscosityForceY = 0.0F;
  for (int cellY = max(0, centerY - 1); cellY <= min(kSphGridHeight - 1, centerY + 1); ++cellY) {
    for (int cellX = max(0, centerX - 1); cellX <= min(kSphGridWidth - 1, centerX + 1); ++cellX) {
      const int cell = sphCellIndex(cellX, cellY);
      int count = 0;
      neighborRange(cell, cellCounts, count);
      for (int entry = 0; entry < count; ++entry) {
        const int neighbor = static_cast<int>(cellParticles[cell * kSphMaxParticlesPerCell + entry]);
        if (neighbor == particle) {
          continue;
        }
        const float dx = x - positionX[neighbor];
        const float dy = y - positionY[neighbor];
        const float distanceSquared = dx * dx + dy * dy;
        if (distanceSquared >= radius * radius) {
          continue;
        }
        const float distance = sqrtf(fmaxf(distanceSquared, 1.0e-8F));
        const float q = 1.0F - distance / radius;
        const float normalX = dx / distance;
        const float normalY = dy / distance;
        const float neighborDensity = safeDensity(density[neighbor]);
        const float pressureTerm = 0.5F * (pressure[particle] + pressure[neighbor]) * q / neighborDensity;
        pressureForceX -= normalX * pressureTerm * cSphParams.particleMass;
        pressureForceY -= normalY * pressureTerm * cSphParams.particleMass;
        viscosityForceX += (velocityX[neighbor] - velocityX[particle]) * q / neighborDensity;
        viscosityForceY += (velocityY[neighbor] - velocityY[particle]) * q / neighborDensity;
      }
    }
  }
  forceX[particle] = pressureForceX + viscosityForceX * cSphParams.viscosity + cSphParams.gravityX;
  forceY[particle] = pressureForceY + viscosityForceY * cSphParams.viscosity + cSphParams.gravityY;
}

__global__ void integrateBoundaryKernel(
    const float* __restrict__ positionXIn,
    const float* __restrict__ positionYIn,
    const float* __restrict__ velocityXIn,
    const float* __restrict__ velocityYIn,
    const float* __restrict__ forceX,
    const float* __restrict__ forceY,
    float* __restrict__ positionXOut,
    float* __restrict__ positionYOut,
    float* __restrict__ velocityXOut,
    float* __restrict__ velocityYOut) {
  const int particle = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (particle >= kSphParticleCount) {
    return;
  }
  float velocityX = velocityXIn[particle] + forceX[particle] * cSphParams.deltaTime;
  float velocityY = velocityYIn[particle] + forceY[particle] * cSphParams.deltaTime;
  float positionX = positionXIn[particle] + velocityX * cSphParams.deltaTime;
  float positionY = positionYIn[particle] + velocityY * cSphParams.deltaTime;
  constexpr float margin = 2.0F;
  if (positionX < margin) {
    positionX = margin;
    velocityX = fabsf(velocityX) * cSphParams.boundaryDamping;
  } else if (positionX > cSphParams.worldWidth - margin) {
    positionX = cSphParams.worldWidth - margin;
    velocityX = -fabsf(velocityX) * cSphParams.boundaryDamping;
  }
  if (positionY < margin) {
    positionY = margin;
    velocityY = fabsf(velocityY) * cSphParams.boundaryDamping;
  } else if (positionY > cSphParams.worldHeight - margin) {
    positionY = cSphParams.worldHeight - margin;
    velocityY = -fabsf(velocityY) * cSphParams.boundaryDamping;
  }
  positionXOut[particle] = positionX;
  positionYOut[particle] = positionY;
  velocityXOut[particle] = velocityX * 0.999F;
  velocityYOut[particle] = velocityY * 0.999F;
}

__global__ void renderSphKernel(
    const float* __restrict__ positionX,
    const float* __restrict__ positionY,
    const unsigned int* __restrict__ cellCounts,
    const unsigned int* __restrict__ cellParticles,
    uchar4* __restrict__ output) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= kGridWidth || y >= kGridHeight) {
    return;
  }
  const float pixelX = static_cast<float>(x) + 0.5F;
  const float pixelY = static_cast<float>(y) + 0.5F;
  const float radius = cSphParams.smoothingRadius;
  const int centerX = sphCellX(pixelX);
  const int centerY = sphCellY(pixelY);
  float field = 0.0F;
  for (int cellY = max(0, centerY - 1); cellY <= min(kSphGridHeight - 1, centerY + 1); ++cellY) {
    for (int cellX = max(0, centerX - 1); cellX <= min(kSphGridWidth - 1, centerX + 1); ++cellX) {
      const int cell = sphCellIndex(cellX, cellY);
      int count = 0;
      neighborRange(cell, cellCounts, count);
      for (int entry = 0; entry < count; ++entry) {
        const int particle = static_cast<int>(cellParticles[cell * kSphMaxParticlesPerCell + entry]);
        const float dx = pixelX - positionX[particle];
        const float dy = pixelY - positionY[particle];
        const float distanceSquared = dx * dx + dy * dy;
        if (distanceSquared < radius * radius) {
          const float q = 1.0F - sqrtf(distanceSquared) / radius;
          field += q * q;
        }
      }
    }
  }
  const float phase = cSphParams.colorPhase;
  const float intensity = fminf(1.0F, field * 0.08F);
  const float red = fminf(1.0F, intensity * (0.45F + 0.25F * sinf(phase)));
  const float green = fminf(1.0F, intensity * (0.72F + 0.16F * cosf(phase * 0.7F)));
  const float blue = fminf(1.0F, intensity * 1.2F);
  output[y * kGridWidth + x] = make_uchar4(
      static_cast<unsigned char>(red * 255.0F),
      static_cast<unsigned char>(green * 255.0F),
      static_cast<unsigned char>(blue * 255.0F),
      255);
}

}  // namespace

SphSolver::SphSolver(SphParams params, int deviceIndex) : params_(params), deviceIndex_(deviceIndex) {
  FLUID_CUDA_CHECK(cudaSetDevice(deviceIndex_));
  refreshMetrics();
  try {
    FLUID_CUDA_CHECK(cudaStreamCreateWithFlags(&computeStream_, cudaStreamNonBlocking));
    FLUID_CUDA_CHECK(cudaStreamCreateWithFlags(&copyStream_, cudaStreamNonBlocking));
    FLUID_CUDA_CHECK(cudaEventCreate(&frameStart_));
    FLUID_CUDA_CHECK(cudaEventCreate(&frameEnd_));
    FLUID_CUDA_CHECK(cudaEventCreateWithFlags(&frameReady_, cudaEventDisableTiming));
    allocatePersistentResources();
    reset();
  } catch (...) {
    releasePersistentResources();
    throw;
  }
}

SphSolver::~SphSolver() {
  releasePersistentResources();
}

void SphSolver::refreshMetrics() {
  cudaDeviceProp properties{};
  FLUID_CUDA_CHECK(cudaGetDeviceProperties(&properties, deviceIndex_));
  std::strncpy(metrics_.name, properties.name, sizeof(metrics_.name) - 1);
  metrics_.computeCapabilityMajor = properties.major;
  metrics_.computeCapabilityMinor = properties.minor;
  metrics_.multiprocessors = properties.multiProcessorCount;
  metrics_.warpSize = properties.warpSize;
  metrics_.maxThreadsPerBlock = properties.maxThreadsPerBlock;
  metrics_.globalMemoryBytes = properties.totalGlobalMem;
  cudaFuncAttributes attributes{};
  FLUID_CUDA_CHECK(cudaFuncGetAttributes(&attributes, densityPressureKernel));
  metrics_.divergenceRegistersPerThread = attributes.numRegs;
  metrics_.divergenceStaticSharedBytes = attributes.sharedSizeBytes;
}

void SphSolver::allocatePersistentResources() {
  const std::size_t particleBytes = static_cast<std::size_t>(kSphParticleCount) * sizeof(float);
  const std::size_t cellCountsBytes = static_cast<std::size_t>(kSphCellCount) * sizeof(unsigned int);
  const std::size_t cellParticlesBytes = static_cast<std::size_t>(kSphCellCount) * kSphMaxParticlesPerCell * sizeof(unsigned int);
  const std::size_t frameBytes = static_cast<std::size_t>(kCellCount) * sizeof(uchar4);
  auto allocate = [this](void** destination, std::size_t bytes) {
    FLUID_CUDA_CHECK(cudaMalloc(destination, bytes));
    persistentDeviceBytes_ += bytes;
  };
  for (int index = 0; index < 2; ++index) {
    allocate(reinterpret_cast<void**>(&positionX_[index]), particleBytes);
    allocate(reinterpret_cast<void**>(&positionY_[index]), particleBytes);
    allocate(reinterpret_cast<void**>(&velocityX_[index]), particleBytes);
    allocate(reinterpret_cast<void**>(&velocityY_[index]), particleBytes);
  }
  allocate(reinterpret_cast<void**>(&density_), particleBytes);
  allocate(reinterpret_cast<void**>(&pressure_), particleBytes);
  allocate(reinterpret_cast<void**>(&forceX_), particleBytes);
  allocate(reinterpret_cast<void**>(&forceY_), particleBytes);
  allocate(reinterpret_cast<void**>(&cellCounts_), cellCountsBytes);
  allocate(reinterpret_cast<void**>(&cellParticles_), cellParticlesBytes);
  allocate(reinterpret_cast<void**>(&neighborOverflow_), sizeof(unsigned int));
  allocate(reinterpret_cast<void**>(&deviceFrame_), frameBytes);
  FLUID_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pinnedHostFrame_), frameBytes, cudaHostAllocPortable));
  FLUID_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pinnedNeighborOverflow_), sizeof(unsigned int), cudaHostAllocPortable));
  frameStats_.persistentDeviceBytes = persistentDeviceBytes_;
}

void SphSolver::releasePersistentResources() noexcept {
  if (computeStream_ != nullptr) {
    cudaStreamSynchronize(computeStream_);
  }
  if (copyStream_ != nullptr) {
    cudaStreamSynchronize(copyStream_);
  }
  for (float*& buffer : positionX_) {
    if (buffer != nullptr) {
      cudaFree(buffer);
      buffer = nullptr;
    }
  }
  for (float*& buffer : positionY_) {
    if (buffer != nullptr) {
      cudaFree(buffer);
      buffer = nullptr;
    }
  }
  for (float*& buffer : velocityX_) {
    if (buffer != nullptr) {
      cudaFree(buffer);
      buffer = nullptr;
    }
  }
  for (float*& buffer : velocityY_) {
    if (buffer != nullptr) {
      cudaFree(buffer);
      buffer = nullptr;
    }
  }
  for (float* buffer : {density_, pressure_, forceX_, forceY_}) {
    if (buffer != nullptr) {
      cudaFree(buffer);
    }
  }
  density_ = nullptr;
  pressure_ = nullptr;
  forceX_ = nullptr;
  forceY_ = nullptr;
  if (cellCounts_ != nullptr) {
    cudaFree(cellCounts_);
    cellCounts_ = nullptr;
  }
  if (cellParticles_ != nullptr) {
    cudaFree(cellParticles_);
    cellParticles_ = nullptr;
  }
  if (neighborOverflow_ != nullptr) {
    cudaFree(neighborOverflow_);
    neighborOverflow_ = nullptr;
  }
  if (deviceFrame_ != nullptr) {
    cudaFree(deviceFrame_);
    deviceFrame_ = nullptr;
  }
  if (pinnedHostFrame_ != nullptr) {
    cudaFreeHost(pinnedHostFrame_);
    pinnedHostFrame_ = nullptr;
  }
  if (pinnedNeighborOverflow_ != nullptr) {
    cudaFreeHost(pinnedNeighborOverflow_);
    pinnedNeighborOverflow_ = nullptr;
  }
  if (frameStart_ != nullptr) {
    cudaEventDestroy(frameStart_);
    frameStart_ = nullptr;
  }
  if (frameEnd_ != nullptr) {
    cudaEventDestroy(frameEnd_);
    frameEnd_ = nullptr;
  }
  if (frameReady_ != nullptr) {
    cudaEventDestroy(frameReady_);
    frameReady_ = nullptr;
  }
  if (computeStream_ != nullptr) {
    cudaStreamDestroy(computeStream_);
    computeStream_ = nullptr;
  }
  if (copyStream_ != nullptr) {
    cudaStreamDestroy(copyStream_);
    copyStream_ = nullptr;
  }
  persistentDeviceBytes_ = 0;
}

void SphSolver::initializeParticles() {
  constexpr int columns = 256;
  std::vector<float> positionX(kSphParticleCount);
  std::vector<float> positionY(kSphParticleCount);
  std::vector<float> velocityX(kSphParticleCount, 0.0F);
  std::vector<float> velocityY(kSphParticleCount, 0.0F);
  for (int particle = 0; particle < kSphParticleCount; ++particle) {
    const int column = particle % columns;
    const int row = particle / columns;
    positionX[particle] = 14.0F + static_cast<float>(column) * 1.82F;
    positionY[particle] = 14.0F + static_cast<float>(row) * 3.05F;
  }
  const std::size_t bytes = static_cast<std::size_t>(kSphParticleCount) * sizeof(float);
  for (int index = 0; index < 2; ++index) {
    FLUID_CUDA_CHECK(cudaMemcpyAsync(positionX_[index], positionX.data(), bytes, cudaMemcpyHostToDevice, computeStream_));
    FLUID_CUDA_CHECK(cudaMemcpyAsync(positionY_[index], positionY.data(), bytes, cudaMemcpyHostToDevice, computeStream_));
    FLUID_CUDA_CHECK(cudaMemcpyAsync(velocityX_[index], velocityX.data(), bytes, cudaMemcpyHostToDevice, computeStream_));
    FLUID_CUDA_CHECK(cudaMemcpyAsync(velocityY_[index], velocityY.data(), bytes, cudaMemcpyHostToDevice, computeStream_));
  }
}

void SphSolver::reset() {
  FLUID_CUDA_CHECK(cudaStreamSynchronize(computeStream_));
  FLUID_CUDA_CHECK(cudaStreamSynchronize(copyStream_));
  const std::size_t particleBytes = static_cast<std::size_t>(kSphParticleCount) * sizeof(float);
  const std::size_t cellCountsBytes = static_cast<std::size_t>(kSphCellCount) * sizeof(unsigned int);
  const std::size_t cellParticlesBytes = static_cast<std::size_t>(kSphCellCount) * kSphMaxParticlesPerCell * sizeof(unsigned int);
  FLUID_CUDA_CHECK(cudaMemsetAsync(density_, 0, particleBytes, computeStream_));
  FLUID_CUDA_CHECK(cudaMemsetAsync(pressure_, 0, particleBytes, computeStream_));
  FLUID_CUDA_CHECK(cudaMemsetAsync(forceX_, 0, particleBytes, computeStream_));
  FLUID_CUDA_CHECK(cudaMemsetAsync(forceY_, 0, particleBytes, computeStream_));
  FLUID_CUDA_CHECK(cudaMemsetAsync(cellCounts_, 0, cellCountsBytes, computeStream_));
  FLUID_CUDA_CHECK(cudaMemsetAsync(cellParticles_, 0, cellParticlesBytes, computeStream_));
  FLUID_CUDA_CHECK(cudaMemsetAsync(neighborOverflow_, 0, sizeof(unsigned int), computeStream_));
  FLUID_CUDA_CHECK(cudaMemsetAsync(deviceFrame_, 0, static_cast<std::size_t>(kCellCount) * sizeof(uchar4), computeStream_));
  initializeParticles();
  FLUID_CUDA_CHECK(cudaStreamSynchronize(computeStream_));
  particleRead_ = 0;
  frameIndex_ = 0;
  frameStats_ = {};
  frameStats_.persistentDeviceBytes = persistentDeviceBytes_;
  frameStats_.pressureIterations = 0;
  frameStats_.activeParticles = kSphParticleCount;
}

void SphSolver::launchClearGrid() {
  const dim3 block(256);
  const dim3 grid((kSphCellCount + static_cast<int>(block.x) - 1) / static_cast<int>(block.x));
  clearSpatialGridKernel<<<grid, block, 0, computeStream_>>>(cellCounts_, neighborOverflow_);
  FLUID_CUDA_KERNEL_CHECK("clearSpatialGridKernel");
}

void SphSolver::launchBuildGrid() {
  const dim3 block(kSphParticleBlockSize);
  const dim3 grid(kSphParticleBlocks);
  buildSpatialGridKernel<<<grid, block, 0, computeStream_>>>(positionX_[particleRead_], positionY_[particleRead_], cellCounts_, cellParticles_, neighborOverflow_);
  FLUID_CUDA_KERNEL_CHECK("buildSpatialGridKernel");
}

void SphSolver::launchDensityPressure() {
  const dim3 block(kSphParticleBlockSize);
  const dim3 grid(kSphParticleBlocks);
  densityPressureKernel<<<grid, block, 0, computeStream_>>>(positionX_[particleRead_], positionY_[particleRead_], cellCounts_, cellParticles_, density_, pressure_);
  FLUID_CUDA_KERNEL_CHECK("densityPressureKernel");
}

void SphSolver::launchForces() {
  const dim3 block(kSphParticleBlockSize);
  const dim3 grid(kSphParticleBlocks);
  forceKernel<<<grid, block, 0, computeStream_>>>(positionX_[particleRead_], positionY_[particleRead_], velocityX_[particleRead_], velocityY_[particleRead_], density_, pressure_, cellCounts_, cellParticles_, forceX_, forceY_);
  FLUID_CUDA_KERNEL_CHECK("forceKernel");
}

void SphSolver::launchIntegrate() {
  const dim3 block(kSphParticleBlockSize);
  const dim3 grid(kSphParticleBlocks);
  integrateBoundaryKernel<<<grid, block, 0, computeStream_>>>(positionX_[particleRead_], positionY_[particleRead_], velocityX_[particleRead_], velocityY_[particleRead_], forceX_, forceY_, positionX_[1 - particleRead_], positionY_[1 - particleRead_], velocityX_[1 - particleRead_], velocityY_[1 - particleRead_]);
  FLUID_CUDA_KERNEL_CHECK("integrateBoundaryKernel");
  particleRead_ = 1 - particleRead_;
}

void SphSolver::launchRender() {
  const dim3 block(kBlockSize, kBlockSize);
  const dim3 grid(kBlocksX, kBlocksY);
  renderSphKernel<<<grid, block, 0, computeStream_>>>(positionX_[particleRead_], positionY_[particleRead_], cellCounts_, cellParticles_, reinterpret_cast<uchar4*>(deviceFrame_));
  FLUID_CUDA_KERNEL_CHECK("renderSphKernel");
}

void SphSolver::step(const SphParams& params) {
  params_ = params;
  params_.worldWidth = static_cast<float>(kGridWidth);
  params_.worldHeight = static_cast<float>(kGridHeight);
  params_.smoothingRadius = kSphSmoothingRadius;
  params_.cellSize = kSphSmoothingRadius;
  FLUID_CUDA_CHECK(cudaMemcpyToSymbolAsync(cSphParams, &params_, sizeof(SphParams), 0, cudaMemcpyHostToDevice, computeStream_));
  FLUID_CUDA_CHECK(cudaEventRecord(frameStart_, computeStream_));
  launchClearGrid();
  launchBuildGrid();
  launchDensityPressure();
  launchForces();
  launchIntegrate();
  // Integration changes positions; rebuild the index before GPU rendering so visualization queries the current state.
  launchClearGrid();
  launchBuildGrid();
  launchRender();
  FLUID_CUDA_CHECK(cudaEventRecord(frameEnd_, computeStream_));
  FLUID_CUDA_CHECK(cudaEventRecord(frameReady_, computeStream_));
  FLUID_CUDA_CHECK(cudaStreamWaitEvent(copyStream_, frameReady_, 0));
  FLUID_CUDA_CHECK(cudaMemcpyAsync(
      pinnedHostFrame_,
      deviceFrame_,
      static_cast<std::size_t>(kCellCount) * sizeof(uchar4),
      cudaMemcpyDeviceToHost,
      copyStream_));
  FLUID_CUDA_CHECK(cudaMemcpyAsync(
      pinnedNeighborOverflow_,
      neighborOverflow_,
      sizeof(unsigned int),
      cudaMemcpyDeviceToHost,
      copyStream_));
  ++frameIndex_;
  frameStats_.frameIndex = frameIndex_;
  frameStats_.persistentDeviceBytes = persistentDeviceBytes_;
  frameStats_.pressureIterations = 0;
  frameStats_.activeParticles = kSphParticleCount;
}

void SphSolver::downloadFrame(std::vector<std::uint8_t>& rgba) {
  const std::size_t bytes = static_cast<std::size_t>(kCellCount) * sizeof(uchar4);
  rgba.resize(bytes);
  FLUID_CUDA_CHECK(cudaStreamSynchronize(copyStream_));
  FLUID_CUDA_CHECK(cudaEventElapsedTime(&frameStats_.gpuMilliseconds, frameStart_, frameEnd_));
  frameStats_.neighborOverflow = *pinnedNeighborOverflow_;
  std::memcpy(rgba.data(), pinnedHostFrame_, bytes);
}

}  // namespace gpu_fluids
