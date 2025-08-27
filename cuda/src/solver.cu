#include "gpu_fluids/solver.hpp"

#include "gpu_fluids/cuda_utils.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace gpu_fluids {
namespace {

__constant__ SimulationParams cParams;

__device__ __forceinline__ int clampX(int value) {
  return max(0, min(kGridWidth - 1, value));
}

__device__ __forceinline__ int clampY(int value) {
  return max(0, min(kGridHeight - 1, value));
}

__device__ __forceinline__ int cellIndex(int x, int y) {
  return y * kGridWidth + x;
}

__device__ __forceinline__ float clampUnit(float value) {
  return fminf(1.0F, fmaxf(0.0F, value));
}

__device__ __forceinline__ float2 sampleVelocity(const float2* __restrict__ field, float2 position) {
  const float x = fminf(static_cast<float>(kGridWidth) - 0.5F, fmaxf(0.5F, position.x)) - 0.5F;
  const float y = fminf(static_cast<float>(kGridHeight) - 0.5F, fmaxf(0.5F, position.y)) - 0.5F;
  const int x0 = clampX(static_cast<int>(floorf(x)));
  const int y0 = clampY(static_cast<int>(floorf(y)));
  const int x1 = clampX(x0 + 1);
  const int y1 = clampY(y0 + 1);
  const float tx = x - floorf(x);
  const float ty = y - floorf(y);
  const float2 topLeft = field[cellIndex(x0, y0)];
  const float2 topRight = field[cellIndex(x1, y0)];
  const float2 bottomLeft = field[cellIndex(x0, y1)];
  const float2 bottomRight = field[cellIndex(x1, y1)];
  const float2 top = make_float2(topLeft.x + (topRight.x - topLeft.x) * tx, topLeft.y + (topRight.y - topLeft.y) * tx);
  const float2 bottom = make_float2(bottomLeft.x + (bottomRight.x - bottomLeft.x) * tx, bottomLeft.y + (bottomRight.y - bottomLeft.y) * tx);
  return make_float2(top.x + (bottom.x - top.x) * ty, top.y + (bottom.y - top.y) * ty);
}

__device__ __forceinline__ float4 sampleDensity(const float4* __restrict__ field, float2 position) {
  const float x = fminf(static_cast<float>(kGridWidth) - 0.5F, fmaxf(0.5F, position.x)) - 0.5F;
  const float y = fminf(static_cast<float>(kGridHeight) - 0.5F, fmaxf(0.5F, position.y)) - 0.5F;
  const int x0 = clampX(static_cast<int>(floorf(x)));
  const int y0 = clampY(static_cast<int>(floorf(y)));
  const int x1 = clampX(x0 + 1);
  const int y1 = clampY(y0 + 1);
  const float tx = x - floorf(x);
  const float ty = y - floorf(y);
  const float4 topLeft = field[cellIndex(x0, y0)];
  const float4 topRight = field[cellIndex(x1, y0)];
  const float4 bottomLeft = field[cellIndex(x0, y1)];
  const float4 bottomRight = field[cellIndex(x1, y1)];
  const float4 top = make_float4(
      topLeft.x + (topRight.x - topLeft.x) * tx,
      topLeft.y + (topRight.y - topLeft.y) * tx,
      topLeft.z + (topRight.z - topLeft.z) * tx,
      topLeft.w + (topRight.w - topLeft.w) * tx);
  const float4 bottom = make_float4(
      bottomLeft.x + (bottomRight.x - bottomLeft.x) * tx,
      bottomLeft.y + (bottomRight.y - bottomLeft.y) * tx,
      bottomLeft.z + (bottomRight.z - bottomLeft.z) * tx,
      bottomLeft.w + (bottomRight.w - bottomLeft.w) * tx);
  return make_float4(
      top.x + (bottom.x - top.x) * ty,
      top.y + (bottom.y - top.y) * ty,
      top.z + (bottom.z - top.z) * ty,
      top.w + (bottom.w - top.w) * ty);
}

__device__ __forceinline__ void stageVelocityTile(float2* tile, const float2* __restrict__ field) {
  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int x = static_cast<int>(blockIdx.x) * kBlockSize + tx;
  const int y = static_cast<int>(blockIdx.y) * kBlockSize + ty;
  tile[(ty + 1) * kTileExtent + (tx + 1)] = field[cellIndex(clampX(x), clampY(y))];
  if (tx == 0) {
    tile[(ty + 1) * kTileExtent] = field[cellIndex(clampX(x - 1), clampY(y))];
  }
  if (tx == kBlockSize - 1) {
    tile[(ty + 1) * kTileExtent + kTileExtent - 1] = field[cellIndex(clampX(x + 1), clampY(y))];
  }
  if (ty == 0) {
    tile[tx + 1] = field[cellIndex(clampX(x), clampY(y - 1))];
  }
  if (ty == kBlockSize - 1) {
    tile[(kTileExtent - 1) * kTileExtent + tx + 1] = field[cellIndex(clampX(x), clampY(y + 1))];
  }
  __syncthreads();
}

__device__ __forceinline__ void stageScalarTile(float* tile, const float* __restrict__ field) {
  const int tx = static_cast<int>(threadIdx.x);
  const int ty = static_cast<int>(threadIdx.y);
  const int x = static_cast<int>(blockIdx.x) * kBlockSize + tx;
  const int y = static_cast<int>(blockIdx.y) * kBlockSize + ty;
  tile[(ty + 1) * kTileExtent + (tx + 1)] = field[cellIndex(clampX(x), clampY(y))];
  if (tx == 0) {
    tile[(ty + 1) * kTileExtent] = field[cellIndex(clampX(x - 1), clampY(y))];
  }
  if (tx == kBlockSize - 1) {
    tile[(ty + 1) * kTileExtent + kTileExtent - 1] = field[cellIndex(clampX(x + 1), clampY(y))];
  }
  if (ty == 0) {
    tile[tx + 1] = field[cellIndex(clampX(x), clampY(y - 1))];
  }
  if (ty == kBlockSize - 1) {
    tile[(kTileExtent - 1) * kTileExtent + tx + 1] = field[cellIndex(clampX(x), clampY(y + 1))];
  }
  __syncthreads();
}

__global__ void splatKernel(
    const float2* __restrict__ velocityIn,
    const float4* __restrict__ densityIn,
    float2* __restrict__ velocityOut,
    float4* __restrict__ densityOut) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= kGridWidth || y >= kGridHeight) {
    return;
  }
  const int index = cellIndex(x, y);
  float2 velocity = velocityIn[index];
  float4 density = densityIn[index];
  if (cParams.pointerActive > 0.5F) {
    const float2 point = make_float2(static_cast<float>(x) + 0.5F, static_cast<float>(y) + 0.5F);
    const float2 start = make_float2(cParams.strokeStartX, cParams.strokeStartY);
    const float2 end = make_float2(cParams.strokeEndX, cParams.strokeEndY);
    const float2 segment = make_float2(end.x - start.x, end.y - start.y);
    const float segmentLengthSquared = fmaxf(segment.x * segment.x + segment.y * segment.y, 1.0e-5F);
    const float projection = clampUnit(((point.x - start.x) * segment.x + (point.y - start.y) * segment.y) / segmentLengthSquared);
    const float nearestX = start.x + segment.x * projection;
    const float nearestY = start.y + segment.y * projection;
    const float dx = point.x - nearestX;
    const float dy = point.y - nearestY;
    const float radius = fmaxf(cParams.brushRadius, 1.0F);
    const float falloff = expf(-(dx * dx + dy * dy) / (radius * radius * 0.65F));
    const float injection = falloff * cParams.inkAmount * cParams.deltaTime * 60.0F;
    velocity.x += cParams.injectedVelocityX * cParams.velocityForce * falloff * 0.0125F;
    velocity.y += cParams.injectedVelocityY * cParams.velocityForce * falloff * 0.0125F;
    density.x += cParams.inkColorR * injection;
    density.y += cParams.inkColorG * injection;
    density.z += cParams.inkColorB * injection;
    density.w = fminf(1.0F, density.w + injection * 0.65F);
  }
  velocityOut[index] = velocity;
  densityOut[index] = density;
}

__global__ void advectKernel(
    const float2* __restrict__ velocityIn,
    const float4* __restrict__ densityIn,
    float2* __restrict__ velocityOut,
    float4* __restrict__ densityOut) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= kGridWidth || y >= kGridHeight) {
    return;
  }
  const int index = cellIndex(x, y);
  const float2 position = make_float2(static_cast<float>(x) + 0.5F, static_cast<float>(y) + 0.5F);
  const float2 localVelocity = sampleVelocity(velocityIn, position);
  const float2 backtraced = make_float2(
      position.x - localVelocity.x * cParams.deltaTime,
      position.y - localVelocity.y * cParams.deltaTime);
  const float2 advectedVelocity = sampleVelocity(velocityIn, backtraced);
  const float4 advectedDensity = sampleDensity(densityIn, backtraced);
  const float velocityDecay = expf(-cParams.velocityDissipation * cParams.deltaTime * 60.0F);
  const float densityDecay = expf(-cParams.densityDissipation * cParams.deltaTime * 60.0F);
  velocityOut[index] = make_float2(advectedVelocity.x * velocityDecay, advectedVelocity.y * velocityDecay);
  densityOut[index] = make_float4(
      advectedDensity.x * densityDecay,
      advectedDensity.y * densityDecay,
      advectedDensity.z * densityDecay,
      advectedDensity.w * densityDecay);
}

__global__ void divergenceKernel(const float2* __restrict__ velocity, float* __restrict__ divergence) {
  __shared__ float2 tile[kTileExtent * kTileExtent];
  stageVelocityTile(tile, velocity);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= kGridWidth || y >= kGridHeight) {
    return;
  }
  const int center = (static_cast<int>(threadIdx.y) + 1) * kTileExtent + static_cast<int>(threadIdx.x) + 1;
  divergence[cellIndex(x, y)] = 0.5F * (
      tile[center + 1].x - tile[center - 1].x +
      tile[center + kTileExtent].y - tile[center - kTileExtent].y);
}

__global__ void pressureJacobiKernel(
    const float* __restrict__ pressureIn,
    const float* __restrict__ divergence,
    float* __restrict__ pressureOut) {
  __shared__ float tile[kTileExtent * kTileExtent];
  stageScalarTile(tile, pressureIn);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= kGridWidth || y >= kGridHeight) {
    return;
  }
  const int center = (static_cast<int>(threadIdx.y) + 1) * kTileExtent + static_cast<int>(threadIdx.x) + 1;
  pressureOut[cellIndex(x, y)] = 0.25F * (
      tile[center - 1] + tile[center + 1] +
      tile[center - kTileExtent] + tile[center + kTileExtent] -
      divergence[cellIndex(x, y)]);
}

__global__ void vorticityKernel(const float2* __restrict__ velocity, float* __restrict__ vorticity) {
  __shared__ float2 tile[kTileExtent * kTileExtent];
  stageVelocityTile(tile, velocity);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= kGridWidth || y >= kGridHeight) {
    return;
  }
  const int center = (static_cast<int>(threadIdx.y) + 1) * kTileExtent + static_cast<int>(threadIdx.x) + 1;
  vorticity[cellIndex(x, y)] = 0.5F * (
      tile[center + 1].y - tile[center - 1].y -
      tile[center + kTileExtent].x + tile[center - kTileExtent].x);
}

__global__ void confinementKernel(
    const float2* __restrict__ velocityIn,
    const float* __restrict__ vorticity,
    float2* __restrict__ velocityOut) {
  __shared__ float tile[kTileExtent * kTileExtent];
  stageScalarTile(tile, vorticity);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= kGridWidth || y >= kGridHeight) {
    return;
  }
  const int center = (static_cast<int>(threadIdx.y) + 1) * kTileExtent + static_cast<int>(threadIdx.x) + 1;
  const float gradientX = fabsf(tile[center + 1]) - fabsf(tile[center - 1]);
  const float gradientY = fabsf(tile[center + kTileExtent]) - fabsf(tile[center - kTileExtent]);
  const float gradientLength = rsqrtf(fmaxf(gradientX * gradientX + gradientY * gradientY, 1.0e-6F));
  const float signedCurl = tile[center];
  const float forceX = gradientY * gradientLength * signedCurl * cParams.vorticityConfinement;
  const float forceY = -gradientX * gradientLength * signedCurl * cParams.vorticityConfinement;
  const float2 input = velocityIn[cellIndex(x, y)];
  velocityOut[cellIndex(x, y)] = make_float2(
      input.x + forceX * cParams.deltaTime * 12.0F,
      input.y + forceY * cParams.deltaTime * 12.0F);
}

__global__ void gradientSubtractKernel(
    const float* __restrict__ pressure,
    const float2* __restrict__ velocityIn,
    float2* __restrict__ velocityOut) {
  __shared__ float tile[kTileExtent * kTileExtent];
  stageScalarTile(tile, pressure);
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= kGridWidth || y >= kGridHeight) {
    return;
  }
  const int center = (static_cast<int>(threadIdx.y) + 1) * kTileExtent + static_cast<int>(threadIdx.x) + 1;
  float2 velocity = velocityIn[cellIndex(x, y)];
  velocity.x -= 0.5F * (tile[center + 1] - tile[center - 1]);
  velocity.y -= 0.5F * (tile[center + kTileExtent] - tile[center - kTileExtent]);
  if (x == 0 || x == kGridWidth - 1) {
    velocity.x = 0.0F;
  }
  if (y == 0 || y == kGridHeight - 1) {
    velocity.y = 0.0F;
  }
  velocityOut[cellIndex(x, y)] = velocity;
}

__device__ __forceinline__ std::uint32_t nextSeed(std::uint32_t seed) {
  return seed * 1664525u + 1013904223u;
}

__global__ void particleAdvectionKernel(
    const float2* __restrict__ velocity,
    const float4* __restrict__ particlesIn,
    float4* __restrict__ particlesOut) {
  const int particle = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (particle >= kParticleCount) {
    return;
  }
  float4 value = particlesIn[particle];
  const float2 sampledVelocity = sampleVelocity(velocity, make_float2(value.x, value.y));
  value.x += sampledVelocity.x * cParams.deltaTime;
  value.y += sampledVelocity.y * cParams.deltaTime;
  value.z += cParams.deltaTime;
  std::uint32_t seed = __float_as_uint(value.w);
  const bool escaped = value.x < 1.0F || value.x >= static_cast<float>(kGridWidth - 1) ||
                       value.y < 1.0F || value.y >= static_cast<float>(kGridHeight - 1) || value.z > 8.0F;
  if (escaped) {
    seed = nextSeed(seed);
    value.x = 16.0F + static_cast<float>(seed % (kGridWidth - 32));
    seed = nextSeed(seed);
    value.y = 16.0F + static_cast<float>(seed % (kGridHeight - 32));
    value.z = 0.0F;
    value.w = __uint_as_float(seed);
  }
  particlesOut[particle] = value;
}

__global__ void renderDensityKernel(const float4* __restrict__ density, uchar4* __restrict__ output) {
  const int x = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const int y = static_cast<int>(blockIdx.y * blockDim.y + threadIdx.y);
  if (x >= kGridWidth || y >= kGridHeight) {
    return;
  }
  const float4 source = density[cellIndex(x, y)];
  const float exposure = fmaxf(cParams.exposure, 0.05F);
  const float r = powf(1.0F - expf(-fmaxf(source.x, 0.0F) * exposure), 0.45454545F);
  const float g = powf(1.0F - expf(-fmaxf(source.y, 0.0F) * exposure), 0.45454545F);
  const float b = powf(1.0F - expf(-fmaxf(source.z, 0.0F) * exposure), 0.45454545F);
  output[cellIndex(x, y)] = make_uchar4(
      static_cast<unsigned char>(clampUnit(r) * 255.0F),
      static_cast<unsigned char>(clampUnit(g) * 255.0F),
      static_cast<unsigned char>(clampUnit(b) * 255.0F),
      255);
}

}  // namespace

StableFluidSolver::StableFluidSolver(SolverConfig config) : config_(config) {
  config_.pressureIterations = std::max(1, std::min(config_.pressureIterations, 128));
  FLUID_CUDA_CHECK(cudaSetDevice(config_.deviceIndex));
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

StableFluidSolver::~StableFluidSolver() {
  releasePersistentResources();
}

void StableFluidSolver::refreshMetrics() {
  cudaDeviceProp properties{};
  FLUID_CUDA_CHECK(cudaGetDeviceProperties(&properties, config_.deviceIndex));
  std::strncpy(metrics_.name, properties.name, sizeof(metrics_.name) - 1);
  metrics_.computeCapabilityMajor = properties.major;
  metrics_.computeCapabilityMinor = properties.minor;
  metrics_.multiprocessors = properties.multiProcessorCount;
  metrics_.warpSize = properties.warpSize;
  metrics_.maxThreadsPerBlock = properties.maxThreadsPerBlock;
  metrics_.globalMemoryBytes = properties.totalGlobalMem;
  cudaFuncAttributes attributes{};
  FLUID_CUDA_CHECK(cudaFuncGetAttributes(&attributes, divergenceKernel));
  metrics_.divergenceRegistersPerThread = attributes.numRegs;
  metrics_.divergenceStaticSharedBytes = attributes.sharedSizeBytes;
}

void StableFluidSolver::allocatePersistentResources() {
  const std::size_t velocityBytes = static_cast<std::size_t>(kCellCount) * sizeof(float2);
  const std::size_t densityBytes = static_cast<std::size_t>(kCellCount) * sizeof(float4);
  const std::size_t scalarBytes = static_cast<std::size_t>(kCellCount) * sizeof(float);
  const std::size_t particleBytes = static_cast<std::size_t>(kParticleCount) * sizeof(float4);
  const std::size_t frameBytes = static_cast<std::size_t>(kCellCount) * sizeof(uchar4);
  auto allocate = [this](void** destination, std::size_t bytes) {
    FLUID_CUDA_CHECK(cudaMalloc(destination, bytes));
    persistentDeviceBytes_ += bytes;
  };
  allocate(reinterpret_cast<void**>(&velocity_[0]), velocityBytes);
  allocate(reinterpret_cast<void**>(&velocity_[1]), velocityBytes);
  allocate(reinterpret_cast<void**>(&density_[0]), densityBytes);
  allocate(reinterpret_cast<void**>(&density_[1]), densityBytes);
  allocate(reinterpret_cast<void**>(&pressure_[0]), scalarBytes);
  allocate(reinterpret_cast<void**>(&pressure_[1]), scalarBytes);
  allocate(reinterpret_cast<void**>(&divergence_), scalarBytes);
  allocate(reinterpret_cast<void**>(&vorticity_), scalarBytes);
  allocate(reinterpret_cast<void**>(&particles_[0]), particleBytes);
  allocate(reinterpret_cast<void**>(&particles_[1]), particleBytes);
  allocate(reinterpret_cast<void**>(&deviceFrame_), frameBytes);
  FLUID_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&pinnedHostFrame_), frameBytes, cudaHostAllocPortable));
  frameStats_.persistentDeviceBytes = persistentDeviceBytes_;
}

void StableFluidSolver::releasePersistentResources() noexcept {
  if (computeStream_ != nullptr) {
    cudaStreamSynchronize(computeStream_);
  }
  if (copyStream_ != nullptr) {
    cudaStreamSynchronize(copyStream_);
  }
  for (float2*& buffer : velocity_) {
    if (buffer != nullptr) {
      cudaFree(buffer);
      buffer = nullptr;
    }
  }
  for (float4*& buffer : density_) {
    if (buffer != nullptr) {
      cudaFree(buffer);
      buffer = nullptr;
    }
  }
  for (float*& buffer : pressure_) {
    if (buffer != nullptr) {
      cudaFree(buffer);
      buffer = nullptr;
    }
  }
  if (divergence_ != nullptr) {
    cudaFree(divergence_);
    divergence_ = nullptr;
  }
  if (vorticity_ != nullptr) {
    cudaFree(vorticity_);
    vorticity_ = nullptr;
  }
  for (float4*& buffer : particles_) {
    if (buffer != nullptr) {
      cudaFree(buffer);
      buffer = nullptr;
    }
  }
  if (deviceFrame_ != nullptr) {
    cudaFree(deviceFrame_);
    deviceFrame_ = nullptr;
  }
  if (pinnedHostFrame_ != nullptr) {
    cudaFreeHost(pinnedHostFrame_);
    pinnedHostFrame_ = nullptr;
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

void StableFluidSolver::initializeParticles() {
  std::vector<float4> initial(static_cast<std::size_t>(kParticleCount));
  for (int particle = 0; particle < kParticleCount; ++particle) {
    std::uint32_t seed = static_cast<std::uint32_t>(particle) * 1664525u + 1013904223u;
    seed = seed * 22695477u + 1u;
    float seedAsFloat = 0.0F;
    std::memcpy(&seedAsFloat, &seed, sizeof(seedAsFloat));
    initial[particle] = make_float4(
        0.5F + static_cast<float>(seed) / 4294967296.0F * (kGridWidth - 1),
        0.5F + static_cast<float>(seed ^ 0x9e3779b9u) / 4294967296.0F * (kGridHeight - 1),
        static_cast<float>(particle % 240) / 240.0F * 4.0F,
        seedAsFloat);
  }
  const std::size_t bytes = initial.size() * sizeof(float4);
  FLUID_CUDA_CHECK(cudaMemcpyAsync(particles_[0], initial.data(), bytes, cudaMemcpyHostToDevice, computeStream_));
  FLUID_CUDA_CHECK(cudaMemcpyAsync(particles_[1], initial.data(), bytes, cudaMemcpyHostToDevice, computeStream_));
}

void StableFluidSolver::reset() {
  FLUID_CUDA_CHECK(cudaStreamSynchronize(computeStream_));
  FLUID_CUDA_CHECK(cudaStreamSynchronize(copyStream_));
  const std::size_t velocityBytes = static_cast<std::size_t>(kCellCount) * sizeof(float2);
  const std::size_t densityBytes = static_cast<std::size_t>(kCellCount) * sizeof(float4);
  const std::size_t scalarBytes = static_cast<std::size_t>(kCellCount) * sizeof(float);
  FLUID_CUDA_CHECK(cudaMemsetAsync(velocity_[0], 0, velocityBytes, computeStream_));
  FLUID_CUDA_CHECK(cudaMemsetAsync(velocity_[1], 0, velocityBytes, computeStream_));
  FLUID_CUDA_CHECK(cudaMemsetAsync(density_[0], 0, densityBytes, computeStream_));
  FLUID_CUDA_CHECK(cudaMemsetAsync(density_[1], 0, densityBytes, computeStream_));
  FLUID_CUDA_CHECK(cudaMemsetAsync(pressure_[0], 0, scalarBytes, computeStream_));
  FLUID_CUDA_CHECK(cudaMemsetAsync(pressure_[1], 0, scalarBytes, computeStream_));
  FLUID_CUDA_CHECK(cudaMemsetAsync(divergence_, 0, scalarBytes, computeStream_));
  FLUID_CUDA_CHECK(cudaMemsetAsync(vorticity_, 0, scalarBytes, computeStream_));
  initializeParticles();
  FLUID_CUDA_CHECK(cudaStreamSynchronize(computeStream_));
  velocityRead_ = 0;
  densityRead_ = 0;
  pressureRead_ = 0;
  particleRead_ = 0;
  frameIndex_ = 0;
  frameStats_ = {};
  frameStats_.persistentDeviceBytes = persistentDeviceBytes_;
  frameStats_.pressureIterations = config_.pressureIterations;
  frameStats_.activeParticles = kParticleCount;
}

void StableFluidSolver::launchSplat() {
  const dim3 block(kBlockSize, kBlockSize);
  const dim3 grid(kBlocksX, kBlocksY);
  splatKernel<<<grid, block, 0, computeStream_>>>(velocity_[velocityRead_], density_[densityRead_], velocity_[1 - velocityRead_], density_[1 - densityRead_]);
  FLUID_CUDA_KERNEL_CHECK("splatKernel");
  velocityRead_ = 1 - velocityRead_;
  densityRead_ = 1 - densityRead_;
}

void StableFluidSolver::launchAdvection() {
  const dim3 block(kBlockSize, kBlockSize);
  const dim3 grid(kBlocksX, kBlocksY);
  advectKernel<<<grid, block, 0, computeStream_>>>(velocity_[velocityRead_], density_[densityRead_], velocity_[1 - velocityRead_], density_[1 - densityRead_]);
  FLUID_CUDA_KERNEL_CHECK("advectKernel");
  velocityRead_ = 1 - velocityRead_;
  densityRead_ = 1 - densityRead_;
}

void StableFluidSolver::launchVorticity() {
  const dim3 block(kBlockSize, kBlockSize);
  const dim3 grid(kBlocksX, kBlocksY);
  vorticityKernel<<<grid, block, 0, computeStream_>>>(velocity_[velocityRead_], vorticity_);
  FLUID_CUDA_KERNEL_CHECK("vorticityKernel");
}

void StableFluidSolver::launchConfinement() {
  const dim3 block(kBlockSize, kBlockSize);
  const dim3 grid(kBlocksX, kBlocksY);
  confinementKernel<<<grid, block, 0, computeStream_>>>(velocity_[velocityRead_], vorticity_, velocity_[1 - velocityRead_]);
  FLUID_CUDA_KERNEL_CHECK("confinementKernel");
  velocityRead_ = 1 - velocityRead_;
}

void StableFluidSolver::launchDivergence() {
  const dim3 block(kBlockSize, kBlockSize);
  const dim3 grid(kBlocksX, kBlocksY);
  divergenceKernel<<<grid, block, 0, computeStream_>>>(velocity_[velocityRead_], divergence_);
  FLUID_CUDA_KERNEL_CHECK("divergenceKernel");
}

void StableFluidSolver::launchPressure() {
  const dim3 block(kBlockSize, kBlockSize);
  const dim3 grid(kBlocksX, kBlocksY);
  for (int iteration = 0; iteration < config_.pressureIterations; ++iteration) {
    pressureJacobiKernel<<<grid, block, 0, computeStream_>>>(pressure_[pressureRead_], divergence_, pressure_[1 - pressureRead_]);
    FLUID_CUDA_KERNEL_CHECK("pressureJacobiKernel");
    pressureRead_ = 1 - pressureRead_;
  }
}

void StableFluidSolver::launchGradient() {
  const dim3 block(kBlockSize, kBlockSize);
  const dim3 grid(kBlocksX, kBlocksY);
  gradientSubtractKernel<<<grid, block, 0, computeStream_>>>(pressure_[pressureRead_], velocity_[velocityRead_], velocity_[1 - velocityRead_]);
  FLUID_CUDA_KERNEL_CHECK("gradientSubtractKernel");
  velocityRead_ = 1 - velocityRead_;
}

void StableFluidSolver::launchParticles() {
  const dim3 block(kParticleBlockSize);
  const dim3 grid(kParticleBlocks);
  particleAdvectionKernel<<<grid, block, 0, computeStream_>>>(velocity_[velocityRead_], particles_[particleRead_], particles_[1 - particleRead_]);
  FLUID_CUDA_KERNEL_CHECK("particleAdvectionKernel");
  particleRead_ = 1 - particleRead_;
}

void StableFluidSolver::launchRender() {
  const dim3 block(kBlockSize, kBlockSize);
  const dim3 grid(kBlocksX, kBlocksY);
  renderDensityKernel<<<grid, block, 0, computeStream_>>>(density_[densityRead_], reinterpret_cast<uchar4*>(deviceFrame_));
  FLUID_CUDA_KERNEL_CHECK("renderDensityKernel");
}

void StableFluidSolver::step(const SimulationParams& params) {
  params_ = params;
  params_.gridWidth = static_cast<float>(kGridWidth);
  params_.gridHeight = static_cast<float>(kGridHeight);
  if (!config_.enableVorticity) {
    params_.vorticityConfinement = 0.0F;
  }
  FLUID_CUDA_CHECK(cudaMemcpyToSymbolAsync(cParams, &params_, sizeof(SimulationParams), 0, cudaMemcpyHostToDevice, computeStream_));
  FLUID_CUDA_CHECK(cudaEventRecord(frameStart_, computeStream_));
  launchSplat();
  launchAdvection();
  if (config_.enableVorticity && params_.vorticityConfinement > 0.0001F) {
    launchVorticity();
    launchConfinement();
  }
  launchDivergence();
  launchPressure();
  launchGradient();
  launchParticles();
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
  ++frameIndex_;
  frameStats_.frameIndex = frameIndex_;
  frameStats_.persistentDeviceBytes = persistentDeviceBytes_;
  frameStats_.pressureIterations = config_.pressureIterations;
  frameStats_.activeParticles = kParticleCount;
}

void StableFluidSolver::downloadFrame(std::vector<std::uint8_t>& rgba) {
  const std::size_t bytes = static_cast<std::size_t>(kCellCount) * sizeof(uchar4);
  rgba.resize(bytes);
  FLUID_CUDA_CHECK(cudaStreamSynchronize(copyStream_));
  FLUID_CUDA_CHECK(cudaEventElapsedTime(&frameStats_.gpuMilliseconds, frameStart_, frameEnd_));
  std::memcpy(rgba.data(), pinnedHostFrame_, bytes);
}

}  // namespace gpu_fluids
