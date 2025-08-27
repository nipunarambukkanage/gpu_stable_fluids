import { existsSync, readFileSync } from "node:fs";

const requiredFiles = [
  "CMakeLists.txt",
  "cuda/include/gpu_fluids/config.hpp",
  "cuda/include/gpu_fluids/cuda_utils.hpp",
  "cuda/include/gpu_fluids/solver.hpp",
  "cuda/include/gpu_fluids/visualization.hpp",
  "cuda/src/main.cpp",
  "cuda/src/solver.cu",
  "cuda/src/visualization.cpp",
  "docs/CUDA_NATIVE.md"
];

const failures = [];
const forbiddenMarker = ["Est", "uary"].join("");
const assert = (condition, message) => {
  if (!condition) failures.push(message);
};
const read = (filePath) => readFileSync(filePath, "utf8");

for (const filePath of requiredFiles) {
  assert(existsSync(filePath), "missing native CUDA file: " + filePath);
}

const cmake = read("CMakeLists.txt");
const solver = read("cuda/src/solver.cu");
const header = read("cuda/include/gpu_fluids/solver.hpp");
const config = read("cuda/include/gpu_fluids/config.hpp");
const cli = read("cuda/src/main.cpp");

assert(cmake.includes("project(gpu_stable_fluids_native LANGUAGES CXX CUDA)"), "CMake must enable native C++ and CUDA languages");
assert(cmake.includes("find_package(CUDAToolkit REQUIRED)"), "CMake must link the CUDA toolkit explicitly");
assert(cmake.includes("CUDA_SEPARABLE_COMPILATION ON"), "CMake must enable separable CUDA compilation");
assert(config.includes("static_assert(sizeof(SimulationParams) == 112"), "host/device parameter layout must remain explicitly validated");
assert(solver.includes("__global__ void divergenceKernel"), "native solver must contain a CUDA divergence kernel");
assert(solver.includes("__shared__ float2 tile"), "neighbor stencils must use shared-memory tiles");
assert((solver.match(/__syncthreads\(\)/g) || []).length >= 2, "shared-memory stencil stages must use block synchronization");
assert(solver.includes("stageVelocityTile") && solver.includes("stageScalarTile"), "stencil kernels must share explicit velocity and scalar tile loaders");
assert(solver.includes("__restrict__"), "global-memory kernel pointers should communicate aliasing intent");
assert(solver.includes("cudaMalloc"), "device allocations must be explicit and persistent");
assert(solver.includes("cudaMemcpyToSymbolAsync"), "per-frame controls must use an asynchronous constant-memory upload");
assert(solver.includes("cudaStreamCreateWithFlags"), "solver must own nonblocking CUDA streams");
assert(solver.includes("cudaEventRecord"), "solver must use CUDA events for ordering and timing");
assert(solver.includes("cudaHostAlloc"), "visualization output must use pinned host memory");
assert(solver.includes("cudaStreamWaitEvent"), "copy coordination must be stream-ordered");
assert(solver.includes("launchPressure"), "CPU orchestration must keep global synchronization between pressure stages");
assert(solver.includes("pressureJacobiKernel<<<"), "pressure iterations must be distinct kernel launches");
assert(cli.includes("writePpm"), "CPU visualization coordination must export device-generated frames");
assert(!new RegExp(forbiddenMarker, "i").test([cmake, solver, header, config, cli].join("\n")), "native CUDA source must remain unrelated to external integrations");

if (failures.length > 0) {
  console.error("cuda-contract: failed");
  for (const failure of failures) console.error("- " + failure);
  process.exitCode = 1;
} else {
  console.log("cuda-contract: passed");
  console.log("- native CMake/CUDA source tree found");
  console.log("- persistent buffers, shared-memory tiling, streams, events, and pinned output found");
}
