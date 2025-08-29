import { existsSync, readFileSync } from "node:fs";

const requiredFiles = [
  "CMakeLists.txt",
  "CMakePresets.json",
  "cuda/include/gpu_fluids/config.hpp",
  "cuda/include/gpu_fluids/benchmark.hpp",
  "cuda/include/gpu_fluids/cuda_utils.hpp",
  "cuda/include/gpu_fluids/solver.hpp",
  "cuda/include/gpu_fluids/sph_solver.hpp",
  "cuda/include/gpu_fluids/visualization.hpp",
  "cuda/include/gpu_fluids/cpu_solver.hpp",
  "cuda/include/gpu_fluids/experiment_manifest.hpp",
  "cuda/include/gpu_fluids/gpu_diagnostics.hpp",
  "cuda/include/gpu_fluids/launch_policy.hpp",
  "cuda/include/gpu_fluids/native_runtime.hpp",
  "cuda/include/gpu_fluids/numeric_validation.hpp",
  "cuda/include/gpu_fluids/performance_model.hpp",
  "cuda/include/gpu_fluids/telemetry.hpp",
  "cuda/include/gpu_fluids/trace_recorder.hpp",
  "cuda/src/main.cpp",
  "cuda/src/benchmark.cpp",
  "cuda/src/solver.cu",
  "cuda/src/sph_solver.cu",
  "cuda/src/cpu_solver.cpp",
  "cuda/src/experiment_manifest.cpp",
  "cuda/src/gpu_diagnostics.cpp",
  "cuda/src/launch_policy.cpp",
  "cuda/src/native_runtime.cpp",
  "cuda/src/numeric_validation.cpp",
  "cuda/src/performance_model.cpp",
  "cuda/src/telemetry.cpp",
  "cuda/src/trace_recorder.cpp",
  "cuda/src/reference_main.cpp",
  "cuda/src/visualization.cpp",
  "tests/native/cpu_solver_contract.cpp",
  "tests/native/experiment_manifest_contract.cpp",
  "tests/native/native_runtime_contract.cpp",
  "tests/native/numeric_validation_contract.cpp",
  "tests/native/performance_model_contract.cpp",
  "tests/native/trace_recorder_contract.cpp",
  "tests/native/benchmark_contract.cpp",
  "tests/native/config_contract.cpp",
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
const sphSolver = read("cuda/src/sph_solver.cu");
const header = read("cuda/include/gpu_fluids/solver.hpp");
const sphHeader = read("cuda/include/gpu_fluids/sph_solver.hpp");
const cpuHeader = read("cuda/include/gpu_fluids/cpu_solver.hpp");
const cpuSolver = read("cuda/src/cpu_solver.cpp");
const manifest = read("cuda/src/experiment_manifest.cpp");
const gpuDiagnostics = read("cuda/src/gpu_diagnostics.cpp");
const launchPolicy = read("cuda/src/launch_policy.cpp");
const runtimeHeader = read("cuda/include/gpu_fluids/native_runtime.hpp");
const runtime = read("cuda/src/native_runtime.cpp");
const performanceModel = read("cuda/src/performance_model.cpp");
const telemetry = read("cuda/src/telemetry.cpp");
const traceRecorder = read("cuda/src/trace_recorder.cpp");
const referenceMain = read("cuda/src/reference_main.cpp");
const config = read("cuda/include/gpu_fluids/config.hpp");
const benchmark = read("cuda/src/benchmark.cpp");
const cli = read("cuda/src/main.cpp");
const cmakePresets = read("CMakePresets.json");
const benchmarkContract = read("tests/native/benchmark_contract.cpp");

assert(cmake.includes("project(gpu_stable_fluids_native LANGUAGES CXX CUDA)"), "CMake must enable native C++ and CUDA languages");
assert(cmake.includes("find_package(CUDAToolkit REQUIRED)"), "CMake must link the CUDA toolkit explicitly");
assert(cmake.includes("CUDA_SEPARABLE_COMPILATION ON"), "CMake must enable separable CUDA compilation");
assert(cmake.includes("add_test(NAME native_config_contract"), "CMake must register the native configuration contract test");
assert(cmakePresets.includes('"cuda-release"') && cmakePresets.includes('"CMAKE_CUDA_ARCHITECTURES"'), "CMake presets must define a reproducible CUDA release configuration");
assert(config.includes("static_assert(sizeof(SimulationParams) == 112"), "host/device parameter layout must remain explicitly validated");
assert(benchmark.includes("BenchmarkLedger::summarize") && benchmark.includes("p95Milliseconds"), "native benchmark ledger must compute summary latency statistics");
assert(benchmark.includes("effectiveBandwidthGBPerSecond") && benchmark.includes("BenchmarkLedger::toJson"), "native benchmark ledger must report bandwidth and structured JSON");
assert(config.includes("kMaxBacktraceDistance = 48.0F"), "native configuration must own the backtrace stability limit");
assert(solver.includes("backtraceDisplacement") && solver.includes("hypotf"), "native advection must bound extreme backtrace displacement");
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
assert(sphSolver.includes("__global__ void buildSpatialGridKernel"), "SPH mode must build a GPU uniform neighbor grid");
assert(sphSolver.includes("atomicAdd"), "SPH neighbor insertion must use GPU atomic indexing");
assert(sphSolver.includes("densityPressureKernel") && sphSolver.includes("forceKernel"), "SPH mode must separate density/pressure and force stages");
assert(sphSolver.includes("cellParticles"), "SPH mode must retain a device-resident cell-to-particle index");
assert(sphSolver.includes("cudaMemcpyToSymbolAsync"), "SPH controls must use asynchronous constant-memory upload");
assert(sphHeader.includes("class SphSolver"), "SPH mode must expose a CPU orchestration boundary");
assert(cli.includes("writePpm"), "CPU visualization coordination must export device-generated frames");
assert(cmake.includes("FLUID_BUILD_REFERENCE"), "CMake must expose the deterministic C++ reference runtime option");
assert(cmake.includes("add_library(fluid_reference STATIC"), "CMake must build the C++ reference runtime library");
assert(cmake.includes("add_executable(fluid_reference_demo"), "CMake must expose a native C++ reference executable");
assert(cpuHeader.includes("class CpuStableFluidSolver"), "reference solver must expose a C++ host implementation");
assert(cpuSolver.includes("sample") && cpuSolver.includes("solvePressure") && cpuSolver.includes("projectVelocity"), "reference solver must implement advection, pressure, and projection stages");
assert(telemetry.includes("TelemetryJournal::writeJson") && telemetry.includes("TelemetryJournal::writeCsv"), "native runtime must export structured telemetry");
assert(traceRecorder.includes("TraceRecorder::writeChromeTrace") && traceRecorder.includes("traceEvents"), "native runtime must export a Chrome-compatible execution trace");
assert(referenceMain.includes("NativeReferenceRuntime") && referenceMain.includes("runtime.run"), "reference executable must exercise the native runtime orchestration component");
assert(runtimeHeader.includes("class NativeReferenceRuntime") && runtimeHeader.includes("RuntimeCommandType"), "native runtime must expose a command-driven orchestration boundary");
assert(runtime.includes("NativeReferenceRuntime::applyCommands") && runtime.includes("NativeReferenceRuntime::writeReport"), "native runtime must implement command processing and structured reports");
assert(runtime.includes("singleStepRequested_") && runtime.includes("writeTelemetryArtifacts"), "native runtime must provide deterministic stepping and telemetry artifacts");
assert(runtime.includes("validateRgbaFrame") && runtime.includes("validateSimulationDiagnostics"), "native runtime must validate frames and numerical diagnostics");
assert(cmake.includes("cuda/src/gpu_diagnostics.cpp"), "CUDA library must compile the native GPU diagnostics component");
assert(gpuDiagnostics.includes("cudaMemGetInfo") && gpuDiagnostics.includes("cudaDeviceGetAttribute"), "GPU diagnostics must query runtime memory and device attributes");
assert(gpuDiagnostics.includes("evaluateLaunch") && gpuDiagnostics.includes("theoreticalOccupancy"), "GPU diagnostics must expose launch legality and occupancy estimates");
assert(launchPolicy.includes("LaunchPolicySelector::select") && launchPolicy.includes("sharedTileBytes"), "CUDA launch policy must adapt block and shared-memory budgets");
assert(cli.includes("LaunchPolicySelector::select") && cli.includes("gpu-diagnostics.json"), "CUDA CLI must use adaptive launch policy and emit device diagnostics");
assert(performanceModel.includes("estimateCoalescedTransactions") && performanceModel.includes("estimateTileBytes"), "performance model must account for coalescing and shared-memory tiles");
assert(performanceModel.includes("estimateStableFluidPipeline") && performanceModel.includes("dominantBottleneck"), "performance model must expose a stage-level pipeline estimate");
assert(cmake.includes("cuda/src/numeric_validation.cpp"), "CMake must build the native numerical validation component");
assert(cmake.includes("add_test(NAME native_benchmark_contract"), "CMake must register the native benchmark contract test");
assert(benchmarkContract.includes("BenchmarkLedger") && benchmarkContract.includes("p95Milliseconds"), "native benchmark contract must exercise summary percentiles");
assert(benchmarkContract.includes("effectiveBandwidthGBPerSecond") && benchmarkContract.includes("throwsInvalidArgument"), "native benchmark contract must exercise bandwidth and validation behavior");
assert(read("cuda/src/numeric_validation.cpp").includes("ValidationReport::toJson"), "numeric validation must expose structured failure details");
assert(manifest.includes("makeRuntimeManifest") && manifest.includes("ExperimentManifest::write"), "native runtime must emit a reproducibility manifest");
assert(!new RegExp(forbiddenMarker, "i").test([cmake, solver, sphSolver, header, sphHeader, config, cli].join("\n")), "native CUDA source must remain unrelated to external integrations");

if (failures.length > 0) {
  console.error("cuda-contract: failed");
  for (const failure of failures) console.error("- " + failure);
  process.exitCode = 1;
} else {
  console.log("cuda-contract: passed");
  console.log("- native CMake/CUDA source tree found");
  console.log("- persistent buffers, shared-memory tiling, streams, events, and pinned output found");
}
