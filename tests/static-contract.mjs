import { existsSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";

const requiredFiles = [
  "CMakePresets.json",
  ".editorconfig",
  ".clang-format",
  ".gitattributes",
  ".github/workflows/quality.yml",
  ".github/workflows/native-cuda.yml",
  "fluid-simulation.html",
  "styles/fluid-lab.css",
  "src/main.js",
  "src/config/simulation.js",
  "src/gpu/capabilities.js",
  "src/gpu/pipeline-factory.js",
  "src/gpu/timestamp-profiler.js",
  "src/gpu/telemetry.js",
  "src/gpu/shaders.js",
  "src/runtime/adaptive-quality.js",
  "src/runtime/diagnostics.js",
  "src/runtime/input-recorder.js",
  "src/runtime/settings-store.js",
  "src/ui/performance-hud.js",
  "docs/ARCHITECTURE.md",
  "docs/CUDA_TO_WEBGPU.md",
  "docs/DIAGNOSTICS_SCHEMA.md",
  "docs/NUMERICAL_METHOD.md",
  "docs/VALIDATION.md",
  "docs/screenshots/webgpu-lab-mockup.png",
  "docs/screenshots/cuda-sph-pipeline-mockup.png",
  "CONTRIBUTING.md",
  "tools/serve.mjs"
];

const failures = [];
const assert = (condition, message) => {
  if (!condition) failures.push(message);
};
const read = (filePath) => readFileSync(filePath, "utf8");
const forbiddenMarker = ["Est", "uary"].join("");

for (const filePath of requiredFiles) {
  assert(existsSync(filePath), `missing required file: ${filePath}`);
}

const html = read("fluid-simulation.html");
const runtime = read("src/main.js");
const config = read("src/config/simulation.js");
const capabilities = read("src/gpu/capabilities.js");
const pipelineFactory = read("src/gpu/pipeline-factory.js");
const profiler = read("src/gpu/timestamp-profiler.js");
const telemetry = read("src/gpu/telemetry.js");
const shaders = read("src/gpu/shaders.js");
const adaptiveQuality = read("src/runtime/adaptive-quality.js");
const diagnostics = read("src/runtime/diagnostics.js");
const inputRecorder = read("src/runtime/input-recorder.js");
const settingsStore = read("src/runtime/settings-store.js");
const hud = read("src/ui/performance-hud.js");
const styles = read("styles/fluid-lab.css");
const readme = read("README.md");

assert(html.includes('<link rel="stylesheet" href="./styles/fluid-lab.css">'), "HTML shell must load the external stylesheet");
assert(html.includes('<script type="module" src="./src/main.js"></script>'), "HTML shell must load the module entry point");
assert(!/<style[ >]/i.test(html), "HTML shell must not contain inline CSS");
assert(!/<script>/.test(html), "HTML shell must not contain a classic inline script");
assert(runtime.includes('"./config/simulation.js"'), "main entry must import shared simulation configuration");
assert(runtime.includes('"./gpu/timestamp-profiler.js"'), "main entry must import the GPU profiler module");
assert(runtime.includes('"./gpu/shaders.js"'), "main entry must import the GPU shader module");
assert(config.includes("export const GRID_WIDTH = 512"), "simulation configuration must own the fixed grid size");
assert(config.includes("export const MAX_BACKTRACE_DISTANCE = 48"), "simulation configuration must own the backtrace stability limit");
assert(config.includes("export const QUALITY_PROFILES"), "simulation configuration must own the runtime quality profiles");
assert(capabilities.includes("inspectWebGpuAdapter") && capabilities.includes("requestDeviceWithFallback"), "GPU capability policy must be isolated in a testable module");
assert(pipelineFactory.includes("createComputePipelineAsync") && pipelineFactory.includes("createGpuPipelineBatch"), "pipeline creation must support asynchronous compilation with a fallback");
assert(profiler.includes("timestamp-query"), "GPU profiler module must use optional timestamp-query");
assert(telemetry.includes("snapshotRuntimeTelemetry"), "runtime telemetry must expose a serializable diagnostics snapshot");
assert(adaptiveQuality.includes("createAdaptiveQualityGovernor"), "adaptive quality must be isolated in a reusable governor module");
assert(diagnostics.includes("DIAGNOSTICS_SCHEMA_VERSION"), "diagnostics module must expose a versioned report schema");
assert(inputRecorder.includes("validateInputRecording"), "input macro recording must expose a validation boundary");
assert(settingsStore.includes("SETTINGS_SCHEMA_VERSION") && settingsStore.includes("sanitizeSettings"), "settings persistence must be versioned and sanitized");
assert(hud.includes("createPerformanceHud"), "performance HUD must be isolated as a reusable UI component");
assert(html.includes("Save diagnostics JSON"), "UI must expose local diagnostics export");
assert(html.includes('id="qualityProfile"') && html.includes('id="hudToggle"'), "UI must expose quality and performance HUD controls");
assert(html.includes('id="renderMode"') && html.includes('id="brushMode"'), "UI must expose diagnostic field and brush mode controls");
assert(html.includes('id="adaptiveQualityToggle"') && html.includes('id="rememberToggle"'), "UI must expose adaptive quality and settings persistence controls");
assert(html.includes('id="recordButton"') && html.includes('id="replayButton"'), "UI must expose stroke macro controls");
assert(runtime.includes("recordRuntimeSubmission"), "runtime must count submitted GPU command buffers");
assert(runtime.includes("simulationSettings.pressureIterations") && runtime.includes("createPerformanceHud"), "runtime must apply quality profiles and update the HUD");
assert(runtime.includes("particleIndex: 0"), "GPU resource state must initialize tracer ping-pong index");
assert(runtime.includes("app.qualityGovernor.observe") && runtime.includes("updateReplayPointer"), "runtime must connect adaptive quality and macro replay to the frame loop");
assert(readme.includes("docs/screenshots/webgpu-lab-mockup.png") && readme.includes("docs/screenshots/cuda-sph-pipeline-mockup.png"), "README must reference both repository visual assets");
assert(readme.includes("Save diagnostics JSON"), "README must document the diagnostics export");
assert(shaders.includes("export const particleComputeShaderCode"), "GPU shader module must own the tracer compute source");
assert(shaders.includes("export const indirectArgsShaderCode"), "GPU shader module must own the indirect draw-arguments source");
assert(shaders.includes("let brushMode") && shaders.includes("velocityTexture") && shaders.includes("vorticityTexture"), "GPU shaders must expose brush modes and diagnostic field inputs");
assert(styles.includes(":root"), "external stylesheet must contain the application design tokens");
assert((shaders.match(/@compute @workgroup_size\(/g) || []).length === 9, "GPU shader module must contain the nine expected compute entry points");
assert((shaders.match(/workgroupBarrier\(\)/g) || []).length === 3, "tiled stencil kernels must retain three workgroup barriers");
assert(runtime.includes("PARTICLE_COUNT"), "runtime must retain GPU tracer workload configuration");
assert(runtime.includes("GPUBufferUsage.INDIRECT"), "tracer draw arguments must use an indirect-capable GPU buffer");
assert(runtime.includes("drawIndirect"), "tracer rendering must consume GPU-generated indirect arguments");
assert(!new RegExp(`TODO|${forbiddenMarker}|fetch\\(|from ['"]https?:`, "i").test(`${runtime}\n${config}\n${capabilities}\n${pipelineFactory}\n${profiler}\n${telemetry}\n${shaders}\n${adaptiveQuality}\n${diagnostics}\n${inputRecorder}\n${settingsStore}\n${hud}`), "runtime must remain local and free of forbidden dependencies");

for (const modulePath of ["src/main.js", "src/config/simulation.js", "src/gpu/capabilities.js", "src/gpu/pipeline-factory.js", "src/gpu/timestamp-profiler.js", "src/gpu/telemetry.js", "src/gpu/shaders.js", "src/runtime/adaptive-quality.js", "src/runtime/diagnostics.js", "src/runtime/input-recorder.js", "src/runtime/settings-store.js", "src/ui/performance-hud.js", "tools/serve.mjs", "tests/static-contract.mjs"]) {
  const result = spawnSync(process.execPath, ["--check", modulePath], { encoding: "utf8" });
  assert(result.status === 0, `${modulePath} failed Node syntax validation: ${result.stderr.trim()}`);
}

if (failures.length > 0) {
  console.error("static-contract: failed");
  for (const failure of failures) console.error(`- ${failure}`);
  process.exitCode = 1;
} else {
  console.log("static-contract: passed");
  console.log("- modular HTML shell, external CSS, and GPU entry modules found");
  console.log("- nine WGSL compute entry points and three tiled barriers found in src/gpu/shaders.js");
  console.log("- Node syntax validation passed for runtime and tooling modules");
}
