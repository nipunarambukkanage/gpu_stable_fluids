import { existsSync, readFileSync } from "node:fs";
import { spawnSync } from "node:child_process";

const requiredFiles = [
  "fluid-simulation.html",
  "styles/fluid-lab.css",
  "src/main.js",
  "src/config/simulation.js",
  "src/gpu/timestamp-profiler.js",
  "src/gpu/shaders.js",
  "docs/ARCHITECTURE.md",
  "docs/CUDA_TO_WEBGPU.md",
  "docs/NUMERICAL_METHOD.md",
  "docs/VALIDATION.md",
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
const profiler = read("src/gpu/timestamp-profiler.js");
const shaders = read("src/gpu/shaders.js");
const styles = read("styles/fluid-lab.css");

assert(html.includes('<link rel="stylesheet" href="./styles/fluid-lab.css">'), "HTML shell must load the external stylesheet");
assert(html.includes('<script type="module" src="./src/main.js"></script>'), "HTML shell must load the module entry point");
assert(!/<style[ >]/i.test(html), "HTML shell must not contain inline CSS");
assert(!/<script>/.test(html), "HTML shell must not contain a classic inline script");
assert(runtime.includes('"./config/simulation.js"'), "main entry must import shared simulation configuration");
assert(runtime.includes('"./gpu/timestamp-profiler.js"'), "main entry must import the GPU profiler module");
assert(runtime.includes('"./gpu/shaders.js"'), "main entry must import the GPU shader module");
assert(config.includes("export const GRID_WIDTH = 512"), "simulation configuration must own the fixed grid size");
assert(profiler.includes("timestamp-query"), "GPU profiler module must use optional timestamp-query");
assert(shaders.includes("export const particleComputeShaderCode"), "GPU shader module must own the tracer compute source");
assert(styles.includes(":root"), "external stylesheet must contain the application design tokens");
assert((shaders.match(/@compute @workgroup_size\(/g) || []).length === 8, "GPU shader module must contain the eight expected compute entry points");
assert((shaders.match(/workgroupBarrier\(\)/g) || []).length === 3, "tiled stencil kernels must retain three workgroup barriers");
assert(runtime.includes("PARTICLE_COUNT"), "runtime must retain GPU tracer workload configuration");
assert(!new RegExp(`TODO|${forbiddenMarker}|fetch\\(|from ['"]https?:`, "i").test(`${runtime}\n${config}\n${profiler}\n${shaders}`), "runtime must remain local and free of forbidden dependencies");

for (const modulePath of ["src/main.js", "src/config/simulation.js", "src/gpu/timestamp-profiler.js", "src/gpu/shaders.js", "tools/serve.mjs", "tests/static-contract.mjs"]) {
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
  console.log("- eight WGSL compute entry points and three tiled barriers found in src/gpu/shaders.js");
  console.log("- Node syntax validation passed for runtime and tooling modules");
}
