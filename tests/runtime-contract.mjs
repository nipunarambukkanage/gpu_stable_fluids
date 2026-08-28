import assert from "node:assert/strict";

import { formatCapabilityFailure, inspectWebGpuAdapter, requestDeviceWithFallback } from "../src/gpu/capabilities.js";
import { createGpuPipelineBatch } from "../src/gpu/pipeline-factory.js";
import { createDiagnosticsReport, DIAGNOSTICS_SCHEMA_VERSION } from "../src/runtime/diagnostics.js";
import { createAdaptiveQualityGovernor } from "../src/runtime/adaptive-quality.js";
import { createInputRecorder, validateInputRecording } from "../src/runtime/input-recorder.js";
import { clearSettings, loadSettings, saveSettings, SETTINGS_SCHEMA_VERSION, SETTINGS_STORAGE_KEY } from "../src/runtime/settings-store.js";
import { createRuntimeTelemetry, recordGpuSample, recordRuntimeFrame, recordRuntimeSubmission, snapshotRuntimeTelemetry } from "../src/gpu/telemetry.js";
import { BRUSH_MODES, DEFAULT_BRUSH_MODE, DEFAULT_QUALITY_PROFILE, DEFAULT_RENDER_MODE, QUALITY_PROFILES, RENDER_MODES } from "../src/config/simulation.js";

assert.equal(DEFAULT_QUALITY_PROFILE, "balanced");
assert.deepEqual(
  Object.fromEntries(Object.entries(QUALITY_PROFILES).map(([name, profile]) => [name, profile.pressureIterations])),
  { performance: 8, balanced: 20, cinematic: 36 }
);
assert.equal(DEFAULT_RENDER_MODE, "density");
assert.equal(DEFAULT_BRUSH_MODE, "paint");
assert.equal(BRUSH_MODES.erase.value, 2);
assert.equal(RENDER_MODES.vorticity.label, "Vorticity");

const governor = createAdaptiveQualityGovernor({ cooldownMs: 0, downshiftSampleThreshold: 2, upshiftSampleThreshold: 3 });
assert.equal(governor.observe(30, 0), null);
assert.equal(governor.observe(30, 1).profile, "performance");
assert.equal(governor.snapshot().pressureIterations, 8);
assert.equal(governor.observe(1, 2), null);
assert.equal(governor.observe(1, 3), null);
assert.equal(governor.observe(1, 4).profile, "balanced");

const recorder = createInputRecorder({ maxSamples: 2 });
recorder.start(100);
assert.equal(recorder.record({ x: 8, y: 9, velocityX: 2, active: true }, 110), true);
assert.equal(recorder.record({ x: 10, y: 12, active: false }, 120), true);
assert.equal(recorder.record({ x: 20, y: 22 }, 130), false);
const recording = validateInputRecording(recorder.snapshot());
assert.equal(recording.samples.length, 2);
assert.equal(recording.durationMs, 20);

const memoryStorage = new Map();
const storage = {
  getItem: (key) => memoryStorage.get(key) ?? null,
  setItem: (key, value) => memoryStorage.set(key, value),
  removeItem: (key) => memoryStorage.delete(key)
};
assert.equal(saveSettings(storage, { inkColor: "#ABCDEF", brushRadius: 100, renderMode: "vorticity", brushMode: "erase", adaptiveQuality: true }), true);
assert.equal(JSON.parse(memoryStorage.get(SETTINGS_STORAGE_KEY)).schemaVersion, SETTINGS_SCHEMA_VERSION);
const restoredSettings = loadSettings(storage);
assert.equal(restoredSettings.inkColor, "#abcdef");
assert.equal(restoredSettings.brushRadius, 64);
assert.equal(restoredSettings.renderMode, "vorticity");
assert.equal(restoredSettings.brushMode, "erase");
assert.equal(restoredSettings.adaptiveQuality, true);
assert.equal(clearSettings(storage), true);
assert.equal(loadSettings(storage), null);

const supportedAdapter = {
  limits: {
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 16,
    maxComputeWorkgroupSizeY: 16,
    maxStorageBufferBindingSize: 2_000_000
  },
  features: new Set(["timestamp-query"])
};
const capabilityRequirements = { workgroupSize: 16, particleBufferSize: 1_000_000 };
const capabilities = inspectWebGpuAdapter(supportedAdapter, capabilityRequirements);
assert.equal(capabilities.supported, true);
assert.deepEqual(capabilities.optionalFeatures, ["timestamp-query"]);
assert.equal(capabilities.requiredLimits.maxStorageBufferBindingSize, 1_000_000);

let deviceRequests = 0;
const fallbackSelection = await requestDeviceWithFallback({
  requestDevice: async ({ requiredFeatures }) => {
    deviceRequests += 1;
    if (requiredFeatures.length > 0) {
      throw new Error("optional timestamp feature rejected by test adapter");
    }
    return { id: "baseline-device" };
  }
}, capabilities);
assert.equal(deviceRequests, 2);
assert.equal(fallbackSelection.capabilities.featureTier, "baseline");

const pipelineCalls = [];
const pipelineDevice = {
  createComputePipelineAsync: async (descriptor) => {
    pipelineCalls.push(`async:${descriptor.label}`);
    return { label: descriptor.label };
  },
  createRenderPipelineAsync: async (descriptor) => {
    pipelineCalls.push(`async:${descriptor.label}`);
    return { label: descriptor.label };
  },
  createComputePipeline: () => { throw new Error("synchronous fallback should not be selected"); },
  createRenderPipeline: () => { throw new Error("synchronous fallback should not be selected"); }
};
const pipelines = await createGpuPipelineBatch(pipelineDevice, [
  { kind: "compute", descriptor: { label: "compute-a" } },
  { kind: "render", descriptor: { label: "render-a" } }
]);
assert.deepEqual(pipelineCalls, ["async:compute-a", "async:render-a"]);
assert.deepEqual(pipelines.map((pipeline) => pipeline.label), ["compute-a", "render-a"]);

const unsupportedAdapter = {
  limits: { maxComputeInvocationsPerWorkgroup: 64, maxComputeWorkgroupSizeX: 8, maxComputeWorkgroupSizeY: 8, maxStorageBufferBindingSize: 512 },
  features: new Set()
};
const unsupported = inspectWebGpuAdapter(unsupportedAdapter, capabilityRequirements);
assert.equal(unsupported.supported, false);
assert.match(formatCapabilityFailure(unsupported), /maxComputeInvocationsPerWorkgroup/);

const telemetry = createRuntimeTelemetry(100);
recordRuntimeFrame(telemetry, 16, 2);
recordRuntimeFrame(telemetry, 50, 4);
recordRuntimeFrame(telemetry, Number.NaN, Number.POSITIVE_INFINITY);
recordRuntimeSubmission(telemetry);
recordGpuSample(telemetry, 3.5);
const runtime = snapshotRuntimeTelemetry(telemetry, 1_100);
assert.equal(runtime.frameCount, 3);
assert.equal(runtime.submittedFrames, 1);
assert.equal(runtime.longFrames, 1);
assert.equal(runtime.gpuSamples, 1);
assert.equal(runtime.uptimeMs, 1_000);
assert.equal(runtime.lastFrameDeltaMs, 0);
assert.equal(runtime.lastCpuEncodeMs, 0);

const report = createDiagnosticsReport({
  generatedAt: "2026-08-27T00:00:00.000Z",
  grid: [512, 512],
  pressureIterations: 20,
  maxBacktraceDistance: 48,
  tracerCount: 8192,
  workgroup: [16, 16, 1],
  settings: { brushRadius: 18 },
  tracersEnabled: true,
  adapterInfo: { vendor: "test" },
  adapterLimits: supportedAdapter.limits,
  capabilities,
  features: ["timestamp-query"],
  runtime,
  qualityGovernor: governor.snapshot()
});
assert.equal(report.schemaVersion, DIAGNOSTICS_SCHEMA_VERSION);
assert.equal(report.simulation.maxBacktraceDistance, 48);
assert.equal(report.capabilities.featureTier, "timestamp-query");
assert.equal(report.adapter.limits.maxStorageBufferBindingSize, 2_000_000);
assert.equal(report.runtime.qualityGovernor.profile, "balanced");

console.log("runtime-contract: passed");
console.log("- capability policy, telemetry aggregation, and diagnostics schema validated");
