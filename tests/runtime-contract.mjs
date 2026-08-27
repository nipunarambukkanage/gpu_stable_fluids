import assert from "node:assert/strict";

import { formatCapabilityFailure, inspectWebGpuAdapter } from "../src/gpu/capabilities.js";
import { createDiagnosticsReport, DIAGNOSTICS_SCHEMA_VERSION } from "../src/runtime/diagnostics.js";
import { createRuntimeTelemetry, recordGpuSample, recordRuntimeFrame, recordRuntimeSubmission, snapshotRuntimeTelemetry } from "../src/gpu/telemetry.js";

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
recordRuntimeSubmission(telemetry);
recordGpuSample(telemetry, 3.5);
const runtime = snapshotRuntimeTelemetry(telemetry, 1_100);
assert.equal(runtime.frameCount, 2);
assert.equal(runtime.submittedFrames, 1);
assert.equal(runtime.longFrames, 1);
assert.equal(runtime.gpuSamples, 1);
assert.equal(runtime.uptimeMs, 1_000);

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
  runtime
});
assert.equal(report.schemaVersion, DIAGNOSTICS_SCHEMA_VERSION);
assert.equal(report.simulation.maxBacktraceDistance, 48);
assert.equal(report.capabilities.featureTier, "timestamp-query");
assert.equal(report.adapter.limits.maxStorageBufferBindingSize, 2_000_000);

console.log("runtime-contract: passed");
console.log("- capability policy, telemetry aggregation, and diagnostics schema validated");
