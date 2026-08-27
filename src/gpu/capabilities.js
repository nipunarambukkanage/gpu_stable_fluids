/**
 * Capability policy for the browser GPU backend.
 *
 * This module is intentionally independent of WebGPU globals so it can be
 * exercised by Node-based contract tests and reused by future front ends.
 */
export function inspectWebGpuAdapter(adapter, { workgroupSize, particleBufferSize }) {
  const limits = adapter?.limits ?? {};
  const requiredLimits = {
    maxComputeInvocationsPerWorkgroup: workgroupSize * workgroupSize,
    maxComputeWorkgroupSizeX: workgroupSize,
    maxComputeWorkgroupSizeY: workgroupSize,
    maxStorageBufferBindingSize: particleBufferSize
  };
  const failures = Object.entries(requiredLimits)
    .map(([name, required]) => ({ name, required, actual: Number(limits[name] ?? 0) }))
    .filter(({ required, actual }) => actual < required);
  const timestampQuery = adapter?.features?.has?.("timestamp-query") ?? false;

  return {
    supported: failures.length === 0,
    featureTier: timestampQuery ? "timestamp-query" : "baseline",
    optionalFeatures: timestampQuery ? ["timestamp-query"] : [],
    requiredLimits,
    failures
  };
}

export function formatCapabilityFailure(capabilities) {
  if (capabilities.supported) {
    return "GPU capability requirements satisfied.";
  }
  return capabilities.failures
    .map(({ name, required, actual }) => `${name} requires ${required}, adapter exposes ${actual}`)
    .join("; ");
}
