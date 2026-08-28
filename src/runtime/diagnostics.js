export const DIAGNOSTICS_SCHEMA_VERSION = 1;

function supportedAdapterLimits(adapterLimits) {
  return Object.fromEntries(
    Object.entries(adapterLimits || {})
      .filter(([key]) => key.startsWith("maxCompute") || key.startsWith("maxStorage"))
  );
}

export function createDiagnosticsReport({
  generatedAt = new Date().toISOString(),
  grid,
  pressureIterations,
  maxBacktraceDistance,
  tracerCount,
  workgroup,
  settings,
  tracersEnabled,
  adapterInfo = {},
  adapterLimits = {},
  capabilities = null,
  features = [],
  runtime = {},
  qualityGovernor = null
}) {
  return {
    schemaVersion: DIAGNOSTICS_SCHEMA_VERSION,
    generatedAt,
    simulation: {
      grid: [...grid],
      pressureIterations,
      maxBacktraceDistance,
      tracerCount,
      workgroup: [...workgroup],
      settings: { ...settings },
      tracersEnabled
    },
    capabilities: capabilities
      ? {
          supported: capabilities.supported,
          featureTier: capabilities.featureTier,
          optionalFeatures: [...capabilities.optionalFeatures],
          requiredLimits: { ...capabilities.requiredLimits },
          failures: capabilities.failures.map((failure) => ({ ...failure }))
        }
      : null,
    adapter: {
      description: adapterInfo.description || null,
      device: adapterInfo.device || null,
      vendor: adapterInfo.vendor || null,
      architecture: adapterInfo.architecture || null,
      limits: supportedAdapterLimits(adapterLimits)
    },
    features: [...features],
    runtime: {
      ...runtime,
      qualityGovernor: qualityGovernor ? { ...qualityGovernor } : null
    }
  };
}
