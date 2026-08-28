# Diagnostics schema

`Save diagnostics JSON` produces a local report for reproducible bug reports and performance comparisons. The browser never uploads this file. The schema is versioned so tooling can reject or migrate reports explicitly rather than guessing their shape.

## Version 1

```json
{
  "schemaVersion": 1,
  "generatedAt": "ISO-8601 timestamp",
  "simulation": {
    "grid": [512, 512],
    "pressureIterations": 20,
    "maxBacktraceDistance": 48,
    "tracerCount": 8192,
    "workgroup": [16, 16, 1],
    "settings": {
      "qualityProfile": "balanced",
      "pressureIterations": 20
    },
    "tracersEnabled": true
  },
  "capabilities": {
    "supported": true,
    "featureTier": "baseline|timestamp-query",
    "optionalFeatures": [],
    "requiredLimits": {},
    "failures": []
  },
  "adapter": {
    "description": "string|null",
    "device": "string|null",
    "vendor": "string|null",
    "architecture": "string|null",
    "limits": {}
  },
  "features": [],
  "runtime": {
    "uptimeMs": 0,
    "frameCount": 0,
    "submittedFrames": 0,
    "longFrames": 0,
    "lastFrameDeltaMs": 0,
    "maximumFrameDeltaMs": 0,
    "lastCpuEncodeMs": 0,
    "averageCpuEncodeMs": 0,
    "gpuSamples": 0,
    "lastGpuMs": null,
    "averageGpuMs": 0,
    "qualityGovernor": {
      "enabled": false,
      "profile": "balanced",
      "pressureIterations": 20,
      "targetGpuMs": 16.7,
      "lastGpuMs": null
    }
  }
}
```

The `runtime.qualityGovernor` object records whether adaptive quality was enabled, the active profile, the pressure iteration count, the target GPU budget, and the latest observed timestamp. This makes performance reports explainable when two runs use different pressure workloads.

## Compatibility policy

Consumers must read `schemaVersion` first. New additive fields may appear in a later minor revision. A breaking field rename or semantic change requires a new schema version. Adapter limits are filtered to compute and storage-related values to keep reports useful without collecting unrelated platform data.
