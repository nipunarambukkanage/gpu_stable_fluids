/**
 * Small dependency-free runtime telemetry model for the browser preview.
 * CPU submission cost and optional GPU timestamps remain separate signals.
 */
export function createRuntimeTelemetry(startedAtMs = 0) {
  return {
    startedAtMs,
    frameCount: 0,
    submittedFrames: 0,
    longFrames: 0,
    lastFrameDeltaMs: 0,
    maximumFrameDeltaMs: 0,
    lastCpuEncodeMs: 0,
    averageCpuEncodeMs: 0,
    gpuSamples: 0,
    lastGpuMs: null,
    averageGpuMs: 0
  };
}

export function resetRuntimeTelemetry(telemetry, startedAtMs = 0) {
  Object.assign(telemetry, createRuntimeTelemetry(startedAtMs));
}

export function recordRuntimeFrame(telemetry, frameDeltaMs, cpuEncodeMs) {
  telemetry.frameCount += 1;
  telemetry.lastFrameDeltaMs = frameDeltaMs;
  telemetry.maximumFrameDeltaMs = Math.max(telemetry.maximumFrameDeltaMs, frameDeltaMs);
  if (frameDeltaMs > 1000 / 30) {
    telemetry.longFrames += 1;
  }
  telemetry.lastCpuEncodeMs = cpuEncodeMs;
  telemetry.averageCpuEncodeMs = telemetry.frameCount === 1
    ? cpuEncodeMs
    : telemetry.averageCpuEncodeMs * 0.9 + cpuEncodeMs * 0.1;
}

export function recordRuntimeSubmission(telemetry) {
  telemetry.submittedFrames += 1;
}

export function recordGpuSample(telemetry, milliseconds) {
  if (!Number.isFinite(milliseconds) || milliseconds < 0) {
    return;
  }
  telemetry.gpuSamples += 1;
  telemetry.lastGpuMs = milliseconds;
  telemetry.averageGpuMs = telemetry.gpuSamples === 1
    ? milliseconds
    : telemetry.averageGpuMs * 0.85 + milliseconds * 0.15;
}

export function snapshotRuntimeTelemetry(telemetry, nowMs = telemetry.startedAtMs) {
  return {
    uptimeMs: Math.max(0, nowMs - telemetry.startedAtMs),
    frameCount: telemetry.frameCount,
    submittedFrames: telemetry.submittedFrames,
    longFrames: telemetry.longFrames,
    lastFrameDeltaMs: telemetry.lastFrameDeltaMs,
    maximumFrameDeltaMs: telemetry.maximumFrameDeltaMs,
    lastCpuEncodeMs: telemetry.lastCpuEncodeMs,
    averageCpuEncodeMs: telemetry.averageCpuEncodeMs,
    gpuSamples: telemetry.gpuSamples,
    lastGpuMs: telemetry.lastGpuMs,
    averageGpuMs: telemetry.averageGpuMs
  };
}
