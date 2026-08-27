/**
 * Small dependency-free runtime telemetry model for the browser preview.
 * CPU submission cost and optional GPU timestamps remain separate signals.
 */
function nonNegativeFinite(value, fallback = 0) {
  return Number.isFinite(value) && value >= 0 ? value : fallback;
}

export function createRuntimeTelemetry(startedAtMs = 0) {
  return {
    startedAtMs: nonNegativeFinite(startedAtMs),
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
  const safeFrameDeltaMs = nonNegativeFinite(frameDeltaMs);
  const safeCpuEncodeMs = nonNegativeFinite(cpuEncodeMs);
  telemetry.frameCount += 1;
  telemetry.lastFrameDeltaMs = safeFrameDeltaMs;
  telemetry.maximumFrameDeltaMs = Math.max(telemetry.maximumFrameDeltaMs, safeFrameDeltaMs);
  if (safeFrameDeltaMs > 1000 / 30) {
    telemetry.longFrames += 1;
  }
  telemetry.lastCpuEncodeMs = safeCpuEncodeMs;
  telemetry.averageCpuEncodeMs = telemetry.frameCount === 1
    ? safeCpuEncodeMs
    : telemetry.averageCpuEncodeMs * 0.9 + safeCpuEncodeMs * 0.1;
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
  const safeNowMs = nonNegativeFinite(nowMs, telemetry.startedAtMs);
  return {
    uptimeMs: Math.max(0, safeNowMs - telemetry.startedAtMs),
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
