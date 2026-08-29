export const INPUT_RECORDING_SCHEMA_VERSION = 1;

function finiteOr(value, fallback = 0) {
  return Number.isFinite(value) ? value : fallback;
}

function sanitizeSample(sample) {
  return {
    timeMs: Math.max(0, finiteOr(sample?.timeMs)),
    x: finiteOr(sample?.x),
    y: finiteOr(sample?.y),
    velocityX: finiteOr(sample?.velocityX),
    velocityY: finiteOr(sample?.velocityY),
    active: Boolean(sample?.active)
  };
}

export function createInputRecorder({ maxSamples = 8_192 } = {}) {
  const state = { active: false, startedAtMs: 0, samples: [], maxSamples: Math.max(1, Math.floor(maxSamples)) };
  return {
    start(nowMs = 0) {
      state.active = true;
      state.startedAtMs = finiteOr(nowMs);
      state.samples.length = 0;
    },
    stop() {
      state.active = false;
    },
    isRecording() {
      return state.active;
    },
    sampleCount() {
      return state.samples.length;
    },
    record(sample, nowMs = 0) {
      if (!state.active || state.samples.length >= state.maxSamples) {
        return false;
      }
      const normalized = sanitizeSample(sample);
      normalized.timeMs = Math.max(0, finiteOr(nowMs) - state.startedAtMs);
      state.samples.push(normalized);
      return true;
    },
    snapshot() {
      return {
        schemaVersion: INPUT_RECORDING_SCHEMA_VERSION,
        durationMs: state.samples.at(-1)?.timeMs || 0,
        samples: state.samples.map((sample) => ({ ...sample }))
      };
    }
  };
}

export function validateInputRecording(recording) {
  if (!recording || recording.schemaVersion !== INPUT_RECORDING_SCHEMA_VERSION || !Array.isArray(recording.samples)) {
    return null;
  }
  const samples = recording.samples.map(sanitizeSample).filter((sample, index, list) => index === 0 || sample.timeMs >= list[index - 1].timeMs);
  return samples.length > 0
    ? { schemaVersion: INPUT_RECORDING_SCHEMA_VERSION, durationMs: samples.at(-1).timeMs, samples }
    : null;
}
