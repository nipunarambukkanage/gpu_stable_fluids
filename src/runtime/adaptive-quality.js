import { QUALITY_PROFILES, DEFAULT_QUALITY_PROFILE } from "../config/simulation.js";

const PROFILE_ORDER = Object.freeze(["performance", "balanced", "cinematic"]);

function finiteOr(value, fallback) {
  return Number.isFinite(value) ? value : fallback;
}

/**
 * Hysteretic quality governor driven by completed GPU timestamp samples.
 * It changes pressure work only after repeated evidence and a cooldown, so
 * transient spikes do not cause visible profile oscillation.
 */
export function createAdaptiveQualityGovernor({
  initialProfile = DEFAULT_QUALITY_PROFILE,
  targetGpuMs = 16.7,
  downshiftRatio = 1.2,
  upshiftRatio = 0.72,
  downshiftSampleThreshold = 2,
  upshiftSampleThreshold = 4,
  cooldownMs = 2_000
} = {}) {
  const state = {
    currentProfile: QUALITY_PROFILES[initialProfile] ? initialProfile : DEFAULT_QUALITY_PROFILE,
    targetGpuMs: Math.max(1, finiteOr(targetGpuMs, 16.7)),
    downshiftRatio: Math.max(1, finiteOr(downshiftRatio, 1.2)),
    upshiftRatio: Math.min(0.99, Math.max(0.1, finiteOr(upshiftRatio, 0.72))),
    downshiftSampleThreshold: Math.max(1, Math.floor(finiteOr(downshiftSampleThreshold, 2))),
    upshiftSampleThreshold: Math.max(1, Math.floor(finiteOr(upshiftSampleThreshold, 4))),
    cooldownMs: Math.max(0, finiteOr(cooldownMs, 2_000)),
    lastChangeAtMs: Number.NEGATIVE_INFINITY,
    highSamples: 0,
    lowSamples: 0,
    lastGpuMs: null
  };

  function setProfile(profileName, nowMs = 0) {
    if (!QUALITY_PROFILES[profileName]) {
      return false;
    }
    state.currentProfile = profileName;
    state.highSamples = 0;
    state.lowSamples = 0;
    state.lastChangeAtMs = finiteOr(nowMs, 0);
    return true;
  }

  function observe(gpuMs, nowMs = 0) {
    const safeGpuMs = finiteOr(gpuMs, Number.NaN);
    const safeNowMs = finiteOr(nowMs, 0);
    if (!Number.isFinite(safeGpuMs) || safeGpuMs < 0) {
      return null;
    }
    state.lastGpuMs = safeGpuMs;
    if (safeNowMs - state.lastChangeAtMs < state.cooldownMs) {
      return null;
    }

    if (safeGpuMs > state.targetGpuMs * state.downshiftRatio) {
      state.highSamples += 1;
      state.lowSamples = 0;
    } else if (safeGpuMs < state.targetGpuMs * state.upshiftRatio) {
      state.lowSamples += 1;
      state.highSamples = 0;
    } else {
      state.highSamples = 0;
      state.lowSamples = 0;
    }

    const currentIndex = PROFILE_ORDER.indexOf(state.currentProfile);
    let nextIndex = currentIndex;
    if (state.highSamples >= state.downshiftSampleThreshold && currentIndex > 0) {
      nextIndex = currentIndex - 1;
    } else if (state.lowSamples >= state.upshiftSampleThreshold && currentIndex < PROFILE_ORDER.length - 1) {
      nextIndex = currentIndex + 1;
    }
    if (nextIndex === currentIndex) {
      return null;
    }

    const previousProfile = state.currentProfile;
    setProfile(PROFILE_ORDER[nextIndex], safeNowMs);
    return {
      previousProfile,
      profile: state.currentProfile,
      pressureIterations: QUALITY_PROFILES[state.currentProfile].pressureIterations,
      reason: nextIndex < currentIndex ? "gpu-budget" : "headroom",
      gpuMs: safeGpuMs
    };
  }

  return {
    setProfile,
    observe,
    snapshot() {
      return {
        enabled: true,
        profile: state.currentProfile,
        pressureIterations: QUALITY_PROFILES[state.currentProfile].pressureIterations,
        targetGpuMs: state.targetGpuMs,
        lastGpuMs: state.lastGpuMs,
        highSamples: state.highSamples,
        lowSamples: state.lowSamples
      };
    }
  };
}
