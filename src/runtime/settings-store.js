import { BRUSH_MODES, DEFAULT_BRUSH_MODE, DEFAULT_RENDER_MODE, DEFAULT_QUALITY_PROFILE, QUALITY_PROFILES, RENDER_MODES } from "../config/simulation.js";

export const SETTINGS_SCHEMA_VERSION = 1;
export const SETTINGS_STORAGE_KEY = "gpu-stable-fluids.lab-settings.v1";

const DEFAULTS = Object.freeze({
  inkColor: "#48b7ff",
  brushRadius: 18,
  velocityForce: 1,
  inkAmount: 1.8,
  velocityDissipation: 0.08,
  inkDissipation: 0.025,
  vorticityConfinement: 0,
  scenePreset: "studio",
  qualityProfile: DEFAULT_QUALITY_PROFILE,
  renderMode: DEFAULT_RENDER_MODE,
  brushMode: DEFAULT_BRUSH_MODE,
  tracersEnabled: true,
  hudEnabled: false,
  adaptiveQuality: false
});
const SCENE_PRESETS = new Set(["studio", "soft", "long-trails", "turbulent"]);

function clamp(value, minimum, maximum, fallback) {
  return Number.isFinite(value) ? Math.min(maximum, Math.max(minimum, value)) : fallback;
}

export function sanitizeSettings(input = {}) {
  const source = input && typeof input === "object" ? input : {};
  return {
    inkColor: typeof source.inkColor === "string" && /^#[0-9a-f]{6}$/i.test(source.inkColor) ? source.inkColor.toLowerCase() : DEFAULTS.inkColor,
    brushRadius: clamp(Number(source.brushRadius), 4, 64, DEFAULTS.brushRadius),
    velocityForce: clamp(Number(source.velocityForce), 0.1, 2.4, DEFAULTS.velocityForce),
    inkAmount: clamp(Number(source.inkAmount), 0.1, 4, DEFAULTS.inkAmount),
    velocityDissipation: clamp(Number(source.velocityDissipation), 0, 1, DEFAULTS.velocityDissipation),
    inkDissipation: clamp(Number(source.inkDissipation), 0, 1, DEFAULTS.inkDissipation),
    vorticityConfinement: clamp(Number(source.vorticityConfinement), 0, 1.5, DEFAULTS.vorticityConfinement),
    scenePreset: typeof source.scenePreset === "string" && SCENE_PRESETS.has(source.scenePreset) ? source.scenePreset : DEFAULTS.scenePreset,
    qualityProfile: QUALITY_PROFILES[source.qualityProfile] ? source.qualityProfile : DEFAULTS.qualityProfile,
    renderMode: RENDER_MODES[source.renderMode] ? source.renderMode : DEFAULT_RENDER_MODE,
    brushMode: BRUSH_MODES[source.brushMode] ? source.brushMode : DEFAULT_BRUSH_MODE,
    tracersEnabled: source.tracersEnabled !== false,
    hudEnabled: source.hudEnabled === true,
    adaptiveQuality: source.adaptiveQuality === true
  };
}

export function saveSettings(storage, settings) {
  if (!storage) return false;
  try {
    storage.setItem(SETTINGS_STORAGE_KEY, JSON.stringify({ schemaVersion: SETTINGS_SCHEMA_VERSION, settings: sanitizeSettings(settings) }));
    return true;
  } catch {
    return false;
  }
}

export function loadSettings(storage) {
  if (!storage) return null;
  try {
    const raw = storage.getItem(SETTINGS_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed?.schemaVersion === SETTINGS_SCHEMA_VERSION ? sanitizeSettings(parsed.settings) : null;
  } catch {
    return null;
  }
}

export function clearSettings(storage) {
  if (!storage) return false;
  try {
    storage.removeItem(SETTINGS_STORAGE_KEY);
    return true;
  } catch {
    return false;
  }
}
