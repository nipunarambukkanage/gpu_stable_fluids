/**
 * Shared visual presets. Keeping these outside the composition root lets
 * settings validation, UI automation, and future native tooling use the same
 * names and numeric ranges.
 */
export const SCENE_PRESETS = Object.freeze({
  studio: Object.freeze({ color: "#48b7ff", brushRadius: 18, inkAmount: 1.8, velocityForce: 1, velocityDissipation: 0.08, inkDissipation: 0.025, vorticity: 0 }),
  soft: Object.freeze({ color: "#a7ff83", brushRadius: 26, inkAmount: 1.15, velocityForce: 0.65, velocityDissipation: 0.16, inkDissipation: 0.04, vorticity: 0 }),
  "long-trails": Object.freeze({ color: "#ffc857", brushRadius: 12, inkAmount: 1.4, velocityForce: 1.35, velocityDissipation: 0.025, inkDissipation: 0.006, vorticity: 0.35 }),
  turbulent: Object.freeze({ color: "#ff5fb3", brushRadius: 20, inkAmount: 1.65, velocityForce: 1.45, velocityDissipation: 0.045, inkDissipation: 0.018, vorticity: 0.9 })
});

export const SCENE_PRESET_NAMES = Object.freeze(Object.keys(SCENE_PRESETS));
