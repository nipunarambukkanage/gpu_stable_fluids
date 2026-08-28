/**
 * Stable simulation and GPU workload constants shared by the browser entry point.
 * Keeping these values in one module prevents shader, resource, and validation drift.
 */
export const GRID_WIDTH = 512;
export const GRID_HEIGHT = 512;
export const WORKGROUP_SIZE = 16;
export const WORKGROUPS_X = Math.ceil(GRID_WIDTH / WORKGROUP_SIZE);
export const WORKGROUPS_Y = Math.ceil(GRID_HEIGHT / WORKGROUP_SIZE);
export const PRESSURE_ITERATIONS = 20;
export const QUALITY_PROFILES = Object.freeze({
  performance: Object.freeze({ label: "Performance", pressureIterations: 8, description: "Lower pressure cost for high frame rates." }),
  balanced: Object.freeze({ label: "Balanced", pressureIterations: PRESSURE_ITERATIONS, description: "Balanced pressure quality and GPU cost." }),
  cinematic: Object.freeze({ label: "Cinematic", pressureIterations: 36, description: "Higher pressure quality for clean dense curls." })
});
export const DEFAULT_QUALITY_PROFILE = "balanced";
export const RENDER_MODES = Object.freeze({
  density: Object.freeze({ label: "Density", description: "Tonemapped ink density field." }),
  velocity: Object.freeze({ label: "Velocity", description: "Velocity magnitude and direction diagnostics." }),
  pressure: Object.freeze({ label: "Pressure", description: "Signed pressure projection field." }),
  divergence: Object.freeze({ label: "Divergence", description: "Residual compressibility diagnostic." }),
  vorticity: Object.freeze({ label: "Vorticity", description: "Curl magnitude diagnostic." })
});
export const DEFAULT_RENDER_MODE = "density";
export const BRUSH_MODES = Object.freeze({
  paint: Object.freeze({ label: "Paint + velocity", value: 0, description: "Inject ink and momentum." }),
  velocity: Object.freeze({ label: "Velocity only", value: 1, description: "Inject momentum without adding ink." }),
  erase: Object.freeze({ label: "Erase", value: 2, description: "Dampen ink and momentum under the brush." })
});
export const DEFAULT_BRUSH_MODE = "paint";
export const MAX_FRAME_DELTA = 1 / 30;
export const MAX_BACKTRACE_DISTANCE = 48;
export const MAX_POINTER_VELOCITY = 1800;
export const MAX_DENSITY = 12;
export const MAX_EFFECTIVE_DPR = 2;
export const PARTICLE_COUNT = 8192;
export const PARTICLE_WORKGROUP_SIZE = 64;
export const PARTICLE_WORKGROUPS = Math.ceil(PARTICLE_COUNT / PARTICLE_WORKGROUP_SIZE);
export const PARTICLE_STRIDE_FLOATS = 4;
export const PARTICLE_BUFFER_SIZE = PARTICLE_COUNT * PARTICLE_STRIDE_FLOATS * Float32Array.BYTES_PER_ELEMENT;
export const PARTICLE_DRAW_VERTEX_COUNT = 6;
export const INDIRECT_ARGS_SIZE = 4 * Uint32Array.BYTES_PER_ELEMENT;
export const UNIFORM_FLOAT_COUNT = 28;
export const UNIFORM_BUFFER_SIZE = UNIFORM_FLOAT_COUNT * Float32Array.BYTES_PER_ELEMENT;

export function createInitialParticleData() {
  const data = new Float32Array(PARTICLE_COUNT * PARTICLE_STRIDE_FLOATS);
  for (let particleIndex = 0; particleIndex < PARTICLE_COUNT; particleIndex += 1) {
    const seed = (Math.imul(particleIndex, 1664525) + 1013904223) >>> 0;
    const secondSeed = (Math.imul(seed, 22695477) + 1) >>> 0;
    const offset = particleIndex * PARTICLE_STRIDE_FLOATS;
    data[offset] = 0.5 + (seed / 4294967296) * (GRID_WIDTH - 1);
    data[offset + 1] = 0.5 + (secondSeed / 4294967296) * (GRID_HEIGHT - 1);
    data[offset + 2] = ((particleIndex % 240) / 240) * 4;
    data[offset + 3] = seed / 4294967296;
  }
  return data;
}
