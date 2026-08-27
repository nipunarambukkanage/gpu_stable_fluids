# Numerical method

The solver is a GPU-parallel adaptation of Jos Stam’s Stable Fluids method for a 2D incompressible field. The simulation grid is fixed at 512 × 512 and has unit spacing: one grid unit is one texel. The canvas is only a presentation surface and does not alter the simulation resolution.

## Coordinate convention

Texel centers are represented as:

```text
center = (integerX + 0.5, integerY + 0.5)
```

X increases to the right and Y increases downward. Pointer coordinates use the same convention after mapping from the CSS canvas rectangle. Velocity is stored as texels per second, so backtracing is dimensionally consistent:

```text
backTracedPosition = center - deltaTime × velocity(center)
```

The JavaScript frame coordinator clamps `deltaTime` to the interval `[1/1000, 1/30]`. Visibility changes clear the previous timestamp so returning to a visible page cannot produce a large simulation jump.

## State fields

- `u(x,y)` — velocity, stored as `rg32float`.
- `d(x,y)` — density/ink, stored as `rgba16float`; RGB carries color intensity and alpha carries coverage.
- `p(x,y)` — pressure, stored as `r32float`.
- `div(x,y)` — divergence, stored as `r32float`.
- `curl(x,y)` — optional scalar vorticity, stored as `r32float`.
- `xᵢ, ageᵢ` — optional GPU-resident Lagrangian tracer position and lifetime records.

Every field that is written from a previous value uses two textures. A pass reads one side and writes the other; the JavaScript index changes only after the pass has been encoded.

## 1. Splat injection

For each destination texel center `x`, the shader calculates the distance to the pointer segment from `s` to `e`:

```text
q = clamp(dot(x - s, e - s) / dot(e - s, e - s), 0, 1)
closest = s + q × (e - s)
distance = |x - closest|
```

The zero-length segment case uses point distance and avoids division by zero. The falloff is Gaussian-like:

```text
falloff = exp(-distance² / (2 × radius²))
```

The selected ink color and amount are added to density, while the pointer velocity is scaled by the velocity-force control and the same falloff. Density and velocity are clamped to finite HDR-safe ranges. When the pointer is active but stationary, start and end are equal, so ink continues to accumulate without inventing a large impulse.

## 2. Semi-Lagrangian advection

The destination cell backtraces through the current velocity field. Both velocity and density use manual bilinear interpolation. Manual sampling is important for `rg32float`, which should not be assumed to have hardware linear filtering.

For a position `q`, the shader clamps to valid half-texel bounds, identifies the four surrounding integer texels, and blends them in X and then Y. The sampled fields receive exponential dissipation:

```text
velocityDecay = exp(-velocityDissipation × deltaTime)
densityDecay  = exp(-inkDissipation × deltaTime)
```

This makes decay approximately frame-rate independent.

Both implementations also cap the backtrace displacement at 48 texels per step. This limiter prevents a long frame or an extreme injected velocity from sampling an unnecessarily distant cell, which reduces temporal popping while preserving the solver's semi-Lagrangian stability. The frame coordinator still clamps normal frame deltas before they reach the GPU.

## 3. Optional vorticity confinement

When the control is `Off`, this path is skipped and the base Stable Fluids sequence is unchanged. When enabled, the curl pass computes:

```text
curl = 0.5 × ((uRight.y - uLeft.y) - (uBottom.x - uTop.x))
```

The confinement pass estimates the gradient of `abs(curl)`, normalizes it safely, and applies a perpendicular force proportional to the local signed curl and the user-selected strength:

```text
normal = grad(abs(curl)) / max(|grad(abs(curl))|, epsilon)
force = strength × (normal.y, -normal.x) × curl
uNew = u + deltaTime × force
```

The force is clamped and outer-boundary normal velocity is zeroed. This is an optional visual enhancement rather than a replacement for pressure projection.

## 4. Divergence

The divergence uses centered finite differences:

```text
div = 0.5 × (uRight.x - uLeft.x + uBottom.y - uTop.y)
```

At solid boundaries, a sampled outside velocity mirrors the relevant normal component. This imposes a no-through-wall condition for the finite difference stencil.

## 5. Jacobi pressure solve

The pressure field is warm-started from the previous frame. Each configured iteration (20 in the default browser profile and native configuration) reads one pressure texture and writes the other:

```text
pNew = 0.25 × (pLeft + pRight + pTop + pBottom - div)
```

Pressure outside the domain is treated as the current edge pressure, which is a Neumann-like boundary behavior. The pressure index is swapped after every iteration. The code uses the tracked index rather than relying on iteration-count parity, which keeps the Performance and Cinematic browser profiles correct as well.

## 6. Projection

The final pressure gradient is centered:

```text
gradient = 0.5 × (pRight - pLeft, pBottom - pTop)
uProjected = u - gradient
```

The projection writes the opposite velocity texture and explicitly sets the normal component to zero on all four outer edges. Density is not modified during projection.

## 7. GPU Lagrangian tracers

The optional tracer overlay is advected from the projected velocity field after pressure projection. For each particle record, the compute shader samples the velocity at the particle’s nearest texel and performs a bounded explicit integration:

```text
xNew = x + deltaTime × 0.5 × u(x)
ageNew = age + deltaTime
```

The half-scale factor keeps the visualization stable when a high velocity crosses several texels in one frame. Particles that leave the valid half-texel domain or exceed their deterministic lifetime are respawned using their stored seed and simulation time. Two storage buffers are ping-ponged so the source record is never read and written in the same dispatch. The render pass uses instancing to expand each record into a small additive quad; particle positions never need a CPU readback.

## Rendering

The fragment shader samples the current density texture with a linear sampler. It maps intensity with a restrained exponential curve and normalizes chroma before mapping, which prevents dense colored ink from bleaching toward neutral white:

```text
intensity = max(density.r, density.g, density.b)
chroma = density / max(intensity, epsilon)
mappedIntensity = 1 - exp(-intensity × exposure)
displayColor = chroma × mappedIntensity
```

A gamma conversion prepares the result for the canvas presentation format. Alpha is always opaque.
