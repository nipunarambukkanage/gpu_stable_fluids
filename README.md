# Zero-Cost Real-Time WebGPU Stable Fluids Lab

An interactive 2D incompressible fluid simulator that runs entirely in the browser on the user’s local GPU. The project implements Jos Stam’s Stable Fluids approach with WebGPU compute shaders and WGSL, with a fixed 512 × 512 simulation grid and a responsive presentation canvas.

There is no backend, package manager, build step, API key, cloud service, analytics, or external asset. The application is intentionally delivered as one self-contained HTML file so it can be copied, archived, or served from any simple local static server.

## What is included

- GPU-resident velocity, density, pressure, divergence, and optional vorticity fields.
- Semi-Lagrangian advection with manual bilinear interpolation for `rg32float` velocity textures.
- Incompressible projection using centered divergence, exactly 20 Jacobi pressure iterations, and pressure-gradient subtraction.
- Continuous pointer strokes using Pointer Events, pointer capture, coalesced events, pressure-safe timing, and a segment-distance splat falloff.
- Optional vorticity confinement for richer curls without changing the default solver path.
- Preset scenes for studio, soft ink, long trails, and turbulent ribbon looks.
- Auto-demo figure-eight flow for hands-off presentation.
- Local PNG snapshots generated from the presentation canvas; nothing is uploaded.
- Pause/resume, clear/reset, hidden controls, responsive sizing, high-DPI support, visibility-aware timing, and device-loss recovery.
- Accessible labels, keyboard shortcuts, visible focus states, readable status reporting, and reduced-motion startup behavior.

## Project structure

```text
gpu_stable_fluids/
├── fluid-simulation.html       # Complete runnable application: HTML, CSS, JS, and WGSL
├── README.md                   # Project overview, usage, design decisions, and maintenance notes
└── docs/
    ├── ARCHITECTURE.md         # Runtime components, resource graph, lifecycle, and invariants
    ├── CUDA_TO_WEBGPU.md       # CUDA concept mapping, tiled kernels, and portability boundaries
    ├── NUMERICAL_METHOD.md     # Stable Fluids equations, coordinate conventions, and GPU passes
    └── VALIDATION.md            # Static checks, browser smoke tests, and regression checklist
```

The runtime deliberately has no `node_modules`, `package.json`, bundler, generated assets, or hidden build output. The documentation is separated into `docs/` so the implementation remains portable while the engineering details remain easy to navigate.

## Quick start

WebGPU is normally available only in a secure context. From this directory, use any free local static server. If Python is already installed:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/fluid-simulation.html
```

No installation is required by this project. Opening the HTML directly may work in some browsers, but `localhost` is the reliable option for WebGPU’s secure-context requirement.

## Browser and hardware target

Use a current desktop Chromium-based browser such as Chrome or Edge with WebGPU enabled. The app checks `window.isSecureContext`, `navigator.gpu`, adapter availability, canvas context creation, shader compilation, pipeline creation, and device loss. If initialization fails, the UI shows a human-readable reason and exposes a `Retry GPU` action.

The application does not provide a WebGL fallback. This is intentional: the numerical path is designed around WebGPU compute and the fallback message explains how to use a supported browser or `localhost`.

## Controls

| Control | Effect |
| --- | --- |
| Ink color | Select any color or use one of the four accessible presets. |
| Brush radius | Sets the soft splat radius in simulation texels. |
| Ink injection | Sets the amount of color and density added per active frame. |
| Velocity force | Scales the drag velocity injected into the field. |
| Velocity dissipation | Exponential velocity decay rate. Lower values preserve motion. |
| Ink dissipation | Exponential density decay rate. Lower values preserve trails. |
| Scene preset | Applies a coherent look and clears the current field. |
| Vortex confinement | Optional curl enhancement. `Off` leaves the base solver unchanged. |
| Pause / Resume | Stops simulation submissions while preserving the current density. |
| Clear | Zeros every simulation texture and resets pointer/timing state. |
| Auto demo | Runs a local procedural figure-eight stroke. Dragging takes control back. |
| Save PNG | Saves the visible canvas locally as a PNG file. |

Keyboard shortcuts:

- `Space` — pause or resume.
- `C` or `R` — clear the simulation.
- `H` — hide or show the control panel.

Shortcuts are ignored while a range or color input is being edited.

## GPU architecture

The simulation grid never changes size. It is always 512 × 512 texels and dispatches 32 × 32 workgroups of 16 × 16 threads. Only the presentation canvas is resized to the CSS size and a capped device-pixel ratio of 2.

The GPU resources are created once during initialization and reused:

- Two `rg32float` velocity textures.
- Two `rgba16float` density textures containing RGB ink and alpha coverage.
- Two `r32float` pressure textures.
- One `r32float` divergence texture.
- One `r32float` vorticity texture used by the optional curl enhancement.
- One 112-byte uniform buffer, cached shader modules, pipeline layouts, pipelines, samplers, texture views, and bind groups.

Every read/write stage uses distinct ping-pong sides. The JavaScript resource state keeps the currently readable velocity, density, and pressure index explicit. Bind groups are created for every legal index combination during initialization; they are never recreated in the animation loop.

The normal active-frame order is:

```text
uniform upload
  → splat injection
  → velocity/density swap
  → semi-Lagrangian advection
  → velocity/density swap
  → optional vorticity + confinement
  → divergence
  → 20 Jacobi pressure passes
  → pressure-gradient subtraction
  → velocity swap
  → full-screen density render
  → one queue submission
```

Each Jacobi iteration is a separate compute pass inside the same command encoder so WebGPU usage scopes do not alias a sampled pressure texture with its storage destination. There is no GPU readback or `queue.onSubmittedWorkDone()` in the normal loop.

### CUDA-inspired GPU techniques (browser-safe)

The solver now uses a CUDA-like tiled stencil design while remaining a zero-install browser application. Divergence, pressure, and vorticity kernels cooperatively stage an 18 × 18 tile in `var<workgroup>` memory: 16 × 16 invocations cover the interior and the surrounding one-texel halo supplies neighbor values. A `workgroupBarrier()` then makes the tile visible before the stencil is evaluated. This reduces repeated global texture reads for the neighbor-heavy passes and mirrors the structure of a CUDA `__shared__` tile plus `__syncthreads()`.

The page also reports adapter identity and compute limits, wraps initialization in a WebGPU validation error scope, and handles uncaptured validation errors and device loss. These are WebGPU equivalents of the diagnostics and capability checks expected in a native GPU application.

This is CUDA-inspired, not a CUDA runtime. A normal web page cannot execute `.cu` kernels, call cuBLAS/cuFFT, inspect CUDA occupancy counters, or use NVIDIA-only warp intrinsics. The actual implementation is WGSL/WebGPU so it remains portable across supported browser GPUs. The exact concept mapping and a native CUDA porting sketch are in [docs/CUDA_TO_WEBGPU.md](docs/CUDA_TO_WEBGPU.md).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the resource graph and recovery lifecycle.

## Numerical conventions

Simulation coordinates use texel units with texel centers at `(x + 0.5, y + 0.5)`. Pointer coordinates map from the CSS canvas rectangle directly into the 512 × 512 domain, with Y increasing downward in every pass. Velocity is stored in texels per second. Frame delta is clamped to 1/30 second and reset after visibility changes.

Advection uses manual bilinear sampling because the velocity field is `rg32float` and must not depend on hardware filtering. Dissipation uses the frame-rate-independent form:

```text
decay = exp(-dissipationRate × deltaTime)
```

The pressure solve uses Neumann-like boundary samples, while velocity boundary samples mirror the normal component and the final projection explicitly zeros normal velocity on the outer grid. Full equations and coordinate details are in [docs/NUMERICAL_METHOD.md](docs/NUMERICAL_METHOD.md).

## Reliability and privacy

- Device loss stops the animation loop before resource references are released.
- Retry rebuilds the adapter, device, canvas configuration, textures, layouts, pipelines, bind groups, and zeroed state.
- Uncaptured validation errors are surfaced in the status panel and console.
- Large frame gaps, pointer re-entry, and long pointer-event gaps cannot create unbounded velocity impulses.
- PNG export uses a browser object URL and is revoked immediately after the local download is triggered.
- No network request, telemetry, tracking, advertisement, external font, image, shader, or package is used at runtime.

## Validation performed

The implementation has been checked with JavaScript parsing, diff whitespace checks, static requirement assertions, and a Chrome WebGPU smoke test. The browser test covered initialization, shader compilation, pipeline creation, normal drag painting, color preset selection, turbulent/vorticity mode, auto-demo, pause/resume, clear, responsive layout, and console validation errors.

Use [docs/VALIDATION.md](docs/VALIDATION.md) for the repeatable checklist and maintenance guidance.

## License and cost

The repository contains no paid service integration or third-party runtime dependency. It uses native browser APIs and the user’s existing hardware only. Add a project license separately if this code is redistributed under a specific legal license.
