# Architecture

This project intentionally keeps the executable surface in `fluid-simulation.html`. That file is the portable artifact: it contains the semantic interface, responsive CSS, application state, WGSL source strings, WebGPU initialization, resource creation, input handling, simulation passes, rendering, recovery, and startup. The `docs/` directory holds engineering documentation without introducing runtime dependencies.

## Runtime layers

```text
HTML / CSS interface
        │
        ├── control state and accessibility updates
        ├── pointer samples → simulation-space stroke segment
        └── resize / visibility / recovery events
                         │
                  JavaScript frame coordinator
                         │
        ┌────────────────┴────────────────┐
        │                                 │
  cached GPU resources              presentation canvas
        │                                 │
  WGSL compute passes                fullscreen render pass
        │
  velocity / density / pressure / divergence / vorticity
```

## Resource ownership

`app.resources` owns the complete initialized GPU graph:

| Resource | Count | Format | Purpose |
| --- | ---: | --- | --- |
| Velocity | 2 | `rg32float` | Horizontal and vertical velocity in texels/second. |
| Density | 2 | `rgba16float` | RGB ink intensity and alpha coverage. |
| Pressure | 2 | `r32float` | Jacobi pressure ping-pong state. |
| Divergence | 1 | `r32float` | Centered velocity divergence. |
| Vorticity | 1 | `r32float` | Optional scalar curl field. |
| Uniform buffer | 1 | 112 bytes | Per-frame parameters with seven 16-byte slots. |

All simulation textures use `TEXTURE_BINDING`, `STORAGE_BINDING`, and `COPY_DST`. The fixed simulation grid is 512 × 512; presentation dimensions are independent and may change with CSS size and device pixel ratio.

## Bind-group strategy

Bind groups are built once after texture views and layouts exist. The cached combinations are:

- Splat and advection: two read-index variants, each reading one velocity/density pair and writing the other.
- Divergence: two velocity read-index variants.
- Pressure: two pressure read-index variants, always reading the single divergence texture and writing the opposite pressure side.
- Gradient: four combinations of final pressure index and velocity read index.
- Vorticity: two velocity read-index variants.
- Confinement: two velocity read-index variants, reading the single vorticity texture and writing the opposite velocity side.
- Render: two density read-index variants.

No simulation texture is sampled and storage-written in the same pass. The JavaScript index fields make the current readable side explicit and are swapped only after encoding the pass that writes the opposite side.

## Frame coordinator

`encodeSimulationFrame()` performs one command encoder and one queue submission for a normal frame. The five base compute stages are always ordered as splat, advection, divergence, pressure, and gradient. When vorticity strength is non-zero, curl and confinement are inserted between advection and divergence. The pressure loop encodes exactly 20 separate compute passes in the same command encoder.

The loop does not map buffers, read textures back to the CPU, wait for GPU completion, or create pipelines/bind groups. Only the per-frame uniform upload and transient encoder/pass descriptors occur in the hot path.

## Tiled stencil execution

Divergence, pressure, and vorticity use CUDA-style workgroup tiling. Each 16 × 16 workgroup stages an 18 × 18 core-plus-halo tile in `var<workgroup>` memory, reaches `workgroupBarrier()`, and then evaluates the neighbor stencil from the tile. This keeps the synchronization local to each workgroup while avoiding repeated global neighbor loads inside those kernels. See [CUDA_TO_WEBGPU.md](CUDA_TO_WEBGPU.md) for the portable API mapping and limitations.

## Initialization and recovery

Initialization follows this sequence:

1. Validate secure context and `navigator.gpu`.
2. Request a high-performance adapter, then retry the default adapter preference if needed.
3. Request a device without optional features.
4. Acquire the WebGPU canvas context and preferred presentation format.
5. Register uncaptured-error and device-loss handlers.
6. Create the uniform buffer, textures, views, sampler, layouts, shader modules, pipeline layouts, pipelines, and bind groups.
7. Configure the presentation canvas and zero every simulation resource.
8. Start either a reduced-motion paused state or the animation loop.

Initialization is guarded by `app.initializing`, and the animation loop is guarded by `app.animationRunning`. Device loss cancels the loop, releases old resources, disables GPU controls, presents the reason, and exposes `Retry GPU`. A successful retry reconstructs the entire resource graph and resets the field.

## UI and input ownership

The control panel owns only browser/UI state. Numeric values are copied into `simulationSettings` when inputs change, which avoids parsing DOM strings in every frame. The uniform upload mirrors that cache into the documented 112-byte GPU layout.

Pointer Events are converted from the canvas CSS rectangle to simulation coordinates. The pointer state retains the last submitted segment start and the latest end point, allowing multiple browser events between frames to become one continuous splat segment. Coalesced events improve stylus/mouse fidelity when available. Demo input uses the same pointer state, so it exercises the same GPU path as a human stroke.

## Extension guidelines

When adding a new simulation field:

1. Add the texture and zero data path.
2. Add its view and a dedicated bind-group layout.
3. Add a complete WGSL module with invocation bounds checks.
4. Cache its shader, pipeline layout, pipeline, and bind groups during initialization.
5. Insert its pass only where resource usage and numerical ordering remain explicit.
6. Update `clearSimulationTextures()`, the README resource table, and the validation checklist.

Avoid adding per-frame resource creation, CPU readback, hidden coordinate conversions, or a second animation scheduler.
