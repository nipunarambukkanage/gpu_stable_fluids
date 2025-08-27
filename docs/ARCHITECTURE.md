# Architecture

The native CUDA C++ target under cuda/ is the primary high-performance execution path. The browser modules remain a portable WebGPU preview with a parallel resource graph, while the native solver owns CUDA Runtime API allocations, streams, events, shared-memory kernels, and CPU-coordinated frame export.

The project uses `fluid-simulation.html` as a stable browser shell and `src/main.js` as the composition root. The shell owns semantic markup and module/style references; `styles/fluid-lab.css` owns presentation; `src/config/simulation.js` owns shared workload constants; `src/gpu/shaders.js` owns WGSL source strings; `src/gpu/timestamp-profiler.js` owns optional GPU timing; and `src/main.js` owns application state, WebGPU initialization, resource creation, input handling, simulation passes, rendering, recovery, and startup. The `docs/`, `tests/`, and `tools/` directories hold engineering support without runtime dependencies.

## Runtime layers

```text
HTML shell + external CSS
        │
        ├── control state and accessibility updates
        ├── pointer samples → simulation-space stroke segment
        └── resize / visibility / recovery events
                         │
                  `src/main.js` frame coordinator
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
| Tracer particles | 2 | 16-byte storage records | GPU-resident positions, ages, and deterministic seeds. |
| Indirect draw arguments | 1 | 16-byte draw record | GPU-generated vertex/instance counts for tracer rendering. |
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
- Tracer compute: four combinations of velocity read index and particle source index, writing the opposite particle buffer.
- Render: two density read-index variants.
- Tracer render: two particle read-index variants, using instanced quads.
- Indirect arguments: one storage bind group for the device-generated four-word draw command.

No simulation texture is sampled and storage-written in the same pass. The JavaScript index fields make the current readable side explicit and are swapped only after encoding the pass that writes the opposite side.

## Frame coordinator

`encodeSimulationFrame()` performs one command encoder and one queue submission for a normal frame. The five base compute stages are always ordered as splat, advection, divergence, pressure, and gradient. When vorticity strength is non-zero, curl and confinement are inserted between advection and divergence. When tracers are enabled, a 64-thread particle advection dispatch runs after the final projected velocity is available and before the render pass. The pressure loop encodes exactly 20 separate compute passes in the same command encoder.

The loop does not map buffers, read textures back to the CPU, wait for GPU completion, or create pipelines/bind groups. Only the per-frame uniform upload and transient encoder/pass descriptors occur in the hot path. On adapters exposing `timestamp-query`, one sampled frame every 30 iterations adds two timestamp writes and an asynchronous readback request; all other frames remain unchanged.

When tracers are enabled, a one-invocation compute pass refreshes the indirect draw record before the tracer workload and render pass. This command data remains device-local and does not require a CPU readback.

## Tiled stencil execution

Divergence, pressure, and vorticity use CUDA-style workgroup tiling. Each 16 × 16 workgroup stages an 18 × 18 core-plus-halo tile in `var<workgroup>` memory, reaches `workgroupBarrier()`, and then evaluates the neighbor stencil from the tile. This keeps the synchronization local to each workgroup while avoiding repeated global neighbor loads inside those kernels. See [CUDA_TO_WEBGPU.md](CUDA_TO_WEBGPU.md) for the portable API mapping and limitations.

## GPU-resident tracers

The optional tracer stage is a separate GPU workload rather than a CPU visualization layer. Two storage buffers contain 8,192 records of `{ position, age, seed }`. The compute pass samples the current projected velocity texture, advances all records with bounded explicit integration, respawns escaped particles, and swaps the source/destination buffer index. The render pass then draws six vertices per instance with an additive blend pipeline. No particle position crosses the GPU/CPU boundary during normal operation.

The indirect argument record stores vertex count, instance count, first vertex, and first instance. Keeping this command data GPU-generated leaves a clean extension point for future GPU culling or particle compaction.

## Optional GPU timing

Initialization negotiates the optional `timestamp-query` feature only when the adapter advertises it. The timer owns a two-query timestamp set, a 16-byte resolve buffer, and a 16-byte `MAP_READ` buffer. A sampled command encoder writes timestamps immediately before and after the full simulation/render sequence, resolves the pair, copies the result, and submits normally. A later asynchronous task waits for submitted work, maps the small readback, converts the nanosecond delta to milliseconds, and releases the mapping. Device teardown destroys the timer resources with the rest of the GPU graph.

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
