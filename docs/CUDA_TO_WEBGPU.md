# CUDA concepts translated to WebGPU

This document explains how the fluid lab uses CUDA-style GPU engineering ideas without pretending that a browser page is running CUDA. The executable implementation is a dependency-free HTML shell plus ES modules under `src/`; the WGSL catalog lives in `src/gpu/shaders.js` and the runtime uses WebGPU.

## Capability boundary

WebGPU is a browser API with portable compute and graphics access. It does not expose the CUDA runtime, `.cu` compilation, CUDA streams, cuBLAS, cuFFT, NVIDIA warp intrinsics, Nsight counters, or a direct CUDA occupancy API. The lab therefore uses the closest portable WebGPU mechanisms and keeps the numerical solver self-contained.

| CUDA concept | WebGPU/WGSL equivalent in this lab |
| --- | --- |
| `threadIdx.x/y` | `@builtin(local_invocation_id)` |
| `blockIdx.x/y` | `@builtin(workgroup_id)` |
| `blockDim.x/y` | `@workgroup_size(16, 16, 1)` |
| Kernel grid launch | `dispatchWorkgroups(32, 32, 1)` for the fixed 512 × 512 domain |
| `__shared__` memory | `var<workgroup>` tile arrays |
| `__syncthreads()` | `workgroupBarrier()` |
| Global texture/array loads | `textureLoad()` from sampled float textures |
| Surface/global writes | `textureStore()` to storage textures |
| Device arrays / structured buffers | `var<storage>` particle records with explicit ping-pong buffers |
| GPU-driven instancing | `draw(6, 8192, ...)` with `instance_index` in the vertex shader |
| Constant memory / symbol upload | One 112-byte uniform buffer updated with `queue.writeBuffer()` |
| Stream-ordered command work | Compute passes encoded in one `GPUCommandEncoder` and submitted once |
| CUDA error checks | `pushErrorScope()` / `popErrorScope()`, uncaptured errors, and `device.lost` |
| Device capability query | `adapter.info` and `adapter.limits`, shown in the status panel |
| CUDA events / GPU elapsed time | Optional WebGPU `timestamp-query`, sampled asynchronously every 30 frames |

The mapping is structural. It does not imply identical compiler behavior, cache policies, occupancy, warp scheduling, or numerical results across APIs.

## Profiling without a synchronization trap

When the adapter exposes `timestamp-query`, initialization requests the feature and creates a two-entry timestamp query set plus two small 16-byte buffers. A sampled command encoder writes a start timestamp before the first compute pass and an end timestamp after the render pass, resolves the query set, copies it to the map-readable buffer, and submits with the same queue path as every other frame.

The CPU does not map that buffer immediately. A background readback waits for submitted work only after the sampled submission has been queued, converts the GPU timestamp delta to milliseconds, and unmaps the buffer. Sampling is throttled to every 30 frames and is suppressed while an earlier readback is pending. This mirrors CUDA event timing while avoiding a `cudaDeviceSynchronize()`-style stall in the steady-state animation loop. If the feature is unavailable or a readback fails, the UI reports the limitation and the simulation continues without timing.

## Tiled stencil kernels

The divergence, pressure Jacobi, and vorticity passes are neighbor stencils. Each 16 × 16 workgroup stages an 18 × 18 tile: the active 16 × 16 region plus a one-texel halo on all four sides. The workgroup then synchronizes before reading neighbor values.

```text
       one-texel halo
    ┌──────────────────┐
    │                  │
    │   16 × 16 core   │  ← 256 invocations
    │                  │
    └──────────────────┘
       one-texel halo
```

The local-memory footprints are small and explicit:

- velocity tile: `18 × 18 × 2 × 4 = 2,592` bytes;
- scalar pressure/vorticity tile: `18 × 18 × 4 = 1,296` bytes.

Each invocation first stages its core cell. Boundary invocations additionally stage the halo cells, with mirrored edge values where the simulation domain ends. All invocations reach `workgroupBarrier()` before the stencil reads the tile. The final bounds check prevents stores outside the 512 × 512 domain, including any future partial dispatch.

This is the WebGPU analogue of a shared-memory CUDA stencil. Workgroup memory is local to one workgroup and is not persistent between passes, so the halo is intentionally reloaded for every divergence, pressure, or vorticity dispatch. The pressure solver still uses 20 Jacobi passes and ping-pongs its pressure textures; tiling changes memory locality, not the algorithm’s synchronization boundary.

## Why 16 × 16 workgroups

The fixed 16 × 16 shape gives 256 invocations per workgroup, maps cleanly to the 512 × 512 grid, and leaves enough room for the 18 × 18 halo tile on typical WebGPU devices. The status panel displays the selected shape and the adapter’s `maxComputeInvocationsPerWorkgroup` limit. It also exposes the maximum dispatch dimension as a tooltip.

The displayed limit is a capability guardrail, not a measurement of actual occupancy. WebGPU does not provide portable warp occupancy counters, register counts, shared-memory bank-conflict reports, or vendor-specific profiler telemetry.

## Command and memory behavior

The JavaScript coordinator creates textures, bind groups, pipeline layouts, shader modules, and pipelines during initialization. The animation loop only uploads the small uniform block, records passes, and submits one command buffer. There is no texture readback or GPU completion wait in the steady-state loop.

The resource graph uses explicit ping-pong fields:

- two `rg32float` velocity textures;
- two `rgba16float` density textures;
- two `r32float` pressure textures;
- one divergence texture and one optional vorticity texture.
- two 16-byte-stride particle storage buffers for GPU-only Lagrangian tracers.

Every pass samples one side and stores to the other side. This preserves WebGPU usage rules and corresponds to the explicit source/destination allocations commonly used in CUDA stencil pipelines.

The tracer kernel is a separate 64-thread workload. It reads the projected velocity texture and one particle buffer, writes the opposite particle buffer, and dispatches `ceil(8192 / 64)` workgroups. The following render pass uses `instance_index` to draw six vertices per particle directly from the active storage buffer. This is the portable equivalent of a CUDA device-array update followed by a GPU-driven visualization draw, with no host-side particle synchronization.

## Native CUDA porting sketch

If a native CUDA backend were added outside this browser project, each WGSL compute entry point could become a CUDA kernel with the same 16 × 16 block shape:

1. Allocate velocity, density, and pressure ping-pong fields in CUDA device memory, with surface or texture access as appropriate.
2. Replace `var<workgroup>` arrays with `__shared__` arrays sized for the 18 × 18 tile.
3. Replace `workgroupBarrier()` with `__syncthreads()` after cooperative core and halo loads.
4. Launch a 32 × 32 grid of blocks, preserving the global bounds guard.
5. Use CUDA events for elapsed-time measurements and a stream for ordered pass submission.
6. Keep the 20 pressure iterations and explicit swaps, or use CUDA Graph capture after the sequence is stable.
7. Add cuFFT only for a deliberately different pressure solver; it is not required by this Jacobi method.

That native backend is deliberately not included here. Adding `.cu` files or a CUDA build toolchain would violate the project’s single-file, zero-cost, browser-runnable constraint and would not make those kernels executable from an ordinary web page.

## Portability and future work

The portable path intentionally avoids warp-level shuffle operations, subgroup-size assumptions, vendor-specific formats, and direct host synchronization. Future WebGPU subgroup features may offer additional optimizations on browsers that expose them, but they would need a capability-gated fallback and should not replace the current workgroup-barrier path by default.

For a change to the tile shape or a new stencil field, update the shader workgroup arrays, capability checks, architecture resource graph, numerical notes, and the validation checklist together. Verify shader compilation on at least one WebGPU browser after every such change.
