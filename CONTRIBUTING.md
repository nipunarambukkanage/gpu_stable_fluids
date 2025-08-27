# Contributing

This repository contains a native CUDA C++ solver plus a dependency-free WebGPU preview. Changes should preserve local execution, explicit device/host ownership, and the resource lifecycle described in `docs/ARCHITECTURE.md` and `docs/CUDA_NATIVE.md`.

## Development workflow

1. Work from a feature branch and keep commits focused by subsystem.
2. Do not add runtime packages, CDN assets, analytics, remote shaders, or backend services.
3. Run `npm run check` before committing. It validates the module graph, shader inventory, forbidden runtime dependencies, and Node syntax.
4. Serve the project with `npm run serve` and complete the browser checklist in `docs/VALIDATION.md` on a WebGPU-capable browser.
5. On a CUDA-capable machine, configure and build the native CMake target, then run a short frame-export smoke test.
6. Update the relevant architecture, numerical, CUDA/WebGPU, native CUDA, and validation documentation when a GPU resource, shader, pass, or synchronization rule changes.

## GPU change checklist

- Keep sampled resources and storage destinations distinct within each pass.
- Keep ping-pong index changes adjacent to the pass that writes the opposite side.
- Guard every global invocation against the simulation dimensions.
- Keep workgroup-local arrays and barriers uniform across all invocations.
- Avoid GPU-to-CPU readback in the animation loop; optional profiling must remain asynchronous and capability-gated.
- Add adapter-limit checks when a feature introduces a new buffer, workgroup shape, format, or optional WebGPU feature.
- Label new textures, buffers, bind groups, pipelines, and passes for browser diagnostics.

## Commit and branch hygiene

## Native CUDA checklist

- Keep all device allocations in the solver lifecycle; never allocate or free inside a simulation iteration.
- Keep row-major X access contiguous for regular cell and particle kernels, and use restrict-qualified pointers when buffers cannot alias.
- Use shared-memory core-plus-halo tiles only when a stencil reuses neighboring values; keep __syncthreads() uniform across the block.
- Use separate kernel launches as global synchronization boundaries between ping-pong stages and pressure iterations.
- Keep the per-frame parameter upload small and asynchronous; do not copy simulation fields or particle positions to the CPU during stepping.
- Use nonblocking CUDA streams, CUDA events, and pinned host memory for visualization handoff and timing.
- Keep fast math opt-in and record the target CUDA architecture used for performance comparisons.
- Run npm run check, then build and run the native CMake target on a CUDA-capable machine before claiming native validation.

Use descriptive commits such as `Add GPU-resident tracer particles and timing`. Review `git diff --check`, confirm the worktree contains only intended files, and push only to the explicitly requested branch and remote.
