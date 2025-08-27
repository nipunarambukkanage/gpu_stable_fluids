# Contributing

This repository is a dependency-free browser GPU laboratory. Changes should preserve the local-only runtime, the WebGPU-first design, and the explicit resource ownership described in `docs/ARCHITECTURE.md`.

## Development workflow

1. Work from a feature branch and keep commits focused by subsystem.
2. Do not add runtime packages, CDN assets, analytics, remote shaders, or backend services.
3. Run `npm run check` before committing. It validates the module graph, shader inventory, forbidden runtime dependencies, and Node syntax.
4. Serve the project with `npm run serve` and complete the browser checklist in `docs/VALIDATION.md` on a WebGPU-capable browser.
5. Update the relevant architecture, numerical, CUDA/WebGPU, and validation documentation when a GPU resource, shader, pass, or synchronization rule changes.

## GPU change checklist

- Keep sampled resources and storage destinations distinct within each pass.
- Keep ping-pong index changes adjacent to the pass that writes the opposite side.
- Guard every global invocation against the simulation dimensions.
- Keep workgroup-local arrays and barriers uniform across all invocations.
- Avoid GPU-to-CPU readback in the animation loop; optional profiling must remain asynchronous and capability-gated.
- Add adapter-limit checks when a feature introduces a new buffer, workgroup shape, format, or optional WebGPU feature.
- Label new textures, buffers, bind groups, pipelines, and passes for browser diagnostics.

## Commit and branch hygiene

Use descriptive commits such as `Add GPU-resident tracer particles and timing`. Review `git diff --check`, confirm the worktree contains only intended files, and push only to the explicitly requested branch and remote.
