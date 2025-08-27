# Validation and maintenance checklist

This checklist is designed for a dependency-free project. It uses local browser functionality and does not require a package install, test service, cloud GPU, API key, or network request.

## Static checks

From the project directory:

```bash
node tests/static-contract.mjs
git diff --check
```

The same checks are exposed as `npm run check`; `package.json` has no dependencies.

The repository should contain exactly one runnable HTML document:

```bash
rg --files -g '*.html'
```

Expected output:

```text
fluid-simulation.html
```

Also review that the runtime contains no `TODO`, CDN, runtime `fetch()`, analytics, or external asset reference. `tests/static-contract.mjs` checks module syntax with Node’s parser; the browser then validates WGSL compilation and WebGPU pipeline creation.

## Browser smoke test

1. Start a free local static server, for example `python -m http.server 8000`.
2. Open `http://localhost:8000/fluid-simulation.html` in a current desktop Chrome or Edge build with WebGPU.
3. Confirm the status panel reaches `Ready`.
4. Confirm resolution is `512 × 512`, pressure is `20 iterations`, and FPS becomes populated.
5. If the adapter exposes timestamp queries, confirm `GPU frame` eventually shows a millisecond value; otherwise it should report `Not exposed` without affecting simulation startup.
6. Drag slowly and quickly across the canvas. Strokes should be continuous and directionally aligned.
7. Hold the pointer still. Ink should continue without a large new velocity impulse.
8. Select each ink preset and verify the input value, active swatch, and rendered hue change.
9. Select `Turbulent ribbon`, confirm vorticity shows a non-zero value, and watch for curls without validation errors.
10. Toggle `GPU tracer particles` off and on. The status should change between `Off` and `8,192 active`, with no GPU validation error.
11. Start and stop `Auto demo`. The button label and status should change, and the figure-eight should remain inside the domain.
12. Pause and resume. The field should freeze and resume without a time-step jump.
13. Clear. The field should become empty, simulation time should reset to `0.0 s`, and pointer history should be cleared.
14. Use `Save PNG` and confirm a local PNG download is generated.
15. Resize the window and test a narrow viewport. The simulation grid must remain 512 × 512 while the presentation canvas follows the viewport.
16. Change browser zoom or use a high-DPI display. Pointer location should remain aligned with the rendered ink.
17. Hide and show the document, then resume. No large burst or instability should appear after returning.
18. Inspect the browser console. Normal operation should produce no WebGPU validation error.

## Resource and shader review

The tracer smoke test should also show the GPU draw status as Indirect when enabled and Skipped when disabled. The indirect buffer must be storage-writable, indirect-readable, initialized for paused rendering, and refreshed before each enabled tracer draw. The two generic compute bullets below apply to the simulation-grid kernels; the indirect writer is intentionally a one-invocation scalar pass.

- Every compute entry point uses `@compute @workgroup_size(16, 16, 1)`.
- Every compute entry point guards global invocation IDs against width and height before loads/stores.
- Divergence, pressure, and vorticity stage an 18 × 18 tile and synchronize with `workgroupBarrier()` before neighbor reads.
- The adapter status reports `adapter.info` and the kernel status reports the workgroup shape plus the device invocation limit.
- Optional timestamp buffers are created only after the adapter advertises `timestamp-query`; unsupported adapters stay on the unsynchronized path.
- Tracer buffers use `STORAGE | COPY_DST`, are initialized identically, and swap only after the tracer compute pass.
- The tracer compute dispatch uses 64-thread workgroups and exactly `ceil(8192 / 64)` workgroups.
- The tracer overlay uses instanced rendering and additive blending without a CPU particle readback.
- `rg32float` velocity, `rgba16float` density, `r32float` pressure, divergence, and vorticity formats match their bind-group layouts.
- Storage destinations are never the sampled source in the same pass.
- The uniform buffer is 112 bytes and each WGSL `vec4<f32>` slot is 16-byte aligned.
- The pressure loop runs exactly 20 iterations and uses the tracked final index.
- Reset zeros both sides of every ping-pong field plus divergence and vorticity.
- Canvas resizing only reconfigures presentation; it never recreates simulation textures.

## Recovery review

The following cases should produce visible status rather than a silent failure:

- insecure context;
- missing `navigator.gpu`;
- no compatible adapter;
- device request failure;
- canvas context failure;
- shader compilation failure;
- pipeline or bind-group creation failure;
- uncaptured WebGPU validation error;
- device loss during the animation loop.

After device loss, `Retry GPU` should rebuild resources, reset the field, and leave exactly one active animation loop.

## Performance guardrails

Avoid introducing:

- GPU-to-CPU readback in `requestAnimationFrame`;
- `queue.onSubmittedWorkDone()` in the normal loop;
- per-frame pipeline, texture, sampler, or bind-group creation;
- unbounded frame deltas after visibility changes;
- arrays, DOM nodes, or closures in the hot path unless measured and justified;
- runtime package, CDN, font, image, telemetry, or backend dependencies.

If a future feature needs a new field or pass, update [ARCHITECTURE.md](ARCHITECTURE.md), [NUMERICAL_METHOD.md](NUMERICAL_METHOD.md), and this checklist together.
