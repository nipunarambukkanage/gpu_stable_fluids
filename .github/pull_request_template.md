## What changed

<!-- Describe the observable behavior change and the CPU/GPU ownership involved. -->

## Validation

- [ ] `npm run check`
- [ ] `git diff --check`
- [ ] Browser smoke test completed, or not applicable
- [ ] CUDA build and native tests completed, or the limitation is documented below

## GPU review checklist

- [ ] New buffers/textures have explicit lifetimes and labels.
- [ ] Kernel bounds, synchronization, and ping-pong ownership are correct.
- [ ] Host/device transfers are outside the hot loop unless justified.
- [ ] Any performance claim includes workload, GPU, driver, and build configuration.

## Evidence

<!-- Add screenshots, local diagnostics, benchmark output, or explain why none apply. -->

## Privacy and scope

- [ ] No analytics, remote shader, credential, or private telemetry code was added.
- [ ] No generated build output or machine-specific IDE metadata is included.
