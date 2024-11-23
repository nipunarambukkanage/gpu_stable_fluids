# Native CUDA C++ execution path

The native target is the primary implementation for the CUDA-oriented architecture. It separates responsibilities deliberately:

    CPU application
      ├── parses controls and owns lifecycle
      ├── selects the CUDA device and reports capabilities
      ├── uploads the small per-frame parameter block
      ├── records the ordered kernel graph on a nonblocking stream
      └── coordinates pinned frame output for visualization/export

    CUDA device
      ├── owns persistent velocity, density, pressure, divergence, vorticity, and particle buffers
      ├── executes one thread per cell or particle
      ├── stages stencil halos in shared memory
      └── renders the density field into a device frame buffer

The browser WebGPU page remains a portable preview. The native path is not a wrapper around WGSL: it contains CUDA C++ kernels, CUDA Runtime API allocation, streams, events, constant memory, and explicit device/host ownership.

## Native source layout

    cuda/
    ├── include/gpu_fluids/
    │   ├── config.hpp          # host/device layout, grid, block, and metrics contracts
    │   ├── cuda_utils.hpp      # checked CUDA Runtime API calls
    │   ├── solver.hpp          # persistent solver ownership boundary
    │   ├── input_trajectory.hpp # deterministic normalized stroke generation
    │   └── visualization.hpp  # CPU frame-export contract
    └── src/
        ├── input_trajectory.cpp # deterministic normalized stroke-path generator
        ├── solver.cu          # kernels plus CUDA resource lifecycle
        ├── sph_solver.cu      # SPH neighbor grid, forces, integration, and rendering
        ├── main.cpp           # CPU controls and demo workload
        └── visualization.cpp  # CPU PPM writer

## Build prerequisites

Install the NVIDIA CUDA Toolkit with nvcc, a C++17 compiler, and CMake 3.24 or newer. Configure for the compute capability of the target GPU; the default CMake value is 75 and can be overridden:

    cmake -S . -B build/cuda -DCMAKE_CUDA_ARCHITECTURES=86
    cmake --build build/cuda --config Release

Run a short native workload and export frames:

    build/cuda/Release/fluid_cuda_demo.exe --frames 120 --export-every 30

On a single-configuration generator, the executable is usually at:

    build/cuda/fluid_cuda_demo

Useful controls:

- --no-vorticity disables the optional curl and confinement stages.
- --mode sph selects the GPU SPH pipeline with uniform-grid neighbor search.
- --pressure-iterations N changes the Jacobi iteration count for experiments.
- --device N selects the CUDA device index owned by the CPU driver.
- --output DIR selects the CPU-side PPM output directory.
- --quiet suppresses per-frame reporting.

The current environment used for repository verification does not include nvcc, so the CUDA translation unit cannot be compiled here. The static CUDA contract test still checks that the native source preserves the required runtime, memory, synchronization, and ownership patterns.

## Persistent device state

The solver allocates the complete resource graph once:

| Allocation | Purpose |
| --- | --- |
| Two float2 velocity arrays | Source/destination velocity ping-pong |
| Two float4 density arrays | RGB ink and coverage ping-pong |
| Two scalar pressure arrays | Jacobi pressure ping-pong |
| Divergence and vorticity arrays | Intermediate scalar fields |
| Two float4 particle arrays | GPU-resident Lagrangian tracer ping-pong |
| Device uchar4 frame | GPU-rendered presentation image |
| Pinned host frame | Asynchronous device-to-host visualization handoff |

The SPH mode adds a separate persistent resource set:

| Allocation | Purpose |
| --- | --- |
| Double-buffered position and velocity SoA arrays | Coalesced particle state with one writer per particle |
| Density, pressure, and force arrays | Neighbor-derived SPH state and force accumulation |
| Cell counters and bounded particle index table | Uniform-grid spatial partition for nine-cell neighbor traversal |
| Device and pinned overflow counters | Makes cell-capacity pressure observable without a full particle readback |

Both solver modes keep their simulation state device-resident between steps. The SPH path only transfers its small constant parameter block during stepping, plus the rendered frame and overflow diagnostic when the CPU requests presentation.

The CPU reference uses an `EllipticalStrokeTrajectory` to generate its default
input. Its coordinates and radii are normalized, so the same configured path
scales predictably to every grid size while passing an ordinary stroke endpoint
and velocity through the parameter block.

Reset uses asynchronous memset and one-time particle initialization on the compute stream. No cudaMalloc, cudaFree, or repeated bulk initialization appears in the simulation iteration. The only normal-loop host-to-device transfer is the small 112-byte SimulationParams block copied to CUDA constant memory.

## Kernel and memory strategy

The cell kernels launch a 32 by 32 grid of 16 by 16 blocks over the fixed 512 by 512 domain. X is the contiguous dimension, so flat row-major arrays give each warp adjacent addresses for regular field loads and stores. Kernel arguments use restrict-qualified pointers to make non-aliasing intent explicit to the compiler.

The divergence, pressure Jacobi, vorticity, confinement, and pressure-gradient kernels cooperatively stage an 18 by 18 core-plus-halo tile in shared memory. Threads load the interior and edge halo, synchronize with __syncthreads(), and then reuse neighboring values from the tile. This reduces repeated global-memory transactions for the stencil stages.

Advection is inherently less regular because backtraced coordinates are data-dependent. It still writes contiguous output cells and uses manual bilinear sampling with clamped coordinates. The backtrace displacement is bounded to 48 texels, limiting pathological interpolation footprints after extreme impulses or long frame gaps. Particle advection uses one 128-thread CUDA thread block per contiguous particle range, with one thread owning one particle record.

The render kernel converts the final density field to an RGBA frame on the GPU. The CPU only receives the finished presentation buffer when visualization requests it; it never receives the simulation field or particle positions during stepping.

## Synchronization and streams

The compute stream is the owner of numerical ordering. Separate kernel launches provide global synchronization points between stages, which is required for ping-pong fields and pressure iterations. Each pressure Jacobi iteration is a distinct kernel launch, preserving the solver dependency chain.

The copy stream waits on a CUDA event recorded after the render kernel. It then performs an asynchronous device-to-pinned-host copy. A caller that needs to present or export a frame calls downloadFrame, which synchronizes only the copy stream at that presentation boundary. Simulation control and kernel recording stay asynchronous.

CUDA events bracket the complete numerical and render sequence. Frame statistics use cudaEventElapsedTime after the frame is ready, so timing measures GPU work rather than CPU submission time. Checked CUDA errors are raised after every kernel launch and every runtime API call.

## SPH neighbor-search pipeline

The SPH mode keeps particle state in structure-of-arrays buffers so position and velocity components are independently coalesced. Each step clears a fixed-capacity uniform grid, inserts one particle per CUDA thread with atomic cell counters, computes density and pressure from the nine neighboring cells, computes pressure and viscosity forces, integrates velocity and position, applies boundary damping, and renders a particle field on the GPU. A device overflow counter is copied with the frame so capacity pressure is visible instead of silently hidden.

## Numerical sequence

Each native step records:

1. Constant-memory parameter upload.
2. Splat injection into the opposite velocity and density buffers.
3. Semi-Lagrangian velocity and density advection.
4. Optional vorticity and confinement.
5. Centered divergence.
6. The configured number of Jacobi pressure launches.
7. Pressure-gradient subtraction and boundary enforcement.
8. GPU-resident particle advection.
9. GPU density-to-RGBA rendering.
10. Event-ordered copy to pinned host memory.

The CPU swaps the logical read indices after the launch that writes the opposite buffer. The device pointers remain allocated for the life of the solver.

## Tuning guidance

- Keep fast math disabled for baseline numerical comparisons; enable FLUID_ENABLE_FAST_MATH only for visual experiments.
- Build for the real GPU architecture instead of relying on a broad virtual target.
- Use Nsight Systems to inspect stream overlap and Nsight Compute to measure achieved occupancy, memory throughput, and shared-memory behavior.
- Treat the reported register count and static shared-memory size as launch diagnostics, not as a substitute for profiling.
- Increase pressure iterations only after measuring convergence and frame time together.
- Consider CUDA Graph capture only after the optional-stage configuration and ping-pong ownership are made graph-stable.

The implementation intentionally avoids pretending that WebGPU exposes CUDA-only tools. Native CUDA profiling and occupancy analysis belong to this target and require the CUDA Toolkit plus an NVIDIA GPU.
