import {
  GRID_WIDTH,
  GRID_HEIGHT,
  WORKGROUP_SIZE,
  WORKGROUPS_X,
  WORKGROUPS_Y,
  QUALITY_PROFILES,
  DEFAULT_QUALITY_PROFILE,
  MAX_FRAME_DELTA,
  MAX_BACKTRACE_DISTANCE,
  MAX_POINTER_VELOCITY,
  MAX_DENSITY,
  MAX_EFFECTIVE_DPR,
  PARTICLE_COUNT,
  PARTICLE_WORKGROUP_SIZE,
  PARTICLE_WORKGROUPS,
  PARTICLE_BUFFER_SIZE,
  PARTICLE_DRAW_VERTEX_COUNT,
  INDIRECT_ARGS_SIZE,
  UNIFORM_FLOAT_COUNT,
  UNIFORM_BUFFER_SIZE,
  BRUSH_MODES,
  DEFAULT_BRUSH_MODE,
  DEFAULT_RENDER_MODE,
  RENDER_MODE_INDICES,
  RENDER_MODES,
  createInitialParticleData
} from "./config/simulation.js";
import { createGpuTimer, destroyGpuTimer, readGpuTimestamp } from "./gpu/timestamp-profiler.js";
import { formatCapabilityFailure, inspectWebGpuAdapter, requestDeviceWithFallback } from "./gpu/capabilities.js";
import { createGpuPipelineBatch } from "./gpu/pipeline-factory.js";
import { createRuntimeTelemetry, recordGpuSample, recordRuntimeFrame, recordRuntimeSubmission, resetRuntimeTelemetry, snapshotRuntimeTelemetry } from "./gpu/telemetry.js";
import { advectionShaderCode, confinementShaderCode, divergenceShaderCode, gradientShaderCode, indirectArgsShaderCode, particleComputeShaderCode, particleFragmentShaderCode, particleVertexShaderCode, pressureShaderCode, renderFragmentShaderCode, renderVertexShaderCode, splatShaderCode, vorticityShaderCode } from "./gpu/shaders.js";
import { createDiagnosticsReport } from "./runtime/diagnostics.js";
import { createAdaptiveQualityGovernor } from "./runtime/adaptive-quality.js";
import { createInputRecorder, validateInputRecording } from "./runtime/input-recorder.js";
import { clearSettings, loadSettings, saveSettings } from "./runtime/settings-store.js";
import { SCENE_PRESETS } from "./config/presets.js";
import { createPerformanceHud } from "./ui/performance-hud.js";

"use strict";

    // 3. JavaScript constants and application state
    const prefersReducedMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

    const canvas = document.getElementById("fluidCanvas");
    const performanceHudRoot = document.getElementById("performanceHud");
    const performanceHud = createPerformanceHud(performanceHudRoot);
    const controls = document.getElementById("controls");
    const panelToggle = document.getElementById("panelToggle");
    const hideControlsButton = document.getElementById("hideControlsButton");
    const compatibilityCard = document.getElementById("compatibilityCard");
    const compatibilityMessage = document.getElementById("compatibilityMessage");
    const gpuStatus = document.getElementById("gpuStatus");
    const resolutionStatus = document.getElementById("resolutionStatus");
    const fpsStatus = document.getElementById("fpsStatus");
    const pressureStatus = document.getElementById("pressureStatus");
    const statusMessage = document.getElementById("statusMessage");
    const retryGpuButton = document.getElementById("retryGpuButton");
    const pauseButton = document.getElementById("pauseButton");
    const clearButton = document.getElementById("clearButton");
    const inkColorInput = document.getElementById("inkColor");
    const brushRadiusInput = document.getElementById("brushRadius");
    const velocityForceInput = document.getElementById("velocityForce");
    const inkAmountInput = document.getElementById("inkAmount");
    const velocityDissipationInput = document.getElementById("velocityDissipation");
    const inkDissipationInput = document.getElementById("inkDissipation");
    const inkColorValue = document.getElementById("inkColorValue");
    const colorPresetButtons = Array.from(document.querySelectorAll(".color-swatch"));
    const gpuAdapterStatus = document.getElementById("gpuAdapterStatus");
    const kernelStatus = document.getElementById("kernelStatus");
    const gpuTimeStatus = document.getElementById("gpuTimeStatus");
    const tracerStatus = document.getElementById("tracerStatus");
    const drawStatus = document.getElementById("drawStatus");
    const brushRadiusValue = document.getElementById("brushRadiusValue");
    const velocityForceValue = document.getElementById("velocityForceValue");
    const inkAmountValue = document.getElementById("inkAmountValue");
    const velocityDissipationValue = document.getElementById("velocityDissipationValue");
    const inkDissipationValue = document.getElementById("inkDissipationValue");
    const simulationTimeStatus = document.getElementById("simulationTimeStatus");
    const framePacingStatus = document.getElementById("framePacingStatus");
    const submissionStatus = document.getElementById("submissionStatus");
    const gpuSamplesStatus = document.getElementById("gpuSamplesStatus");
    const adaptiveQualityStatus = document.getElementById("adaptiveQualityStatus");
    const renderModeStatus = document.getElementById("renderModeStatus");
    const recordingStatus = document.getElementById("recordingStatus");
    const vorticityStatus = document.getElementById("vorticityStatus");
    const scenePresetInput = document.getElementById("scenePreset");
    const scenePresetValue = document.getElementById("scenePresetValue");
    const vorticityForceInput = document.getElementById("vorticityForce");
    const vorticityForceValue = document.getElementById("vorticityForceValue");
    const tracerToggle = document.getElementById("tracerToggle");
    const demoButton = document.getElementById("demoButton");
    const snapshotButton = document.getElementById("snapshotButton");
    const diagnosticsButton = document.getElementById("diagnosticsButton");
    const qualityProfileInput = document.getElementById("qualityProfile");
    const qualityProfileValue = document.getElementById("qualityProfileValue");
    const renderModeInput = document.getElementById("renderMode");
    const renderModeValue = document.getElementById("renderModeValue");
    const brushModeInput = document.getElementById("brushMode");
    const brushModeValue = document.getElementById("brushModeValue");
    const hudToggle = document.getElementById("hudToggle");
    const adaptiveQualityToggle = document.getElementById("adaptiveQualityToggle");
    const rememberToggle = document.getElementById("rememberToggle");
    const recordButton = document.getElementById("recordButton");
    const replayButton = document.getElementById("replayButton");
    const forgetSettingsButton = document.getElementById("forgetSettingsButton");

    // Uniform layout: each vec4 occupies one 16-byte slot.
    // 0..3: time, delta time, velocity dissipation, ink dissipation.
    // 4..7: grid width, grid height, pointer-active flag, brush radius in texels.
    // 8..11: stroke start x/y, reserved, reserved.
    // 12..15: stroke end x/y, reserved, reserved.
    // 16..19: injected velocity x/y, velocity-force multiplier, ink amount.
    // 20..23: ink color RGB, render exposure.
    // 24..27: vorticity strength, brush mode, diagnostic-view flag, diagnostic-view index.
    const uniformData = new Float32Array(UNIFORM_FLOAT_COUNT);
    const simulationSettings = {
      brushRadius: 18,
      velocityForce: 1,
      inkAmount: 1.8,
      velocityDissipation: 0.08,
      inkDissipation: 0.025,
      vorticityConfinement: 0,
      exposure: 0.72,
      qualityProfile: DEFAULT_QUALITY_PROFILE,
      pressureIterations: QUALITY_PROFILES[DEFAULT_QUALITY_PROFILE].pressureIterations,
      renderMode: DEFAULT_RENDER_MODE,
      brushMode: DEFAULT_BRUSH_MODE
    };
    const zeroTextureData = {
      velocity: new Uint8Array(GRID_WIDTH * GRID_HEIGHT * 8),
      density: new Uint8Array(GRID_WIDTH * GRID_HEIGHT * 8),
      scalar: new Uint8Array(GRID_WIDTH * GRID_HEIGHT * 4)
    };
    const initialParticleData = createInitialParticleData();

    const app = {
      adapter: null,
      device: null,
      context: null,
      canvasFormat: null,
      resources: null,
      particlesEnabled: true,
      gpuTimer: null,
      capabilities: null,
      available: false,
      initializing: false,
      paused: prefersReducedMotion,
      pageHidden: document.visibilityState === "hidden",
      animationFrameId: 0,
      animationRunning: false,
      lastFrameTime: null,
      simulationTime: 0,
      fpsAccumulator: 0,
      fpsFrames: 0,
      interfaceAccumulator: 0,
      telemetry: createRuntimeTelemetry(performance.now()),
      qualityGovernor: createAdaptiveQualityGovernor(),
      inputRecorder: createInputRecorder(),
      replay: {
        active: false,
        elapsedMs: 0,
        cursor: 0,
        recording: null
      },
      resizeObserver: null,
      presentationWidth: 0,
      presentationHeight: 0,
      pointer: {
        active: false,
        pointerId: null,
        hasPosition: false,
        segmentStartX: GRID_WIDTH * 0.5,
        segmentStartY: GRID_HEIGHT * 0.5,
        currentX: GRID_WIDTH * 0.5,
        currentY: GRID_HEIGHT * 0.5,
        velocityX: 0,
        velocityY: 0,
        hasPendingMotion: false,
        lastSampleTime: 0
      },
      demo: {
        active: false
      }
    };

    // 4. WGSL compute shaders
















    // 5. WGSL rendering shaders








    // 6. WebGPU initialization
    function setStatus(message, tone = "normal") {
      statusMessage.textContent = message;
      gpuStatus.dataset.tone = tone;
    }

    function setGpuState(label, tone = "normal") {
      gpuStatus.textContent = label;
      gpuStatus.dataset.tone = tone;
    }

    function updatePauseButton() {
      pauseButton.textContent = app.paused ? "Resume" : "Pause";
      pauseButton.setAttribute("aria-label", app.paused ? "Resume simulation" : "Pause simulation");
    }

    function setGpuControlsDisabled(disabled) {
      inkColorInput.disabled = disabled;
      for (const presetButton of colorPresetButtons) {
        presetButton.disabled = disabled;
      }
      brushRadiusInput.disabled = disabled;
      velocityForceInput.disabled = disabled;
      inkAmountInput.disabled = disabled;
      velocityDissipationInput.disabled = disabled;
      inkDissipationInput.disabled = disabled;
      scenePresetInput.disabled = disabled;
      vorticityForceInput.disabled = disabled;
      tracerToggle.disabled = disabled;
      pauseButton.disabled = disabled;
      clearButton.disabled = disabled;
      demoButton.disabled = disabled;
      snapshotButton.disabled = disabled;
      recordButton.disabled = disabled;
      replayButton.disabled = disabled || !app.replay.recording;
      diagnosticsButton.disabled = disabled;
      qualityProfileInput.disabled = disabled;
      renderModeInput.disabled = disabled;
      brushModeInput.disabled = disabled;
      hudToggle.disabled = disabled;
      adaptiveQualityToggle.disabled = disabled;
      rememberToggle.disabled = disabled;
      canvas.style.cursor = disabled ? "not-allowed" : "crosshair";
      qualityProfileInput.disabled = disabled || adaptiveQualityToggle.checked;
      updatePauseButton();
    }

    function showCompatibility(title, message) {
      compatibilityCard.querySelector("h1").textContent = title;
      compatibilityMessage.textContent = message;
      compatibilityCard.hidden = false;
    }

    function hideCompatibility() {
      compatibilityCard.hidden = true;
    }

    function updateAdapterTelemetry(adapter, gpuTimer = null) {
      const adapterInfo = adapter.info || {};
      const adapterLabel = adapterInfo.description || adapterInfo.device || [adapterInfo.vendor, adapterInfo.architecture].filter(Boolean).join(" · ") || "Local WebGPU adapter";
      const maximumInvocations = adapter.limits?.maxComputeInvocationsPerWorkgroup || 256;
      const maximumDispatches = adapter.limits?.maxComputeWorkgroupsPerDimension || 65535;
      const timestampFeatureAvailable = adapter.features?.has?.("timestamp-query") ?? false;
      gpuAdapterStatus.textContent = adapterLabel;
      gpuAdapterStatus.title = adapterLabel;
      kernelStatus.textContent = `${WORKGROUP_SIZE} × ${WORKGROUP_SIZE} / ${maximumInvocations}`;
      kernelStatus.title = `Fluid tile: ${WORKGROUP_SIZE} × ${WORKGROUP_SIZE}; tracer: ${PARTICLE_WORKGROUP_SIZE} threads; dispatch limit: ${maximumDispatches.toLocaleString()} workgroups per dimension`;
      gpuTimeStatus.textContent = gpuTimer?.enabled ? "Timestamp ready" : timestampFeatureAvailable ? "Feature available" : "Not exposed";
      gpuTimeStatus.title = gpuTimer?.enabled
        ? "Asynchronous timestamp-query samples are collected every 30 frames."
        : timestampFeatureAvailable
          ? "The adapter advertises timestamp-query, but the timer could not be created."
          : "This adapter does not expose the optional WebGPU timestamp-query feature.";
      updateTelemetryReadouts();
    }

    function updateTelemetryReadouts() {
      const telemetry = snapshotRuntimeTelemetry(app.telemetry, performance.now());
      framePacingStatus.textContent = telemetry.frameCount > 0 ? `${telemetry.averageCpuEncodeMs.toFixed(2)} ms avg` : "Collecting…";
      framePacingStatus.title = telemetry.frameCount > 0
        ? `Last CPU encode: ${telemetry.lastCpuEncodeMs.toFixed(2)} ms; longest frame gap: ${telemetry.maximumFrameDeltaMs.toFixed(2)} ms.`
        : "Measures CPU command encoding and submission cost, not GPU execution time.";
      submissionStatus.textContent = `${telemetry.submittedFrames.toLocaleString()} queued`;
      submissionStatus.title = `${telemetry.longFrames.toLocaleString()} long frame gaps detected; command buffers remain GPU-owned after submission.`;
      gpuSamplesStatus.textContent = `${telemetry.gpuSamples.toLocaleString()} samples`;
      gpuSamplesStatus.title = telemetry.lastGpuMs === null
        ? "Optional timestamp-query samples have not completed yet."
        : `Last GPU frame: ${telemetry.lastGpuMs.toFixed(2)} ms; EMA: ${telemetry.averageGpuMs.toFixed(2)} ms.`;
      const governor = app.qualityGovernor.snapshot();
      adaptiveQualityStatus.textContent = adaptiveQualityToggle.checked
        ? `Auto · ${governor.pressureIterations}`
        : "Manual";
      adaptiveQualityStatus.title = adaptiveQualityToggle.checked
        ? `Target 16.7 ms; current ${QUALITY_PROFILES[governor.profile].label}; last GPU sample ${governor.lastGpuMs === null ? "pending" : `${governor.lastGpuMs.toFixed(2)} ms`}.`
        : "Manual pressure profile selection is active.";
      renderModeStatus.textContent = RENDER_MODES[simulationSettings.renderMode]?.label || "Density";
      performanceHud.update({
        fps: fpsStatus.textContent,
        telemetry,
        pressureIterations: simulationSettings.pressureIterations,
        particleCount: PARTICLE_COUNT,
        adapter: gpuAdapterStatus.textContent,
        featureTier: app.capabilities?.featureTier || "baseline"
      });
    }

    function destroyGpuResources() {
      destroyGpuTimer(app.gpuTimer);
      app.gpuTimer = null;
      if (app.resources) {
        for (const textureGroup of Object.values(app.resources.textures)) {
          for (const texture of Array.isArray(textureGroup) ? textureGroup : [textureGroup]) {
            texture?.destroy();
          }
        }
        for (const bufferGroup of Object.values(app.resources.buffers || {})) {
          for (const buffer of Array.isArray(bufferGroup) ? bufferGroup : [bufferGroup]) {
            buffer?.destroy();
          }
        }
        app.resources.uniformBuffer?.destroy();
      }
      app.resources = null;
      app.available = false;
      app.adapter = null;
      app.capabilities = null;
      app.device = null;
      app.context = null;
      app.canvasFormat = null;
    }

    function createSimulationTexture(device, label, format) {
      return device.createTexture({
        label,
        size: [GRID_WIDTH, GRID_HEIGHT, 1],
        dimension: "2d",
        format,
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_DST
      });
    }

    function createTextureViews(textures) {
      return {
        velocity: textures.velocity.map((texture) => texture.createView({ label: `${texture.label} view` })),
        density: textures.density.map((texture) => texture.createView({ label: `${texture.label} view` })),
        pressure: textures.pressure.map((texture) => texture.createView({ label: `${texture.label} view` })),
        divergence: textures.divergence.createView({ label: "Divergence view" }),
        vorticity: textures.vorticity.createView({ label: "Vorticity view" })
      };
    }

    function createBindGroupLayouts(device) {
      const computeVisibility = GPUShaderStage.COMPUTE;
      const splat = device.createBindGroupLayout({
        label: "Splat bind group layout",
        entries: [
          { binding: 0, visibility: computeVisibility, buffer: { type: "uniform" } },
          { binding: 1, visibility: computeVisibility, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 2, visibility: computeVisibility, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 3, visibility: computeVisibility, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "2d" } },
          { binding: 4, visibility: computeVisibility, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "2d" } }
        ]
      });
      const advection = device.createBindGroupLayout({
        label: "Advection bind group layout",
        entries: [
          { binding: 0, visibility: computeVisibility, buffer: { type: "uniform" } },
          { binding: 1, visibility: computeVisibility, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 2, visibility: computeVisibility, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 3, visibility: computeVisibility, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "2d" } },
          { binding: 4, visibility: computeVisibility, storageTexture: { access: "write-only", format: "rgba16float", viewDimension: "2d" } }
        ]
      });
      const divergence = device.createBindGroupLayout({
        label: "Divergence bind group layout",
        entries: [
          { binding: 0, visibility: computeVisibility, buffer: { type: "uniform" } },
          { binding: 1, visibility: computeVisibility, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 2, visibility: computeVisibility, storageTexture: { access: "write-only", format: "r32float", viewDimension: "2d" } }
        ]
      });
      const pressure = device.createBindGroupLayout({
        label: "Pressure bind group layout",
        entries: [
          { binding: 0, visibility: computeVisibility, buffer: { type: "uniform" } },
          { binding: 1, visibility: computeVisibility, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 2, visibility: computeVisibility, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 3, visibility: computeVisibility, storageTexture: { access: "write-only", format: "r32float", viewDimension: "2d" } }
        ]
      });
      const gradient = device.createBindGroupLayout({
        label: "Pressure gradient bind group layout",
        entries: [
          { binding: 0, visibility: computeVisibility, buffer: { type: "uniform" } },
          { binding: 1, visibility: computeVisibility, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 2, visibility: computeVisibility, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 3, visibility: computeVisibility, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "2d" } }
        ]
      });
      const vorticity = device.createBindGroupLayout({
        label: "Vorticity bind group layout",
        entries: [
          { binding: 0, visibility: computeVisibility, buffer: { type: "uniform" } },
          { binding: 1, visibility: computeVisibility, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 2, visibility: computeVisibility, storageTexture: { access: "write-only", format: "r32float", viewDimension: "2d" } }
        ]
      });
      const confinement = device.createBindGroupLayout({
        label: "Vorticity confinement bind group layout",
        entries: [
          { binding: 0, visibility: computeVisibility, buffer: { type: "uniform" } },
          { binding: 1, visibility: computeVisibility, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 2, visibility: computeVisibility, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 3, visibility: computeVisibility, storageTexture: { access: "write-only", format: "rg32float", viewDimension: "2d" } }
        ]
      });
      const particles = device.createBindGroupLayout({
        label: "GPU tracer compute bind group layout",
        entries: [
          { binding: 0, visibility: computeVisibility, buffer: { type: "uniform" } },
          { binding: 1, visibility: computeVisibility, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 2, visibility: computeVisibility, buffer: { type: "read-only-storage" } },
          { binding: 3, visibility: computeVisibility, buffer: { type: "storage" } }
        ]
      });
      const render = device.createBindGroupLayout({
        label: "Render bind group layout",
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "float", viewDimension: "2d" } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: "unfilterable-float", viewDimension: "2d" } },
          { binding: 5, visibility: GPUShaderStage.FRAGMENT, sampler: { type: "filtering" } },
          { binding: 6, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }
        ]
      });
      const particleRender = device.createBindGroupLayout({
        label: "GPU tracer render bind group layout",
        entries: [
          { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } },
          { binding: 1, visibility: GPUShaderStage.VERTEX, buffer: { type: "read-only-storage" } }
        ]
      });
      const indirectArgs = device.createBindGroupLayout({
        label: "GPU indirect draw-args bind group layout",
        entries: [
          { binding: 0, visibility: computeVisibility, buffer: { type: "storage" } }
        ]
      });
      return { splat, advection, divergence, pressure, gradient, vorticity, confinement, particles, render, particleRender, indirectArgs };
    }

    async function createShaderModule(device, code, label) {
      const module = device.createShaderModule({ label, code });
      if (typeof module.getCompilationInfo === "function") {
        const compilationInfo = await module.getCompilationInfo();
        const errors = compilationInfo.messages.filter((message) => message.type === "error");
        if (errors.length > 0) {
          const details = errors.map((error) => `${error.lineNum}:${error.linePos} ${error.message}`).join("; ");
          throw new Error(`${label} compilation failed: ${details}`);
        }
      }
      return module;
    }

    async function createPipelines(device, layouts) {
      const [splatModule, advectionModule, divergenceModule, pressureModule, gradientModule, vorticityModule, confinementModule, particleComputeModule, vertexModule, fragmentModule, particleVertexModule, particleFragmentModule, indirectArgsModule] = await Promise.all([
        createShaderModule(device, splatShaderCode, "Splat shader"),
        createShaderModule(device, advectionShaderCode, "Advection shader"),
        createShaderModule(device, divergenceShaderCode, "Divergence shader"),
        createShaderModule(device, pressureShaderCode, "Pressure shader"),
        createShaderModule(device, gradientShaderCode, "Gradient shader"),
        createShaderModule(device, vorticityShaderCode, "Vorticity shader"),
        createShaderModule(device, confinementShaderCode, "Vorticity confinement shader"),
        createShaderModule(device, particleComputeShaderCode, "GPU tracer compute shader"),
        createShaderModule(device, renderVertexShaderCode, "Render vertex shader"),
        createShaderModule(device, renderFragmentShaderCode, "Render fragment shader"),
        createShaderModule(device, particleVertexShaderCode, "GPU tracer vertex shader"),
        createShaderModule(device, particleFragmentShaderCode, "GPU tracer fragment shader"),
        createShaderModule(device, indirectArgsShaderCode, "GPU indirect draw arguments shader")
      ]);

      const pipelineLayouts = {
        splat: device.createPipelineLayout({ label: "Splat pipeline layout", bindGroupLayouts: [layouts.splat] }),
        advection: device.createPipelineLayout({ label: "Advection pipeline layout", bindGroupLayouts: [layouts.advection] }),
        divergence: device.createPipelineLayout({ label: "Divergence pipeline layout", bindGroupLayouts: [layouts.divergence] }),
        pressure: device.createPipelineLayout({ label: "Pressure pipeline layout", bindGroupLayouts: [layouts.pressure] }),
        gradient: device.createPipelineLayout({ label: "Pressure gradient pipeline layout", bindGroupLayouts: [layouts.gradient] }),
        vorticity: device.createPipelineLayout({ label: "Vorticity pipeline layout", bindGroupLayouts: [layouts.vorticity] }),
        confinement: device.createPipelineLayout({ label: "Vorticity confinement pipeline layout", bindGroupLayouts: [layouts.confinement] }),
        particles: device.createPipelineLayout({ label: "GPU tracer compute pipeline layout", bindGroupLayouts: [layouts.particles] }),
        render: device.createPipelineLayout({ label: "Density render pipeline layout", bindGroupLayouts: [layouts.render] }),
        particleRender: device.createPipelineLayout({ label: "GPU tracer render pipeline layout", bindGroupLayouts: [layouts.particleRender] }),
        indirectArgs: device.createPipelineLayout({ label: "GPU indirect draw arguments pipeline layout", bindGroupLayouts: [layouts.indirectArgs] })
      };

      const [splatPipeline, advectionPipeline, divergencePipeline, pressurePipeline, gradientPipeline, vorticityPipeline, confinementPipeline, particlePipeline, renderPipeline, particleRenderPipeline, indirectArgsPipeline] = await createGpuPipelineBatch(device, [
        { kind: "compute", descriptor: { label: "Splat pipeline", layout: pipelineLayouts.splat, compute: { module: splatModule, entryPoint: "main" } } },
        { kind: "compute", descriptor: { label: "Advection pipeline", layout: pipelineLayouts.advection, compute: { module: advectionModule, entryPoint: "main" } } },
        { kind: "compute", descriptor: { label: "Divergence pipeline", layout: pipelineLayouts.divergence, compute: { module: divergenceModule, entryPoint: "main" } } },
        { kind: "compute", descriptor: { label: "Pressure pipeline", layout: pipelineLayouts.pressure, compute: { module: pressureModule, entryPoint: "main" } } },
        { kind: "compute", descriptor: { label: "Pressure gradient pipeline", layout: pipelineLayouts.gradient, compute: { module: gradientModule, entryPoint: "main" } } },
        { kind: "compute", descriptor: { label: "Vorticity pipeline", layout: pipelineLayouts.vorticity, compute: { module: vorticityModule, entryPoint: "main" } } },
        { kind: "compute", descriptor: { label: "Vorticity confinement pipeline", layout: pipelineLayouts.confinement, compute: { module: confinementModule, entryPoint: "main" } } },
        { kind: "compute", descriptor: { label: "GPU tracer compute pipeline", layout: pipelineLayouts.particles, compute: { module: particleComputeModule, entryPoint: "main" } } },
        {
          kind: "render",
          descriptor: {
            label: "Density render pipeline",
            layout: pipelineLayouts.render,
            vertex: { module: vertexModule, entryPoint: "main" },
            fragment: { module: fragmentModule, entryPoint: "main", targets: [{ format: app.canvasFormat }] },
            primitive: { topology: "triangle-list" }
          }
        },
        {
          kind: "render",
          descriptor: {
            label: "GPU tracer render pipeline",
            layout: pipelineLayouts.particleRender,
            vertex: { module: particleVertexModule, entryPoint: "main" },
            fragment: {
              module: particleFragmentModule,
              entryPoint: "main",
              targets: [{
                format: app.canvasFormat,
                blend: {
                  color: { srcFactor: "src-alpha", dstFactor: "one", operation: "add" },
                  alpha: { srcFactor: "one", dstFactor: "one", operation: "add" }
                }
              }]
            },
            primitive: { topology: "triangle-list" }
          }
        },
        { kind: "compute", descriptor: { label: "GPU indirect draw arguments pipeline", layout: pipelineLayouts.indirectArgs, compute: { module: indirectArgsModule, entryPoint: "main" } } }
      ]);

      return {
        shaders: {
          splat: splatModule,
          advection: advectionModule,
          divergence: divergenceModule,
          pressure: pressureModule,
          gradient: gradientModule,
          vorticity: vorticityModule,
          confinement: confinementModule,
          particles: particleComputeModule,
          vertex: vertexModule,
          fragment: fragmentModule,
          particleVertex: particleVertexModule,
          particleFragment: particleFragmentModule,
          indirectArgs: indirectArgsModule
        },
        pipelineLayouts,
        pipelines: {
          splat: splatPipeline,
          advection: advectionPipeline,
          divergence: divergencePipeline,
          pressure: pressurePipeline,
          gradient: gradientPipeline,
          vorticity: vorticityPipeline,
          confinement: confinementPipeline,
          particles: particlePipeline,
          render: renderPipeline,
          particleRender: particleRenderPipeline,
          indirectArgs: indirectArgsPipeline
        }
      };
    }

    function createSplatOrAdvectionBindGroups(device, layout, uniformBuffer, views, kind) {
      return [0, 1].map((readIndex) => device.createBindGroup({
        label: `${kind} bind group ${readIndex}`,
        layout,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: views.velocity[readIndex] },
          { binding: 2, resource: views.density[readIndex] },
          { binding: 3, resource: views.velocity[1 - readIndex] },
          { binding: 4, resource: views.density[1 - readIndex] }
        ]
      }));
    }

    function createParticleBuffers(device) {
      const buffers = [0, 1].map((index) => device.createBuffer({
        label: `GPU tracer particle buffer ${index}`,
        size: PARTICLE_BUFFER_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
      }));
      for (const buffer of buffers) {
        device.queue.writeBuffer(buffer, 0, initialParticleData);
      }
      return buffers;
    }

    function createIndirectArgsBuffer(device) {
      const buffer = device.createBuffer({
        label: "GPU-generated tracer indirect draw arguments",
        size: INDIRECT_ARGS_SIZE,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST
      });
      device.queue.writeBuffer(buffer, 0, new Uint32Array([PARTICLE_DRAW_VERTEX_COUNT, PARTICLE_COUNT, 0, 0]));
      return buffer;
    }

    function createBindGroups(device, layouts, uniformBuffer, views, sampler, particleBuffers, indirectArgsBuffer) {
      const splat = createSplatOrAdvectionBindGroups(device, layouts.splat, uniformBuffer, views, "Splat");
      const advection = createSplatOrAdvectionBindGroups(device, layouts.advection, uniformBuffer, views, "Advection");
      const divergence = [0, 1].map((readIndex) => device.createBindGroup({
        label: `Divergence bind group ${readIndex}`,
        layout: layouts.divergence,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: views.velocity[readIndex] },
          { binding: 2, resource: views.divergence }
        ]
      }));
      const pressure = [0, 1].map((readIndex) => device.createBindGroup({
        label: `Pressure bind group ${readIndex}`,
        layout: layouts.pressure,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: views.pressure[readIndex] },
          { binding: 2, resource: views.divergence },
          { binding: 3, resource: views.pressure[1 - readIndex] }
        ]
      }));
      const gradient = [0, 1].map((pressureIndex) => [0, 1].map((velocityIndex) => device.createBindGroup({
        label: `Gradient bind group p${pressureIndex} v${velocityIndex}`,
        layout: layouts.gradient,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: views.pressure[pressureIndex] },
          { binding: 2, resource: views.velocity[velocityIndex] },
          { binding: 3, resource: views.velocity[1 - velocityIndex] }
        ]
      })));
      const vorticity = [0, 1].map((velocityIndex) => device.createBindGroup({
        label: `Vorticity bind group ${velocityIndex}`,
        layout: layouts.vorticity,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: views.velocity[velocityIndex] },
          { binding: 2, resource: views.vorticity }
        ]
      }));
      const confinement = [0, 1].map((velocityIndex) => device.createBindGroup({
        label: `Vorticity confinement bind group ${velocityIndex}`,
        layout: layouts.confinement,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: views.velocity[velocityIndex] },
          { binding: 2, resource: views.vorticity },
          { binding: 3, resource: views.velocity[1 - velocityIndex] }
        ]
      }));
      const particles = [0, 1].map((velocityIndex) => [0, 1].map((particleIndex) => device.createBindGroup({
        label: `GPU tracer compute bind group v${velocityIndex} p${particleIndex}`,
        layout: layouts.particles,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: views.velocity[velocityIndex] },
          { binding: 2, resource: { buffer: particleBuffers[particleIndex] } },
          { binding: 3, resource: { buffer: particleBuffers[1 - particleIndex] } }
        ]
      })));
      const render = [0, 1].map((densityIndex) => [0, 1].map((velocityIndex) => [0, 1].map((pressureIndex) => device.createBindGroup({
        label: `Render bind group d${densityIndex} v${velocityIndex} p${pressureIndex}`,
        layout: layouts.render,
        entries: [
          { binding: 0, resource: views.density[densityIndex] },
          { binding: 1, resource: views.velocity[velocityIndex] },
          { binding: 2, resource: views.pressure[pressureIndex] },
          { binding: 3, resource: views.divergence },
          { binding: 4, resource: views.vorticity },
          { binding: 5, resource: sampler },
          { binding: 6, resource: { buffer: uniformBuffer } }
        ]
      }))));
      const particleRender = [0, 1].map((particleIndex) => device.createBindGroup({
        label: `GPU tracer render bind group ${particleIndex}`,
        layout: layouts.particleRender,
        entries: [
          { binding: 0, resource: { buffer: uniformBuffer } },
          { binding: 1, resource: { buffer: particleBuffers[particleIndex] } }
        ]
      }));
      const indirectArgs = device.createBindGroup({
        label: "GPU indirect draw arguments bind group",
        layout: layouts.indirectArgs,
        entries: [{ binding: 0, resource: { buffer: indirectArgsBuffer } }]
      });
      return { splat, advection, divergence, pressure, gradient, vorticity, confinement, particles, render, particleRender, indirectArgs };
    }

    async function createGpuResources(device) {
      const uniformBuffer = device.createBuffer({
        label: "Per-frame simulation uniforms",
        size: UNIFORM_BUFFER_SIZE,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      });
      const textures = {
        velocity: [
          createSimulationTexture(device, "Velocity ping-pong A", "rg32float"),
          createSimulationTexture(device, "Velocity ping-pong B", "rg32float")
        ],
        density: [
          createSimulationTexture(device, "Density ping-pong A", "rgba16float"),
          createSimulationTexture(device, "Density ping-pong B", "rgba16float")
        ],
        pressure: [
          createSimulationTexture(device, "Pressure ping-pong A", "r32float"),
          createSimulationTexture(device, "Pressure ping-pong B", "r32float")
        ],
        divergence: createSimulationTexture(device, "Divergence", "r32float"),
        vorticity: createSimulationTexture(device, "Vorticity", "r32float")
      };
      const particleBuffers = createParticleBuffers(device);
      const indirectArgsBuffer = createIndirectArgsBuffer(device);
      const views = createTextureViews(textures);
      const sampler = device.createSampler({ label: "Linear presentation sampler", minFilter: "linear", magFilter: "linear" });
      const layouts = createBindGroupLayouts(device);
      const pipelineBundle = await createPipelines(device, layouts);
      const bindGroups = createBindGroups(device, layouts, uniformBuffer, views, sampler, particleBuffers, indirectArgsBuffer);
      return {
        uniformBuffer,
        textures,
        buffers: { particles: particleBuffers, indirectArgs: indirectArgsBuffer },
        views,
        sampler,
        layouts,
        shaders: pipelineBundle.shaders,
        pipelineLayouts: pipelineBundle.pipelineLayouts,
        pipelines: pipelineBundle.pipelines,
        bindGroups,
        velocityIndex: 0,
        densityIndex: 0,
        pressureIndex: 0,
        particleIndex: 0
      };
    }

    function writeZeroTexture(texture, data, bytesPerRow) {
      app.device.queue.writeTexture(
        { texture },
        data,
        { bytesPerRow, rowsPerImage: GRID_HEIGHT },
        { width: GRID_WIDTH, height: GRID_HEIGHT, depthOrArrayLayers: 1 }
      );
    }

    function clearSimulationTextures() {
      if (!app.device || !app.resources) {
        return;
      }
      for (const texture of app.resources.textures.velocity) {
        writeZeroTexture(texture, zeroTextureData.velocity, GRID_WIDTH * 8);
      }
      for (const texture of app.resources.textures.density) {
        writeZeroTexture(texture, zeroTextureData.density, GRID_WIDTH * 8);
      }
      for (const texture of app.resources.textures.pressure) {
        writeZeroTexture(texture, zeroTextureData.scalar, GRID_WIDTH * 4);
      }
      writeZeroTexture(app.resources.textures.divergence, zeroTextureData.scalar, GRID_WIDTH * 4);
      writeZeroTexture(app.resources.textures.vorticity, zeroTextureData.scalar, GRID_WIDTH * 4);
      for (const buffer of app.resources.buffers.particles) {
        app.device.queue.writeBuffer(buffer, 0, initialParticleData);
      }
      app.device.queue.writeBuffer(app.resources.buffers.indirectArgs, 0, new Uint32Array([PARTICLE_DRAW_VERTEX_COUNT, PARTICLE_COUNT, 0, 0]));
      app.resources.velocityIndex = 0;
      app.resources.densityIndex = 0;
      app.resources.pressureIndex = 0;
      app.resources.particleIndex = 0;
      resetPointerState();
      app.simulationTime = 0;
      app.lastFrameTime = null;
      app.fpsAccumulator = 0;
      app.fpsFrames = 0;
      app.interfaceAccumulator = 0;
      resetRuntimeTelemetry(app.telemetry, performance.now());
      simulationTimeStatus.textContent = "0.0 s";
      updateTelemetryReadouts();
    }

    function handleUncapturedError(event) {
      const message = event.error?.message || "An uncaptured WebGPU validation error occurred.";
      setGpuState("Validation warning", "warning");
      setStatus(message, "warning");
      console.error("WebGPU uncaptured error:", event.error);
    }

    function handleDeviceLost(device, info) {
      if (app.device !== device) {
        return;
      }
      stopAnimationLoop();
      const reason = info?.message || info?.reason || "The GPU device was lost.";
      destroyGpuResources();
      setGpuControlsDisabled(true);
      setGpuState("Device lost", "danger");
      setStatus(`${reason} Use Retry GPU to rebuild the simulation.`, "danger");
      retryGpuButton.hidden = false;
      showCompatibility("GPU device lost", `${reason} The simulation can recover by requesting a fresh local GPU device.`);
    }

    async function initializeGpu() {
      if (app.initializing) {
        return;
      }

      app.initializing = true;
      resetRuntimeTelemetry(app.telemetry, performance.now());
      stopAnimationLoop();
      destroyGpuResources();
      app.demo.active = false;
      resetPointerState();
      demoButton.textContent = "Auto demo";
      setGpuControlsDisabled(true);
      retryGpuButton.hidden = true;
      setGpuState("Starting…", "warning");
      setStatus("Requesting a high-performance local GPU…", "warning");

      try {
        if (!window.isSecureContext) {
          throw new Error("WebGPU requires a secure context. Open this file through HTTPS or localhost.");
        }
        if (!navigator.gpu) {
          throw new Error("This browser does not expose WebGPU. Use a current desktop Chrome or Edge release with WebGPU support.");
        }

        const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" }) || await navigator.gpu.requestAdapter();
        if (!adapter) {
          throw new Error("No compatible WebGPU adapter was found. Check that the browser and local GPU support WebGPU.");
        }
        const capabilities = inspectWebGpuAdapter(adapter, { workgroupSize: WORKGROUP_SIZE, particleBufferSize: PARTICLE_BUFFER_SIZE });
        if (!capabilities.supported) {
          throw new Error(formatCapabilityFailure(capabilities));
        }
        const deviceSelection = await requestDeviceWithFallback(adapter, capabilities);
        const device = deviceSelection.device;
        const context = canvas.getContext("webgpu");
        if (!context) {
          throw new Error("Could not acquire a WebGPU canvas context.");
        }

        app.adapter = adapter;
        app.capabilities = deviceSelection.capabilities;
        app.device = device;
        app.context = context;
        app.canvasFormat = navigator.gpu.getPreferredCanvasFormat();
        app.gpuTimer = createGpuTimer(device);
        updateAdapterTelemetry(adapter, app.gpuTimer);
        device.onuncapturederror = handleUncapturedError;
        device.lost.then((info) => handleDeviceLost(device, info)).catch((error) => console.error("Device-loss handler failed:", error));
        device.pushErrorScope("validation");
        let initializedResources;
        try {
          initializedResources = await createGpuResources(device);
        } finally {
          const initializationValidationError = await device.popErrorScope();
          if (initializationValidationError) {
            throw new Error(`WebGPU validation failed during initialization: ${initializationValidationError.message}`);
          }
        }
        app.resources = initializedResources;
        configurePresentationCanvas();
        clearSimulationTextures();
        app.available = true;
        app.paused = prefersReducedMotion;
        hideCompatibility();
        setGpuControlsDisabled(false);
        setGpuState("Ready", "good");
        setStatus(app.paused ? "Reduced-motion preference: paused. Press Resume or Space to begin." : "Drag anywhere on the canvas to inject ink and motion.", "good");
        updatePauseButton();
        if (app.paused) {
          renderCurrentDensity();
        } else {
          startAnimationLoop();
        }
      } catch (error) {
        console.error("WebGPU initialization failed:", error);
        destroyGpuResources();
        const message = error instanceof Error ? error.message : "Unknown WebGPU initialization failure.";
        setGpuControlsDisabled(true);
        setGpuState("Unavailable", "danger");
        setStatus(message, "danger");
        retryGpuButton.hidden = false;
        showCompatibility("WebGPU is not ready", `${message} WebGPU generally requires a current browser and either HTTPS or localhost.`);
      } finally {
        app.initializing = false;
      }
    }

    // 7. Texture and buffer creation is performed by createGpuResources above.
    // 8. Pipeline and bind-group creation is performed before the first frame.

    // 10. Simulation passes
    const inkColorComponents = new Float32Array(3);

    function clampNumber(value, minimum, maximum) {
      return Math.min(maximum, Math.max(minimum, value));
    }

    function refreshInkColorComponents() {
      const hex = inkColorInput.value;
      inkColorComponents[0] = Number.parseInt(hex.slice(1, 3), 16) / 255;
      inkColorComponents[1] = Number.parseInt(hex.slice(3, 5), 16) / 255;
      inkColorComponents[2] = Number.parseInt(hex.slice(5, 7), 16) / 255;
    }

    function updateReplayPointer(deltaTime) {
      const replay = app.replay;
      if (!replay.active || !replay.recording?.samples.length) {
        return false;
      }
      replay.elapsedMs += deltaTime * 1000;
      const samples = replay.recording.samples;
      while (replay.cursor < samples.length - 1 && samples[replay.cursor + 1].timeMs <= replay.elapsedMs) {
        replay.cursor += 1;
      }
      const sample = samples[replay.cursor];
      const pointer = app.pointer;
      if (!pointer.hasPosition) {
        pointer.segmentStartX = sample.x;
        pointer.segmentStartY = sample.y;
        pointer.hasPosition = true;
      } else {
        pointer.segmentStartX = pointer.currentX;
        pointer.segmentStartY = pointer.currentY;
      }
      pointer.currentX = clampNumber(sample.x, 0.5, GRID_WIDTH - 0.5);
      pointer.currentY = clampNumber(sample.y, 0.5, GRID_HEIGHT - 0.5);
      pointer.velocityX = clampNumber(sample.velocityX, -MAX_POINTER_VELOCITY, MAX_POINTER_VELOCITY);
      pointer.velocityY = clampNumber(sample.velocityY, -MAX_POINTER_VELOCITY, MAX_POINTER_VELOCITY);
      pointer.active = sample.active;
      pointer.hasPendingMotion = true;
      if (replay.elapsedMs > replay.recording.durationMs + 100) {
        replay.active = false;
        resetPointerState();
        setStatus("Stroke macro replay complete. The simulation remains GPU-resident.", "good");
      }
      updateRecordingStatus();
      return true;
    }

    function updateUniformBuffer(deltaTime) {
      const resources = app.resources;
      if (!updateReplayPointer(deltaTime)) {
        updateDemoPointer(deltaTime);
      }
      const pointer = app.pointer;
      const hasPosition = pointer.hasPosition;
      const startX = hasPosition ? pointer.segmentStartX : GRID_WIDTH * 0.5;
      const startY = hasPosition ? pointer.segmentStartY : GRID_HEIGHT * 0.5;
      const endX = hasPosition ? pointer.currentX : startX;
      const endY = hasPosition ? pointer.currentY : startY;
      const injectedVelocityX = pointer.active && pointer.hasPendingMotion ? pointer.velocityX : 0;
      const injectedVelocityY = pointer.active && pointer.hasPendingMotion ? pointer.velocityY : 0;

      uniformData[0] = app.simulationTime;
      uniformData[1] = deltaTime;
      uniformData[2] = simulationSettings.velocityDissipation;
      uniformData[3] = simulationSettings.inkDissipation;
      uniformData[4] = GRID_WIDTH;
      uniformData[5] = GRID_HEIGHT;
      uniformData[6] = pointer.active ? 1 : 0;
      uniformData[7] = simulationSettings.brushRadius;
      uniformData[8] = startX;
      uniformData[9] = startY;
      uniformData[10] = 0;
      uniformData[11] = 0;
      uniformData[12] = endX;
      uniformData[13] = endY;
      uniformData[14] = 0;
      uniformData[15] = 0;
      uniformData[16] = injectedVelocityX;
      uniformData[17] = injectedVelocityY;
      uniformData[18] = simulationSettings.velocityForce;
      uniformData[19] = simulationSettings.inkAmount;
      uniformData[20] = inkColorComponents[0];
      uniformData[21] = inkColorComponents[1];
      uniformData[22] = inkColorComponents[2];
      uniformData[23] = simulationSettings.exposure;
      uniformData[24] = simulationSettings.vorticityConfinement;
      uniformData[25] = BRUSH_MODES[simulationSettings.brushMode]?.value ?? BRUSH_MODES[DEFAULT_BRUSH_MODE].value;
      uniformData[26] = simulationSettings.renderMode === DEFAULT_RENDER_MODE ? 0 : 1;
      uniformData[27] = RENDER_MODE_INDICES[simulationSettings.renderMode] ?? RENDER_MODE_INDICES[DEFAULT_RENDER_MODE];

      app.device.queue.writeBuffer(resources.uniformBuffer, 0, uniformData);

      if (pointer.hasPosition) {
        pointer.segmentStartX = pointer.currentX;
        pointer.segmentStartY = pointer.currentY;
      }
      pointer.hasPendingMotion = false;
    }

    function encodeComputePass(commandEncoder, pipeline, bindGroup, label, options = {}) {
      const computePass = commandEncoder.beginComputePass({ label, timestampWrites: options.timestampWrites });
      computePass.setPipeline(pipeline);
      computePass.setBindGroup(0, bindGroup);
      computePass.dispatchWorkgroups(options.workgroupsX ?? WORKGROUPS_X, options.workgroupsY ?? WORKGROUPS_Y, options.workgroupsZ ?? 1);
      computePass.end();
    }

    function encodeRenderPass(commandEncoder, timestampWrites = undefined) {
      const renderPass = commandEncoder.beginRenderPass({
        label: "Density presentation pass",
        timestampWrites,
        colorAttachments: [{
          view: app.context.getCurrentTexture().createView({ label: "Presentation view" }),
          clearValue: { r: 0.002, g: 0.006, b: 0.012, a: 1 },
          loadOp: "clear",
          storeOp: "store"
        }]
      });
      renderPass.setPipeline(app.resources.pipelines.render);
      renderPass.setBindGroup(
        0,
        app.resources.bindGroups.render[app.resources.densityIndex][app.resources.velocityIndex][app.resources.pressureIndex]
      );
      renderPass.draw(6, 1, 0, 0);
      if (app.particlesEnabled) {
        renderPass.setPipeline(app.resources.pipelines.particleRender);
        renderPass.setBindGroup(0, app.resources.bindGroups.particleRender[app.resources.particleIndex]);
        renderPass.drawIndirect(app.resources.buffers.indirectArgs, 0);
      }
      renderPass.end();
    }

    function encodeSimulationFrame(deltaTime) {
      if (!app.available || !app.device || !app.resources || !app.context) {
        return;
      }

      const resources = app.resources;
      updateUniformBuffer(deltaTime);
      const commandEncoder = app.device.createCommandEncoder({ label: "Stable fluids frame" });
      const gpuTimer = app.gpuTimer;
      const shouldSampleGpuTime = Boolean(gpuTimer?.enabled && !gpuTimer.pending && gpuTimer.frameCounter % gpuTimer.sampleEvery === 0);
      if (gpuTimer) {
        gpuTimer.frameCounter += 1;
      }
      if (shouldSampleGpuTime) {
        encodeComputePass(
          commandEncoder,
          resources.pipelines.splat,
          resources.bindGroups.splat[resources.velocityIndex],
          "Splat injection pass",
          { timestampWrites: { querySet: gpuTimer.querySet, beginningOfPassWriteIndex: 0 } }
        );
      } else {
        encodeComputePass(commandEncoder, resources.pipelines.splat, resources.bindGroups.splat[resources.velocityIndex], "Splat injection pass");
      }
      resources.velocityIndex = 1 - resources.velocityIndex;
      resources.densityIndex = 1 - resources.densityIndex;

      // Semi-Lagrangian advection reads only the new current pair and writes the other pair.
      encodeComputePass(commandEncoder, resources.pipelines.advection, resources.bindGroups.advection[resources.velocityIndex], "Velocity and density advection pass");
      resources.velocityIndex = 1 - resources.velocityIndex;
      resources.densityIndex = 1 - resources.densityIndex;

      // Optional curl enhancement. At zero strength the base Stable Fluids path remains unchanged.
      if (simulationSettings.vorticityConfinement > 0.0001) {
        encodeComputePass(commandEncoder, resources.pipelines.vorticity, resources.bindGroups.vorticity[resources.velocityIndex], "Vorticity pass");
        encodeComputePass(commandEncoder, resources.pipelines.confinement, resources.bindGroups.confinement[resources.velocityIndex], "Vorticity confinement pass");
        resources.velocityIndex = 1 - resources.velocityIndex;
      }

      encodeComputePass(commandEncoder, resources.pipelines.divergence, resources.bindGroups.divergence[resources.velocityIndex], "Divergence pass");

      // Each Jacobi iteration is its own compute pass so the read/write usage scope is distinct.
      for (let iteration = 0; iteration < simulationSettings.pressureIterations; iteration += 1) {
        encodeComputePass(commandEncoder, resources.pipelines.pressure, resources.bindGroups.pressure[resources.pressureIndex], "Pressure Jacobi iteration");
        resources.pressureIndex = 1 - resources.pressureIndex;
      }

      encodeComputePass(
        commandEncoder,
        resources.pipelines.gradient,
        resources.bindGroups.gradient[resources.pressureIndex][resources.velocityIndex],
        "Pressure-gradient subtraction pass"
      );
      resources.velocityIndex = 1 - resources.velocityIndex;

      if (app.particlesEnabled) {
        encodeComputePass(
          commandEncoder,
          resources.pipelines.indirectArgs,
          resources.bindGroups.indirectArgs,
          "GPU indirect draw-arguments pass",
          { workgroupsX: 1, workgroupsY: 1, workgroupsZ: 1 }
        );
        encodeComputePass(
          commandEncoder,
          resources.pipelines.particles,
          resources.bindGroups.particles[resources.velocityIndex][resources.particleIndex],
          "GPU tracer advection pass",
          { workgroupsX: PARTICLE_WORKGROUPS, workgroupsY: 1 }
        );
        resources.particleIndex = 1 - resources.particleIndex;
      }

      encodeRenderPass(commandEncoder, shouldSampleGpuTime ? { querySet: gpuTimer.querySet, endOfPassWriteIndex: 1 } : undefined);
      if (shouldSampleGpuTime) {
        commandEncoder.resolveQuerySet(gpuTimer.querySet, 0, 2, gpuTimer.resolveBuffer, 0);
        commandEncoder.copyBufferToBuffer(gpuTimer.resolveBuffer, 0, gpuTimer.readbackBuffer, 0, 16);
      }
      app.device.queue.submit([commandEncoder.finish()]);
      recordRuntimeSubmission(app.telemetry);
      if (shouldSampleGpuTime) {
        readGpuTimestamp(gpuTimer, app, gpuTimeStatus, (milliseconds) => {
          recordGpuSample(app.telemetry, milliseconds);
          const qualityChange = adaptiveQualityToggle.checked
            ? app.qualityGovernor.observe(milliseconds, performance.now())
            : null;
          if (qualityChange) {
            applyQualityProfile(qualityChange.profile, { fromGovernor: true });
            setStatus(`Adaptive quality shifted to ${QUALITY_PROFILES[qualityChange.profile].label} (${qualityChange.reason}).`, "warning");
          }
          updateTelemetryReadouts();
        });
      }
    }

    // 11. Rendering
    function renderCurrentDensity() {
      if (!app.available || !app.device || !app.resources || !app.context) {
        return;
      }
      try {
        const commandEncoder = app.device.createCommandEncoder({ label: "Initial density render" });
        encodeRenderPass(commandEncoder);
        app.device.queue.submit([commandEncoder.finish()]);
      } catch (error) {
        console.error("Density render failed:", error);
        setStatus("Rendering failed. Use Retry GPU to rebuild the device.", "danger");
      }
    }

    // 13. Main animation loop
    function stopAnimationLoop() {
      if (app.animationFrameId !== 0) {
        cancelAnimationFrame(app.animationFrameId);
      }
      app.animationFrameId = 0;
      app.animationRunning = false;
      app.lastFrameTime = null;
    }

    function startAnimationLoop() {
      if (!app.available || app.paused || app.pageHidden || app.animationRunning) {
        return;
      }
      app.animationRunning = true;
      app.lastFrameTime = null;
      app.animationFrameId = requestAnimationFrame(animationFrame);
    }

    function updateFps(deltaTime) {
      app.fpsAccumulator += deltaTime;
      app.fpsFrames += 1;
      app.interfaceAccumulator += deltaTime;
      if (app.interfaceAccumulator >= 0.25) {
        simulationTimeStatus.textContent = `${app.simulationTime.toFixed(1)} s`;
        updateTelemetryReadouts();
        app.interfaceAccumulator = 0;
      }
      if (app.fpsAccumulator >= 0.5) {
        const framesPerSecond = app.fpsFrames / app.fpsAccumulator;
        fpsStatus.textContent = `${Math.round(framesPerSecond)} FPS`;
        app.fpsAccumulator = 0;
        app.fpsFrames = 0;
      }
    }

    function animationFrame(timestamp) {
      app.animationFrameId = 0;
      if (!app.animationRunning || !app.available || app.paused || app.pageHidden) {
        return;
      }

      const frameStart = performance.now();
      const measuredFrameDeltaMs = app.lastFrameTime === null ? 1000 / 60 : timestamp - app.lastFrameTime;
      const frameDeltaMs = Number.isFinite(measuredFrameDeltaMs) ? Math.max(0, measuredFrameDeltaMs) : 1000 / 60;
      const rawDelta = frameDeltaMs / 1000;
      const deltaTime = clampNumber(Number.isFinite(rawDelta) ? rawDelta : 1 / 60, 1 / 1000, MAX_FRAME_DELTA);
      app.lastFrameTime = timestamp;
      app.simulationTime += deltaTime;
      updateFps(deltaTime);

      try {
        encodeSimulationFrame(deltaTime);
        recordRuntimeFrame(app.telemetry, frameDeltaMs, performance.now() - frameStart);
      } catch (error) {
        console.error("Simulation frame failed:", error);
        stopAnimationLoop();
        app.available = false;
        setGpuControlsDisabled(true);
        setGpuState("Frame error", "danger");
        setStatus("A GPU frame failed. Use Retry GPU to rebuild the simulation.", "danger");
        retryGpuButton.hidden = false;
        showCompatibility("GPU frame failed", "The simulation stopped safely after a GPU command error. Retry to rebuild the local device.");
        return;
      }

      if (app.animationRunning) {
        app.animationFrameId = requestAnimationFrame(animationFrame);
      }
    }

    // 8. Pipeline and bind-group creation is completed before the first frame.

    // 9. Pointer interaction
    function resetPointerState() {
      const pointer = app.pointer;
      pointer.active = false;
      pointer.pointerId = null;
      pointer.hasPosition = false;
      pointer.segmentStartX = GRID_WIDTH * 0.5;
      pointer.segmentStartY = GRID_HEIGHT * 0.5;
      pointer.currentX = GRID_WIDTH * 0.5;
      pointer.currentY = GRID_HEIGHT * 0.5;
      pointer.velocityX = 0;
      pointer.velocityY = 0;
      pointer.hasPendingMotion = false;
      pointer.lastSampleTime = 0;
    }

    function getPointerTimestamp(event) {
      const now = performance.now();
      const eventTime = event.timeStamp;
      // Some browsers expose epoch-based event timestamps while others use the
      // performance time origin. Reject a timestamp from a different clock.
      return Number.isFinite(eventTime) && eventTime > 0 && Math.abs(eventTime - now) < 60_000 ? eventTime : now;
    }

    function pointerToSimulationPosition(event) {
      const rectangle = canvas.getBoundingClientRect();
      const width = Math.max(rectangle.width, 1);
      const height = Math.max(rectangle.height, 1);
      return {
        x: clampNumber(((event.clientX - rectangle.left) / width) * GRID_WIDTH, 0.5, GRID_WIDTH - 0.5),
        y: clampNumber(((event.clientY - rectangle.top) / height) * GRID_HEIGHT, 0.5, GRID_HEIGHT - 0.5)
      };
    }

    function processPointerSample(event, calculateVelocity) {
      const pointer = app.pointer;
      const position = pointerToSimulationPosition(event);
      const sampleTime = getPointerTimestamp(event);

      if (!pointer.hasPosition) {
        pointer.segmentStartX = position.x;
        pointer.segmentStartY = position.y;
        pointer.currentX = position.x;
        pointer.currentY = position.y;
        pointer.hasPosition = true;
        pointer.velocityX = 0;
        pointer.velocityY = 0;
        pointer.hasPendingMotion = false;
      } else {
        const elapsed = (sampleTime - pointer.lastSampleTime) / 1000;
        const deltaX = position.x - pointer.currentX;
        const deltaY = position.y - pointer.currentY;
        if (calculateVelocity && elapsed > 0 && elapsed <= 0.25) {
          let velocityX = deltaX / elapsed;
          let velocityY = deltaY / elapsed;
          const magnitude = Math.hypot(velocityX, velocityY);
          if (magnitude > MAX_POINTER_VELOCITY) {
            const scale = MAX_POINTER_VELOCITY / magnitude;
            velocityX *= scale;
            velocityY *= scale;
          }
          pointer.velocityX = Number.isFinite(velocityX) ? velocityX : 0;
          pointer.velocityY = Number.isFinite(velocityY) ? velocityY : 0;
          pointer.hasPendingMotion = true;
        } else if (calculateVelocity) {
          // A long event gap is usually a tab switch or pointer re-entry, not a real impulse.
          pointer.velocityX = 0;
          pointer.velocityY = 0;
          pointer.hasPendingMotion = true;
        }
        pointer.currentX = position.x;
        pointer.currentY = position.y;
      }
      pointer.lastSampleTime = sampleTime;
      if (app.inputRecorder.isRecording()) {
        app.inputRecorder.record({
          x: pointer.currentX,
          y: pointer.currentY,
          velocityX: pointer.velocityX,
          velocityY: pointer.velocityY,
          active: pointer.active
        }, sampleTime);
        updateRecordingStatus();
      }
    }

    function getPointerSamples(event) {
      if (typeof event.getCoalescedEvents === "function") {
        const coalescedEvents = event.getCoalescedEvents();
        if (coalescedEvents.length > 0) {
          return coalescedEvents;
        }
      }
      return [event];
    }

    function handlePointerDown(event) {
      if (!app.available) {
        return;
      }
      if (app.demo.active) {
        app.demo.active = false;
        resetPointerState();
        demoButton.textContent = "Auto demo";
      }
      if (app.replay.active) {
        app.replay.active = false;
        resetPointerState();
        updateRecordingStatus();
      }
      if (app.pointer.active) {
        return;
      }
      event.preventDefault();
      app.pointer.active = true;
      app.pointer.pointerId = event.pointerId;
      canvas.focus({ preventScroll: true });
      try {
        canvas.setPointerCapture(event.pointerId);
      } catch (error) {
        console.warn("Pointer capture unavailable:", error);
      }
      processPointerSample(event, false);
    }

    function handlePointerMove(event) {
      if (!app.available || !app.pointer.active || event.pointerId !== app.pointer.pointerId) {
        return;
      }
      event.preventDefault();
      for (const sample of getPointerSamples(event)) {
        processPointerSample(sample, true);
      }
    }

    function finishPointerInteraction(event) {
      if (!app.pointer.active || event.pointerId !== app.pointer.pointerId) {
        return;
      }
      event.preventDefault();
      const lastSample = getPointerSamples(event).at(-1) || event;
      processPointerSample(lastSample, false);
      if (app.inputRecorder.isRecording()) {
        app.inputRecorder.record({
          x: app.pointer.currentX,
          y: app.pointer.currentY,
          velocityX: 0,
          velocityY: 0,
          active: false
        }, getPointerTimestamp(lastSample));
      }
      try {
        if (canvas.hasPointerCapture(event.pointerId)) {
          canvas.releasePointerCapture(event.pointerId);
        }
      } catch (error) {
        console.warn("Pointer capture release failed:", error);
      }
      resetPointerState();
    }

    function handlePointerCancel(event) {
      if (app.pointer.active && event.pointerId === app.pointer.pointerId) {
        resetPointerState();
      }
    }

    function collectPersistedSettings() {
      return {
        inkColor: inkColorInput.value,
        brushRadius: Number(brushRadiusInput.value),
        velocityForce: Number(velocityForceInput.value),
        inkAmount: Number(inkAmountInput.value),
        velocityDissipation: Number(velocityDissipationInput.value),
        inkDissipation: Number(inkDissipationInput.value),
        vorticityConfinement: Number(vorticityForceInput.value),
        scenePreset: scenePresetInput.value,
        qualityProfile: simulationSettings.qualityProfile,
        renderMode: simulationSettings.renderMode,
        brushMode: simulationSettings.brushMode,
        tracersEnabled: tracerToggle.checked,
        hudEnabled: hudToggle.checked,
        adaptiveQuality: adaptiveQualityToggle.checked
      };
    }

    function getSettingsStorage() {
      try {
        return window.localStorage;
      } catch {
        return null;
      }
    }

    function persistSettingsIfEnabled() {
      if (!rememberToggle.checked) {
        return;
      }
      saveSettings(getSettingsStorage(), collectPersistedSettings());
    }

    function applyQualityProfile(profileName, { fromGovernor = false } = {}) {
      if (!QUALITY_PROFILES[profileName]) {
        return false;
      }
      simulationSettings.qualityProfile = profileName;
      simulationSettings.pressureIterations = QUALITY_PROFILES[profileName].pressureIterations;
      qualityProfileInput.value = profileName;
      qualityProfileValue.textContent = QUALITY_PROFILES[profileName].label;
      qualityProfileInput.title = QUALITY_PROFILES[profileName].description;
      pressureStatus.textContent = `${simulationSettings.pressureIterations} iterations`;
      if (!fromGovernor) {
        app.qualityGovernor.setProfile(profileName, performance.now());
      }
      persistSettingsIfEnabled();
      return true;
    }

    function restoreStoredSettings() {
      const stored = loadSettings(getSettingsStorage());
      if (!stored) {
        return;
      }
      inkColorInput.value = stored.inkColor;
      brushRadiusInput.value = String(stored.brushRadius);
      velocityForceInput.value = String(stored.velocityForce);
      inkAmountInput.value = String(stored.inkAmount);
      velocityDissipationInput.value = String(stored.velocityDissipation);
      inkDissipationInput.value = String(stored.inkDissipation);
      vorticityForceInput.value = String(stored.vorticityConfinement);
      scenePresetInput.value = stored.scenePreset;
      qualityProfileInput.value = stored.qualityProfile;
      renderModeInput.value = stored.renderMode;
      brushModeInput.value = stored.brushMode;
      simulationSettings.qualityProfile = stored.qualityProfile;
      simulationSettings.pressureIterations = QUALITY_PROFILES[stored.qualityProfile].pressureIterations;
      app.qualityGovernor.setProfile(stored.qualityProfile, performance.now());
      tracerToggle.checked = stored.tracersEnabled;
      hudToggle.checked = stored.hudEnabled;
      adaptiveQualityToggle.checked = stored.adaptiveQuality;
      rememberToggle.checked = true;
    }

    function updateControlReadouts() {
      simulationSettings.brushRadius = Number.parseFloat(brushRadiusInput.value);
      simulationSettings.velocityForce = Number.parseFloat(velocityForceInput.value);
      simulationSettings.inkAmount = Number.parseFloat(inkAmountInput.value);
      simulationSettings.velocityDissipation = Number.parseFloat(velocityDissipationInput.value);
      simulationSettings.inkDissipation = Number.parseFloat(inkDissipationInput.value);
      simulationSettings.vorticityConfinement = Number.parseFloat(vorticityForceInput.value);
      simulationSettings.renderMode = RENDER_MODES[renderModeInput.value] ? renderModeInput.value : DEFAULT_RENDER_MODE;
      simulationSettings.brushMode = BRUSH_MODES[brushModeInput.value] ? brushModeInput.value : DEFAULT_BRUSH_MODE;
      if (!adaptiveQualityToggle.checked) {
        applyQualityProfile(QUALITY_PROFILES[qualityProfileInput.value] ? qualityProfileInput.value : DEFAULT_QUALITY_PROFILE);
      } else {
        applyQualityProfile(simulationSettings.qualityProfile, { fromGovernor: true });
      }
      refreshInkColorComponents();
      const selectedColor = inkColorInput.value.toLowerCase();
      for (const presetButton of colorPresetButtons) {
        presetButton.setAttribute("aria-pressed", String(presetButton.dataset.color === selectedColor));
      }
      inkColorValue.textContent = inkColorInput.value.toUpperCase();
      brushRadiusValue.textContent = `${brushRadiusInput.value} texels`;
      velocityForceValue.textContent = `${Number.parseFloat(velocityForceInput.value).toFixed(2)}×`;
      inkAmountValue.textContent = Number.parseFloat(inkAmountInput.value).toFixed(2);
      velocityDissipationValue.textContent = Number.parseFloat(velocityDissipationInput.value).toFixed(2);
      inkDissipationValue.textContent = Number.parseFloat(inkDissipationInput.value).toFixed(3);
      vorticityForceValue.textContent = simulationSettings.vorticityConfinement > 0 ? `${simulationSettings.vorticityConfinement.toFixed(2)}×` : "Off";
      vorticityStatus.textContent = simulationSettings.vorticityConfinement > 0 ? `${simulationSettings.vorticityConfinement.toFixed(2)}×` : "Off";
      scenePresetValue.textContent = scenePresetInput.options[scenePresetInput.selectedIndex].textContent;
      renderModeValue.textContent = RENDER_MODES[simulationSettings.renderMode].label;
      renderModeInput.title = RENDER_MODES[simulationSettings.renderMode].description;
      brushModeValue.textContent = BRUSH_MODES[simulationSettings.brushMode].label;
      brushModeInput.title = BRUSH_MODES[simulationSettings.brushMode].description;
      pressureStatus.textContent = `${simulationSettings.pressureIterations} iterations`;
      qualityProfileInput.disabled = !app.available || adaptiveQualityToggle.checked;
      persistSettingsIfEnabled();
    }

    function updateTracerState() {
      drawStatus.textContent = tracerToggle.checked ? "Indirect" : "Skipped";
      drawStatus.title = tracerToggle.checked
        ? "Tracer instance count is generated in a GPU storage buffer and consumed by drawIndirect."
        : "GPU tracer compute and indirect draw are disabled.";
      app.particlesEnabled = tracerToggle.checked;
      tracerStatus.textContent = app.particlesEnabled ? `${PARTICLE_COUNT.toLocaleString()} active` : "Off";
      tracerStatus.title = app.particlesEnabled
        ? `${PARTICLE_COUNT.toLocaleString()} GPU-resident Lagrangian tracers are advected in a dedicated compute pass.`
        : "GPU tracer compute and rendering are disabled.";
      persistSettingsIfEnabled();
    }

    function updateHudState() {
      performanceHud.setVisible(hudToggle.checked);
      updateTelemetryReadouts();
      persistSettingsIfEnabled();
    }

    function updateAdaptiveQualityState() {
      if (adaptiveQualityToggle.checked) {
        app.qualityGovernor.setProfile(simulationSettings.qualityProfile, performance.now());
        setStatus("Adaptive quality is monitoring completed GPU timestamps.", "good");
      } else {
        applyQualityProfile(qualityProfileInput.value || DEFAULT_QUALITY_PROFILE);
        setStatus("Manual pressure quality is active.", "good");
      }
      qualityProfileInput.disabled = !app.available || adaptiveQualityToggle.checked;
      updateTelemetryReadouts();
      persistSettingsIfEnabled();
    }

    function updateRecordingStatus() {
      const recorder = app.inputRecorder;
      if (recorder.isRecording()) {
        const count = recorder.snapshot().samples.length;
        recordingStatus.textContent = `Recording · ${count.toLocaleString()}`;
        recordButton.textContent = "Stop recording";
      } else if (app.replay.active) {
        recordingStatus.textContent = "Replaying";
        recordButton.textContent = "Record strokes";
      } else if (app.replay.recording) {
        recordingStatus.textContent = `${app.replay.recording.samples.length.toLocaleString()} samples ready`;
        recordButton.textContent = "Record strokes";
      } else {
        recordingStatus.textContent = "Ready";
        recordButton.textContent = "Record strokes";
      }
      replayButton.disabled = !app.available || !app.replay.recording || app.inputRecorder.isRecording();
    }

    function toggleRecording() {
      if (!app.available) {
        return;
      }
      if (app.inputRecorder.isRecording()) {
        app.inputRecorder.stop();
        const recording = validateInputRecording(app.inputRecorder.snapshot());
        app.replay.recording = recording?.samples.length ? recording : null;
        setStatus(app.replay.recording ? "Stroke macro captured. Replay it without additional CPU/GPU state transfers." : "No stroke samples were captured.", "good");
      } else {
        app.replay.active = false;
        app.inputRecorder.start(performance.now());
        setStatus("Recording pointer strokes. Paint one or more gestures, then stop recording.", "good");
      }
      updateRecordingStatus();
    }

    function replayRecording() {
      const recording = validateInputRecording(app.replay.recording);
      if (!app.available || !recording) {
        setStatus("Record a stroke macro before replaying it.", "warning");
        return;
      }
      app.replay.recording = recording;
      app.replay.active = true;
      app.replay.elapsedMs = 0;
      app.replay.cursor = 0;
      app.demo.active = false;
      resetPointerState();
      demoButton.textContent = "Auto demo";
      setStatus("Replaying the captured stroke macro on the GPU-resident simulation.", "good");
      if (app.paused) {
        togglePause();
      }
      updateRecordingStatus();
    }

    function selectInkPreset(event) {
      const presetButton = event.currentTarget;
      inkColorInput.value = presetButton.dataset.color;
      updateControlReadouts();
    }

    function applyScenePreset() {
      const preset = SCENE_PRESETS[scenePresetInput.value];
      if (!preset) {
        return;
      }
      inkColorInput.value = preset.color;
      brushRadiusInput.value = String(preset.brushRadius);
      inkAmountInput.value = String(preset.inkAmount);
      velocityForceInput.value = String(preset.velocityForce);
      velocityDissipationInput.value = String(preset.velocityDissipation);
      inkDissipationInput.value = String(preset.inkDissipation);
      vorticityForceInput.value = String(preset.vorticity);
      updateControlReadouts();
      if (app.available) {
        clearSimulation();
      }
    }

    function updateDemoPointer(deltaTime) {
      if (!app.demo.active) {
        return;
      }
      const pointer = app.pointer;
      const time = app.simulationTime * 0.72;
      const nextX = GRID_WIDTH * 0.5 + Math.cos(time) * GRID_WIDTH * 0.27;
      const nextY = GRID_HEIGHT * 0.5 + Math.sin(time * 1.37) * GRID_HEIGHT * 0.23;
      if (!pointer.hasPosition) {
        pointer.segmentStartX = nextX;
        pointer.segmentStartY = nextY;
        pointer.currentX = nextX;
        pointer.currentY = nextY;
        pointer.hasPosition = true;
        pointer.hasPendingMotion = false;
        pointer.velocityX = 0;
        pointer.velocityY = 0;
        return;
      }
      const safeDelta = Math.max(deltaTime, 1 / 240);
      let velocityX = (nextX - pointer.currentX) / safeDelta;
      let velocityY = (nextY - pointer.currentY) / safeDelta;
      const magnitude = Math.hypot(velocityX, velocityY);
      if (magnitude > MAX_POINTER_VELOCITY) {
        const scale = MAX_POINTER_VELOCITY / magnitude;
        velocityX *= scale;
        velocityY *= scale;
      }
      pointer.velocityX = velocityX;
      pointer.velocityY = velocityY;
      pointer.currentX = nextX;
      pointer.currentY = nextY;
      pointer.hasPendingMotion = true;
    }

    function toggleDemo() {
      if (!app.available) {
        return;
      }
      if (app.demo.active) {
        app.demo.active = false;
        resetPointerState();
        demoButton.textContent = "Auto demo";
        setStatus("Demo stopped. Drag anywhere on the canvas to take over.", "good");
        return;
      }
      app.replay.active = false;
      resetPointerState();
      app.demo.active = true;
      app.pointer.active = true;
      demoButton.textContent = "Stop demo";
      setStatus("Auto demo is painting a figure-eight flow. Drag to take over.", "good");
      if (app.paused) {
        togglePause();
      }
    }

    function saveSnapshot() {
      if (!app.available) {
        return;
      }
      canvas.toBlob((blob) => {
        if (!blob) {
          setStatus("The browser could not create a PNG snapshot.", "warning");
          return;
        }
        const downloadUrl = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = downloadUrl;
        link.download = `stable-fluids-${new Date().toISOString().replace(/[:.]/g, "-")}.png`;
        link.click();
        window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 0);
        setStatus("PNG snapshot saved locally.", "good");
      }, "image/png");
    }

    function saveDiagnostics() {
      if (!app.available || !app.adapter) {
        return;
      }
      const report = createDiagnosticsReport({
        grid: [GRID_WIDTH, GRID_HEIGHT],
        pressureIterations: simulationSettings.pressureIterations,
        maxBacktraceDistance: MAX_BACKTRACE_DISTANCE,
        tracerCount: PARTICLE_COUNT,
        workgroup: [WORKGROUP_SIZE, WORKGROUP_SIZE, 1],
        settings: simulationSettings,
        tracersEnabled: app.particlesEnabled,
        adapterInfo: app.adapter.info || {},
        adapterLimits: app.adapter.limits || {},
        capabilities: app.capabilities,
        features: app.device ? Array.from(app.device.features || []) : [],
        runtime: snapshotRuntimeTelemetry(app.telemetry, performance.now()),
        qualityGovernor: {
          ...app.qualityGovernor.snapshot(),
          enabled: adaptiveQualityToggle.checked
        }
      });
      const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
      const downloadUrl = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = downloadUrl;
      link.download = `stable-fluids-diagnostics-${new Date().toISOString().replace(/[:.]/g, "-")}.json`;
      link.click();
      window.setTimeout(() => URL.revokeObjectURL(downloadUrl), 0);
      setStatus("Diagnostics JSON saved locally.", "good");
    }

    function togglePause() {
      if (!app.available) {
        return;
      }
      app.paused = !app.paused;
      updatePauseButton();
      if (app.paused) {
        stopAnimationLoop();
        setStatus("Paused. Press Resume or Space to continue without a time-step jump.", "good");
        renderCurrentDensity();
      } else {
        app.lastFrameTime = null;
        setStatus("Running. Drag anywhere on the canvas to inject ink and motion.", "good");
        startAnimationLoop();
      }
    }

    function clearSimulation() {
      if (!app.available) {
        return;
      }
      if (app.demo.active) {
        app.demo.active = false;
        demoButton.textContent = "Auto demo";
        resetPointerState();
      }
      app.replay.active = false;
      clearSimulationTextures();
      setStatus("Simulation cleared. Drag to paint a new field.", "good");
      renderCurrentDensity();
    }

    // 12. Resize and device-loss handling
    function configurePresentationCanvas() {
      if (!app.device || !app.context || !app.canvasFormat) {
        return;
      }
      const rectangle = canvas.getBoundingClientRect();
      const cssWidth = Math.max(rectangle.width || window.innerWidth || 1, 1);
      const cssHeight = Math.max(rectangle.height || window.innerHeight || 1, 1);
      // Cap presentation DPR at 2: the simulation remains 512² while display cost stays bounded on ultra-DPI screens.
      const effectiveDpr = Math.min(window.devicePixelRatio || 1, MAX_EFFECTIVE_DPR);
      const maxTextureDimension = app.adapter?.limits?.maxTextureDimension2D || 8192;
      const width = Math.max(1, Math.min(maxTextureDimension, Math.floor(cssWidth * effectiveDpr)));
      const height = Math.max(1, Math.min(maxTextureDimension, Math.floor(cssHeight * effectiveDpr)));
      if (canvas.width === width && canvas.height === height && app.presentationWidth === width && app.presentationHeight === height) {
        return;
      }

      canvas.width = width;
      canvas.height = height;
      app.presentationWidth = width;
      app.presentationHeight = height;
      app.context.configure({ device: app.device, format: app.canvasFormat, alphaMode: "opaque" });
    }

    function handleResize() {
      try {
        configurePresentationCanvas();
      } catch (error) {
        console.error("Canvas resize failed:", error);
        setStatus("Canvas resizing failed. Use Retry GPU to rebuild the presentation device.", "danger");
      }
    }

    function setupResizeHandling() {
      if (typeof ResizeObserver === "function") {
        app.resizeObserver = new ResizeObserver(handleResize);
        app.resizeObserver.observe(canvas);
      }
      window.addEventListener("resize", handleResize, { passive: true });
    }

    function handleVisibilityChange() {
      app.pageHidden = document.visibilityState === "hidden";
      app.lastFrameTime = null;
      app.pointer.velocityX = 0;
      app.pointer.velocityY = 0;
      app.pointer.hasPendingMotion = false;
      if (app.pageHidden) {
        stopAnimationLoop();
      } else if (app.available && !app.paused) {
        startAnimationLoop();
      }
    }

    function handleKeyboardShortcut(event) {
      const target = event.target;
      if (target instanceof HTMLInputElement && (target.type === "range" || target.type === "color")) {
        return;
      }
      if (event.code === "Space") {
        event.preventDefault();
        togglePause();
      } else if (event.key.toLowerCase() === "c" || event.key.toLowerCase() === "r") {
        clearSimulation();
      } else if (event.key.toLowerCase() === "h") {
        setControlsVisible(controls.classList.contains("is-hidden"));
      }
    }

    function setControlsVisible(visible) {
      controls.classList.toggle("is-hidden", !visible);
      panelToggle.setAttribute("aria-expanded", String(visible));
      panelToggle.textContent = visible ? "Hide controls" : "Controls";
      panelToggle.hidden = visible;
      hideControlsButton.setAttribute("aria-label", visible ? "Hide controls" : "Show controls");
    }

    function setupInteractionAndUi() {
      restoreStoredSettings();
      updateControlReadouts();
      updateTracerState();
      updateHudState();
      updateRecordingStatus();
      for (const input of [inkColorInput, brushRadiusInput, velocityForceInput, inkAmountInput, velocityDissipationInput, inkDissipationInput, vorticityForceInput, renderModeInput, brushModeInput]) {
        input.addEventListener("input", updateControlReadouts);
      }
      scenePresetInput.addEventListener("change", applyScenePreset);
      qualityProfileInput.addEventListener("change", updateControlReadouts);
      adaptiveQualityToggle.addEventListener("change", updateAdaptiveQualityState);
      tracerToggle.addEventListener("change", updateTracerState);
      hudToggle.addEventListener("change", updateHudState);
      rememberToggle.addEventListener("change", () => {
        if (rememberToggle.checked) {
          persistSettingsIfEnabled();
          setStatus("Lab settings will be restored on the next visit.", "good");
        } else {
          clearSettings(getSettingsStorage());
          setStatus("Saved lab settings were removed.", "good");
        }
      });
      for (const presetButton of colorPresetButtons) {
        presetButton.addEventListener("click", selectInkPreset);
      }
      pauseButton.addEventListener("click", togglePause);
      clearButton.addEventListener("click", clearSimulation);
      demoButton.addEventListener("click", toggleDemo);
      snapshotButton.addEventListener("click", saveSnapshot);
      recordButton.addEventListener("click", toggleRecording);
      replayButton.addEventListener("click", replayRecording);
      diagnosticsButton.addEventListener("click", saveDiagnostics);
      forgetSettingsButton.addEventListener("click", () => {
        clearSettings(getSettingsStorage());
        rememberToggle.checked = false;
        setStatus("Saved lab settings forgotten.", "good");
      });
      retryGpuButton.addEventListener("click", initializeGpu);
      hideControlsButton.addEventListener("click", () => setControlsVisible(false));
      panelToggle.addEventListener("click", () => setControlsVisible(true));

      canvas.addEventListener("pointerdown", handlePointerDown);
      canvas.addEventListener("pointermove", handlePointerMove);
      canvas.addEventListener("pointerup", finishPointerInteraction);
      canvas.addEventListener("pointercancel", handlePointerCancel);
      canvas.addEventListener("lostpointercapture", (event) => {
        if (app.pointer.active && event.pointerId === app.pointer.pointerId) {
          resetPointerState();
        }
      });
      document.addEventListener("visibilitychange", handleVisibilityChange);
      window.addEventListener("keydown", handleKeyboardShortcut);
      setupResizeHandling();
    }

    // 14. Startup
    setupInteractionAndUi();
    resolutionStatus.textContent = `${GRID_WIDTH} × ${GRID_HEIGHT}`;
    pressureStatus.textContent = `${simulationSettings.pressureIterations} iterations`;
    updatePauseButton();
    setGpuControlsDisabled(true);
    initializeGpu();
