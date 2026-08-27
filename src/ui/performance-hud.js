/**
 * Small DOM component for live GPU performance signals.
 * It receives already-aggregated values and never reads GPU resources itself.
 */
export function createPerformanceHud(root) {
  const fields = Object.fromEntries(
    Array.from(root.querySelectorAll("[data-hud-field]"))
      .map((field) => [field.dataset.hudField, field])
  );

  return {
    setVisible(visible) {
      root.hidden = !visible;
    },

    update({ fps, telemetry, pressureIterations, particleCount, adapter, featureTier }) {
      fields.fps.textContent = fps || "— FPS";
      fields.cpu.textContent = telemetry.frameCount > 0 ? `${telemetry.averageCpuEncodeMs.toFixed(2)} ms` : "—";
      fields.gpu.textContent = telemetry.lastGpuMs === null ? "—" : `${telemetry.averageGpuMs.toFixed(2)} ms`;
      fields.submissions.textContent = telemetry.submittedFrames.toLocaleString();
      fields.pressure.textContent = `${pressureIterations} Jacobi`;
      fields.tracers.textContent = `${particleCount.toLocaleString()} particles`;
      fields.adapter.textContent = adapter || "Detecting…";
      fields.tier.textContent = featureTier || "baseline";
    }
  };
}
