/**
 * Optional WebGPU timestamp-query support.
 *
 * The profiler deliberately owns only tiny query/readback resources and exposes
 * no synchronous wait. The simulation decides when to bracket a command buffer.
 */
export function createGpuTimer(device) {
  if (!device.features?.has?.("timestamp-query")) {
    return null;
  }
  try {
    const querySet = device.createQuerySet({ label: "Stable fluids GPU timestamp queries", type: "timestamp", count: 2 });
    const resolveBuffer = device.createBuffer({
      label: "Resolved GPU timestamps",
      size: 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC
    });
    const readbackBuffer = device.createBuffer({
      label: "Mapped GPU timestamp readback",
      size: 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
    });
    return { device, querySet, resolveBuffer, readbackBuffer, enabled: true, pending: false, frameCounter: 0, sampleEvery: 30 };
  } catch (error) {
    console.warn("Optional GPU timestamp timer unavailable:", error);
    return null;
  }
}

export function destroyGpuTimer(timer) {
  timer?.querySet?.destroy();
  timer?.resolveBuffer?.destroy();
  timer?.readbackBuffer?.destroy();
}

export async function readGpuTimestamp(timer, app, gpuTimeStatus) {
  if (!timer || timer.pending) {
    return;
  }
  timer.pending = true;
  try {
    await timer.device.queue.onSubmittedWorkDone();
    if (app.gpuTimer !== timer || app.device !== timer.device) {
      return;
    }
    await timer.readbackBuffer.mapAsync(GPUMapMode.READ);
    const timestamps = new BigUint64Array(timer.readbackBuffer.getMappedRange());
    const durationMilliseconds = Number(timestamps[1] - timestamps[0]) / 1_000_000;
    timer.readbackBuffer.unmap();
    if (app.gpuTimer === timer && app.device === timer.device && Number.isFinite(durationMilliseconds)) {
      gpuTimeStatus.textContent = `${durationMilliseconds.toFixed(2)} ms`;
      gpuTimeStatus.title = "Asynchronous timestamp-query sample for the complete simulation and render command sequence.";
    }
  } catch (error) {
    if (app.gpuTimer === timer) {
      timer.enabled = false;
      gpuTimeStatus.textContent = "Timer unavailable";
      gpuTimeStatus.title = error instanceof Error ? error.message : "GPU timestamp readback failed.";
    }
  } finally {
    timer.pending = false;
  }
}
