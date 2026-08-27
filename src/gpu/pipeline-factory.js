/**
 * Pipeline creation policy.
 *
 * Modern WebGPU implementations can compile asynchronously, avoiding a large
 * main-thread pause while the shader graph is initialized. The synchronous
 * methods remain a deliberate compatibility fallback for older browsers.
 */
export function createGpuPipeline(device, kind, descriptor) {
  if (kind !== "compute" && kind !== "render") {
    throw new TypeError(`Unsupported WebGPU pipeline kind: ${kind}`);
  }
  const asyncFactory = kind === "compute" ? device.createComputePipelineAsync : device.createRenderPipelineAsync;
  const syncFactory = kind === "compute" ? device.createComputePipeline : device.createRenderPipeline;
  const factory = typeof asyncFactory === "function" ? asyncFactory : syncFactory;
  if (typeof factory !== "function") {
    throw new Error(`WebGPU device cannot create ${kind} pipelines.`);
  }
  return factory.call(device, descriptor);
}

export function createGpuPipelineBatch(device, definitions) {
  return Promise.all(definitions.map(({ kind, descriptor }) => createGpuPipeline(device, kind, descriptor)));
}
