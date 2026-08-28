import { MAX_BACKTRACE_DISTANCE, PARTICLE_COUNT } from "../config/simulation.js";

export const splatShaderCode = `
      struct SimParams {
        timeDtDissipation: vec4<f32>,
        gridPointer: vec4<f32>,
        strokeStart: vec4<f32>,
        strokeEnd: vec4<f32>,
        injectedVelocityForce: vec4<f32>,
        inkColorExposure: vec4<f32>,
        vortexSettings: vec4<f32>,
      };

      @group(0) @binding(0) var<uniform> params: SimParams;
      @group(0) @binding(1) var velocitySource: texture_2d<f32>;
      @group(0) @binding(2) var densitySource: texture_2d<f32>;
      @group(0) @binding(3) var velocityDestination: texture_storage_2d<rg32float, write>;
      @group(0) @binding(4) var densityDestination: texture_storage_2d<rgba16float, write>;

      fn gridSize() -> vec2<i32> {
        return vec2<i32>(i32(params.gridPointer.x), i32(params.gridPointer.y));
      }

      fn clampCoordinate(coordinate: vec2<i32>) -> vec2<i32> {
        return clamp(coordinate, vec2<i32>(0), gridSize() - vec2<i32>(1));
      }

      fn loadVelocity(coordinate: vec2<i32>) -> vec2<f32> {
        return textureLoad(velocitySource, clampCoordinate(coordinate), 0).xy;
      }

      fn loadDensity(coordinate: vec2<i32>) -> vec4<f32> {
        return textureLoad(densitySource, clampCoordinate(coordinate), 0);
      }

      fn distanceToSegment(point: vec2<f32>, start: vec2<f32>, end: vec2<f32>) -> f32 {
        let segment = end - start;
        let segmentLengthSquared = dot(segment, segment);
        if (segmentLengthSquared <= 0.000001) {
          return distance(point, start);
        }
        let projection = clamp(dot(point - start, segment) / segmentLengthSquared, 0.0, 1.0);
        return distance(point, start + projection * segment);
      }

      @compute @workgroup_size(16, 16, 1)
      fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
        let size = gridSize();
        if (globalId.x >= u32(size.x) || globalId.y >= u32(size.y)) {
          return;
        }

        let coordinate = vec2<i32>(globalId.xy);
        var velocity = loadVelocity(coordinate);
        var density = loadDensity(coordinate);

        if (params.gridPointer.z > 0.5) {
          let point = vec2<f32>(coordinate) + vec2<f32>(0.5);
          let strokeStart = params.strokeStart.xy;
          let strokeEnd = params.strokeEnd.xy;
          let radius = max(params.gridPointer.w, 1.0);
          let distanceToStroke = distanceToSegment(point, strokeStart, strokeEnd);
          let radialFalloff = exp(-(distanceToStroke * distanceToStroke) / (2.0 * radius * radius));
          let injectedVelocity = params.injectedVelocityForce.xy * params.injectedVelocityForce.z * radialFalloff;
          let injectedInk = params.inkColorExposure.xyz * params.injectedVelocityForce.w * radialFalloff;
          let brushMode = params.vortexSettings.y;
          if (brushMode < 0.5) {
            velocity += injectedVelocity;
            density = vec4<f32>(
              clamp(density.rgb + injectedInk, vec3<f32>(0.0), vec3<f32>(12.0)),
              clamp(density.a + params.injectedVelocityForce.w * radialFalloff, 0.0, 12.0)
            );
          } else if (brushMode < 1.5) {
            velocity += injectedVelocity;
          } else {
            let eraseFactor = clamp(1.0 - radialFalloff * 1.35, 0.0, 1.0);
            velocity *= eraseFactor;
            density *= eraseFactor;
          }
        }

        velocity = clamp(velocity, vec2<f32>(-4096.0), vec2<f32>(4096.0));
        textureStore(velocityDestination, coordinate, vec4<f32>(velocity, 0.0, 0.0));
        textureStore(densityDestination, coordinate, density);
      }
    `;
export const advectionShaderCode = `
      struct SimParams {
        timeDtDissipation: vec4<f32>,
        gridPointer: vec4<f32>,
        strokeStart: vec4<f32>,
        strokeEnd: vec4<f32>,
        injectedVelocityForce: vec4<f32>,
        inkColorExposure: vec4<f32>,
        vortexSettings: vec4<f32>,
      };

      @group(0) @binding(0) var<uniform> params: SimParams;
      @group(0) @binding(1) var velocitySource: texture_2d<f32>;
      @group(0) @binding(2) var densitySource: texture_2d<f32>;
      @group(0) @binding(3) var velocityDestination: texture_storage_2d<rg32float, write>;
      @group(0) @binding(4) var densityDestination: texture_storage_2d<rgba16float, write>;

      fn gridSize() -> vec2<i32> {
        return vec2<i32>(i32(params.gridPointer.x), i32(params.gridPointer.y));
      }

      fn clampCoordinate(coordinate: vec2<i32>) -> vec2<i32> {
        return clamp(coordinate, vec2<i32>(0), gridSize() - vec2<i32>(1));
      }

      fn loadVelocity(coordinate: vec2<i32>) -> vec2<f32> {
        return textureLoad(velocitySource, clampCoordinate(coordinate), 0).xy;
      }

      fn loadDensity(coordinate: vec2<i32>) -> vec4<f32> {
        return textureLoad(densitySource, clampCoordinate(coordinate), 0);
      }

      fn sampleVelocity(position: vec2<f32>) -> vec2<f32> {
        let maximumPosition = vec2<f32>(params.gridPointer.xy) - vec2<f32>(0.5);
        let safePosition = clamp(position, vec2<f32>(0.5), maximumPosition);
        let lower = floor(safePosition - vec2<f32>(0.5));
        let fraction = safePosition - (lower + vec2<f32>(0.5));
        let lowerCoordinate = clamp(vec2<i32>(lower), vec2<i32>(0), gridSize() - vec2<i32>(1));
        let upperCoordinate = clamp(lowerCoordinate + vec2<i32>(1), vec2<i32>(0), gridSize() - vec2<i32>(1));
        let lowerLeft = loadVelocity(lowerCoordinate);
        let lowerRight = loadVelocity(vec2<i32>(upperCoordinate.x, lowerCoordinate.y));
        let upperLeft = loadVelocity(vec2<i32>(lowerCoordinate.x, upperCoordinate.y));
        let upperRight = loadVelocity(upperCoordinate);
        let horizontalLower = mix(lowerLeft, lowerRight, fraction.x);
        let horizontalUpper = mix(upperLeft, upperRight, fraction.x);
        return mix(horizontalLower, horizontalUpper, fraction.y);
      }

      fn sampleDensity(position: vec2<f32>) -> vec4<f32> {
        let maximumPosition = vec2<f32>(params.gridPointer.xy) - vec2<f32>(0.5);
        let safePosition = clamp(position, vec2<f32>(0.5), maximumPosition);
        let lower = floor(safePosition - vec2<f32>(0.5));
        let fraction = safePosition - (lower + vec2<f32>(0.5));
        let lowerCoordinate = clamp(vec2<i32>(lower), vec2<i32>(0), gridSize() - vec2<i32>(1));
        let upperCoordinate = clamp(lowerCoordinate + vec2<i32>(1), vec2<i32>(0), gridSize() - vec2<i32>(1));
        let lowerLeft = loadDensity(lowerCoordinate);
        let lowerRight = loadDensity(vec2<i32>(upperCoordinate.x, lowerCoordinate.y));
        let upperLeft = loadDensity(vec2<i32>(lowerCoordinate.x, upperCoordinate.y));
        let upperRight = loadDensity(upperCoordinate);
        let horizontalLower = mix(lowerLeft, lowerRight, fraction.x);
        let horizontalUpper = mix(upperLeft, upperRight, fraction.x);
        return mix(horizontalLower, horizontalUpper, fraction.y);
      }

      @compute @workgroup_size(16, 16, 1)
      fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
        let size = gridSize();
        if (globalId.x >= u32(size.x) || globalId.y >= u32(size.y)) {
          return;
        }

        let coordinate = vec2<i32>(globalId.xy);
        let point = vec2<f32>(coordinate) + vec2<f32>(0.5);
        let currentVelocity = loadVelocity(coordinate);
        var backtraceDisplacement = params.timeDtDissipation.y * currentVelocity;
        let backtraceLength = length(backtraceDisplacement);
        if (backtraceLength > ${MAX_BACKTRACE_DISTANCE}.0) {
          backtraceDisplacement *= ${MAX_BACKTRACE_DISTANCE}.0 / backtraceLength;
        }
        let backTracedPosition = point - backtraceDisplacement;
        let advectedVelocity = sampleVelocity(backTracedPosition);
        let advectedDensity = sampleDensity(backTracedPosition);
        let velocityDecay = exp(-params.timeDtDissipation.z * params.timeDtDissipation.y);
        let densityDecay = exp(-params.timeDtDissipation.w * params.timeDtDissipation.y);

        textureStore(velocityDestination, coordinate, vec4<f32>(clamp(advectedVelocity * velocityDecay, vec2<f32>(-4096.0), vec2<f32>(4096.0)), 0.0, 0.0));
        textureStore(densityDestination, coordinate, clamp(advectedDensity * densityDecay, vec4<f32>(0.0), vec4<f32>(12.0)));
      }
    `;
export const divergenceShaderCode = `
      struct SimParams {
        timeDtDissipation: vec4<f32>,
        gridPointer: vec4<f32>,
        strokeStart: vec4<f32>,
        strokeEnd: vec4<f32>,
        injectedVelocityForce: vec4<f32>,
        inkColorExposure: vec4<f32>,
        vortexSettings: vec4<f32>,
      };

      @group(0) @binding(0) var<uniform> params: SimParams;
      @group(0) @binding(1) var velocitySource: texture_2d<f32>;
      @group(0) @binding(2) var divergenceDestination: texture_storage_2d<r32float, write>;

      fn gridSize() -> vec2<i32> {
        return vec2<i32>(i32(params.gridPointer.x), i32(params.gridPointer.y));
      }

      fn clampCoordinate(coordinate: vec2<i32>) -> vec2<i32> {
        return clamp(coordinate, vec2<i32>(0), gridSize() - vec2<i32>(1));
      }

      fn loadWallVelocity(coordinate: vec2<i32>) -> vec2<f32> {
        var velocity = textureLoad(velocitySource, clampCoordinate(coordinate), 0).xy;
        if (coordinate.x < 0 || coordinate.x >= gridSize().x) {
          velocity = vec2<f32>(-velocity.x, velocity.y);
        }
        if (coordinate.y < 0 || coordinate.y >= gridSize().y) {
          velocity = vec2<f32>(velocity.x, -velocity.y);
        }
        return velocity;
      }

      // CUDA-style shared-memory tile: 16x16 threads cooperatively stage an 18x18 halo tile.
      var<workgroup> velocityTile: array<vec2<f32>, 324>;

      fn tileIndex(x: u32, y: u32) -> u32 {
        return y * 18u + x;
      }

      @compute @workgroup_size(16, 16, 1)
      fn main(@builtin(global_invocation_id) globalId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
        let size = gridSize();
        let inBounds = globalId.x < u32(size.x) && globalId.y < u32(size.y);
        var centerVelocity = vec2<f32>(0.0);
        if (inBounds) {
          centerVelocity = textureLoad(velocitySource, vec2<i32>(globalId.xy), 0).xy;
        }
        let tileX = localId.x + 1u;
        let tileY = localId.y + 1u;
        velocityTile[tileIndex(tileX, tileY)] = centerVelocity;

        if (inBounds && localId.x == 0u) {
          if (globalId.x == 0u) {
            velocityTile[tileIndex(0u, tileY)] = vec2<f32>(-centerVelocity.x, centerVelocity.y);
          } else {
            velocityTile[tileIndex(0u, tileY)] = textureLoad(velocitySource, vec2<i32>(globalId.xy) + vec2<i32>(-1, 0), 0).xy;
          }
        }
        if (inBounds && localId.x == 15u) {
          if (globalId.x == u32(size.x - 1)) {
            velocityTile[tileIndex(17u, tileY)] = vec2<f32>(-centerVelocity.x, centerVelocity.y);
          } else {
            velocityTile[tileIndex(17u, tileY)] = textureLoad(velocitySource, vec2<i32>(globalId.xy) + vec2<i32>(1, 0), 0).xy;
          }
        }
        if (inBounds && localId.y == 0u) {
          if (globalId.y == 0u) {
            velocityTile[tileIndex(tileX, 0u)] = vec2<f32>(centerVelocity.x, -centerVelocity.y);
          } else {
            velocityTile[tileIndex(tileX, 0u)] = textureLoad(velocitySource, vec2<i32>(globalId.xy) + vec2<i32>(0, -1), 0).xy;
          }
        }
        if (inBounds && localId.y == 15u) {
          if (globalId.y == u32(size.y - 1)) {
            velocityTile[tileIndex(tileX, 17u)] = vec2<f32>(centerVelocity.x, -centerVelocity.y);
          } else {
            velocityTile[tileIndex(tileX, 17u)] = textureLoad(velocitySource, vec2<i32>(globalId.xy) + vec2<i32>(0, 1), 0).xy;
          }
        }

        workgroupBarrier();
        if (!inBounds) {
          return;
        }
        let left = velocityTile[tileIndex(tileX - 1u, tileY)];
        let right = velocityTile[tileIndex(tileX + 1u, tileY)];
        let top = velocityTile[tileIndex(tileX, tileY - 1u)];
        let bottom = velocityTile[tileIndex(tileX, tileY + 1u)];
        let divergence = 0.5 * (right.x - left.x + bottom.y - top.y);
        textureStore(divergenceDestination, vec2<i32>(globalId.xy), vec4<f32>(divergence, 0.0, 0.0, 0.0));
      }
    `;
export const pressureShaderCode = `
      struct SimParams {
        timeDtDissipation: vec4<f32>,
        gridPointer: vec4<f32>,
        strokeStart: vec4<f32>,
        strokeEnd: vec4<f32>,
        injectedVelocityForce: vec4<f32>,
        inkColorExposure: vec4<f32>,
        vortexSettings: vec4<f32>,
      };

      @group(0) @binding(0) var<uniform> params: SimParams;
      @group(0) @binding(1) var pressureSource: texture_2d<f32>;
      @group(0) @binding(2) var divergenceSource: texture_2d<f32>;
      @group(0) @binding(3) var pressureDestination: texture_storage_2d<r32float, write>;

      fn gridSize() -> vec2<i32> {
        return vec2<i32>(i32(params.gridPointer.x), i32(params.gridPointer.y));
      }

      fn clampCoordinate(coordinate: vec2<i32>) -> vec2<i32> {
        return clamp(coordinate, vec2<i32>(0), gridSize() - vec2<i32>(1));
      }

      fn loadPressure(coordinate: vec2<i32>) -> f32 {
        return textureLoad(pressureSource, clampCoordinate(coordinate), 0).x;
      }

      // The Jacobi stencil uses a 16x16 block plus a one-texel halo in workgroup memory.
      var<workgroup> pressureTile: array<f32, 324>;

      fn tileIndex(x: u32, y: u32) -> u32 {
        return y * 18u + x;
      }

      @compute @workgroup_size(16, 16, 1)
      fn main(@builtin(global_invocation_id) globalId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
        let size = gridSize();
        let inBounds = globalId.x < u32(size.x) && globalId.y < u32(size.y);
        var centerPressure = 0.0;
        if (inBounds) {
          centerPressure = textureLoad(pressureSource, vec2<i32>(globalId.xy), 0).x;
        }
        let tileX = localId.x + 1u;
        let tileY = localId.y + 1u;
        pressureTile[tileIndex(tileX, tileY)] = centerPressure;

        if (inBounds && localId.x == 0u) {
          if (globalId.x == 0u) {
            pressureTile[tileIndex(0u, tileY)] = centerPressure;
          } else {
            pressureTile[tileIndex(0u, tileY)] = textureLoad(pressureSource, vec2<i32>(globalId.xy) + vec2<i32>(-1, 0), 0).x;
          }
        }
        if (inBounds && localId.x == 15u) {
          if (globalId.x == u32(size.x - 1)) {
            pressureTile[tileIndex(17u, tileY)] = centerPressure;
          } else {
            pressureTile[tileIndex(17u, tileY)] = textureLoad(pressureSource, vec2<i32>(globalId.xy) + vec2<i32>(1, 0), 0).x;
          }
        }
        if (inBounds && localId.y == 0u) {
          if (globalId.y == 0u) {
            pressureTile[tileIndex(tileX, 0u)] = centerPressure;
          } else {
            pressureTile[tileIndex(tileX, 0u)] = textureLoad(pressureSource, vec2<i32>(globalId.xy) + vec2<i32>(0, -1), 0).x;
          }
        }
        if (inBounds && localId.y == 15u) {
          if (globalId.y == u32(size.y - 1)) {
            pressureTile[tileIndex(tileX, 17u)] = centerPressure;
          } else {
            pressureTile[tileIndex(tileX, 17u)] = textureLoad(pressureSource, vec2<i32>(globalId.xy) + vec2<i32>(0, 1), 0).x;
          }
        }

        workgroupBarrier();
        if (!inBounds) {
          return;
        }
        let left = pressureTile[tileIndex(tileX - 1u, tileY)];
        let right = pressureTile[tileIndex(tileX + 1u, tileY)];
        let top = pressureTile[tileIndex(tileX, tileY - 1u)];
        let bottom = pressureTile[tileIndex(tileX, tileY + 1u)];
        let divergence = textureLoad(divergenceSource, vec2<i32>(globalId.xy), 0).x;
        let pressure = (left + right + top + bottom - divergence) * 0.25;
        textureStore(pressureDestination, vec2<i32>(globalId.xy), vec4<f32>(clamp(pressure, -4096.0, 4096.0), 0.0, 0.0, 0.0));
      }
    `;
export const gradientShaderCode = `
      struct SimParams {
        timeDtDissipation: vec4<f32>,
        gridPointer: vec4<f32>,
        strokeStart: vec4<f32>,
        strokeEnd: vec4<f32>,
        injectedVelocityForce: vec4<f32>,
        inkColorExposure: vec4<f32>,
        vortexSettings: vec4<f32>,
      };

      @group(0) @binding(0) var<uniform> params: SimParams;
      @group(0) @binding(1) var pressureSource: texture_2d<f32>;
      @group(0) @binding(2) var velocitySource: texture_2d<f32>;
      @group(0) @binding(3) var velocityDestination: texture_storage_2d<rg32float, write>;

      fn gridSize() -> vec2<i32> {
        return vec2<i32>(i32(params.gridPointer.x), i32(params.gridPointer.y));
      }

      fn clampCoordinate(coordinate: vec2<i32>) -> vec2<i32> {
        return clamp(coordinate, vec2<i32>(0), gridSize() - vec2<i32>(1));
      }

      fn loadPressure(coordinate: vec2<i32>) -> f32 {
        return textureLoad(pressureSource, clampCoordinate(coordinate), 0).x;
      }

      @compute @workgroup_size(16, 16, 1)
      fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
        let size = gridSize();
        if (globalId.x >= u32(size.x) || globalId.y >= u32(size.y)) {
          return;
        }

        let coordinate = vec2<i32>(globalId.xy);
        let pressureLeft = loadPressure(coordinate + vec2<i32>(-1, 0));
        let pressureRight = loadPressure(coordinate + vec2<i32>(1, 0));
        let pressureTop = loadPressure(coordinate + vec2<i32>(0, -1));
        let pressureBottom = loadPressure(coordinate + vec2<i32>(0, 1));
        var velocity = textureLoad(velocitySource, coordinate, 0).xy;
        let gradient = 0.5 * vec2<f32>(pressureRight - pressureLeft, pressureBottom - pressureTop);
        velocity -= gradient;

        if (coordinate.x == 0 || coordinate.x == size.x - 1) {
          velocity = vec2<f32>(0.0, velocity.y);
        }
        if (coordinate.y == 0 || coordinate.y == size.y - 1) {
          velocity = vec2<f32>(velocity.x, 0.0);
        }

        textureStore(velocityDestination, coordinate, vec4<f32>(clamp(velocity, vec2<f32>(-4096.0), vec2<f32>(4096.0)), 0.0, 0.0));
      }
    `;
export const vorticityShaderCode = `
      struct SimParams {
        timeDtDissipation: vec4<f32>,
        gridPointer: vec4<f32>,
        strokeStart: vec4<f32>,
        strokeEnd: vec4<f32>,
        injectedVelocityForce: vec4<f32>,
        inkColorExposure: vec4<f32>,
        vortexSettings: vec4<f32>,
      };

      @group(0) @binding(0) var<uniform> params: SimParams;
      @group(0) @binding(1) var velocitySource: texture_2d<f32>;
      @group(0) @binding(2) var vorticityDestination: texture_storage_2d<r32float, write>;

      fn gridSize() -> vec2<i32> {
        return vec2<i32>(i32(params.gridPointer.x), i32(params.gridPointer.y));
      }

      fn clampCoordinate(coordinate: vec2<i32>) -> vec2<i32> {
        return clamp(coordinate, vec2<i32>(0), gridSize() - vec2<i32>(1));
      }

      fn loadWallVelocity(coordinate: vec2<i32>) -> vec2<f32> {
        var velocity = textureLoad(velocitySource, clampCoordinate(coordinate), 0).xy;
        if (coordinate.x < 0 || coordinate.x >= gridSize().x) {
          velocity = vec2<f32>(-velocity.x, velocity.y);
        }
        if (coordinate.y < 0 || coordinate.y >= gridSize().y) {
          velocity = vec2<f32>(velocity.x, -velocity.y);
        }
        return velocity;
      }

      var<workgroup> velocityTile: array<vec2<f32>, 324>;

      fn tileIndex(x: u32, y: u32) -> u32 {
        return y * 18u + x;
      }

      @compute @workgroup_size(16, 16, 1)
      fn main(@builtin(global_invocation_id) globalId: vec3<u32>, @builtin(local_invocation_id) localId: vec3<u32>) {
        let size = gridSize();
        let inBounds = globalId.x < u32(size.x) && globalId.y < u32(size.y);
        var centerVelocity = vec2<f32>(0.0);
        if (inBounds) {
          centerVelocity = textureLoad(velocitySource, vec2<i32>(globalId.xy), 0).xy;
        }
        let tileX = localId.x + 1u;
        let tileY = localId.y + 1u;
        velocityTile[tileIndex(tileX, tileY)] = centerVelocity;

        if (inBounds && localId.x == 0u) {
          if (globalId.x == 0u) {
            velocityTile[tileIndex(0u, tileY)] = vec2<f32>(-centerVelocity.x, centerVelocity.y);
          } else {
            velocityTile[tileIndex(0u, tileY)] = textureLoad(velocitySource, vec2<i32>(globalId.xy) + vec2<i32>(-1, 0), 0).xy;
          }
        }
        if (inBounds && localId.x == 15u) {
          if (globalId.x == u32(size.x - 1)) {
            velocityTile[tileIndex(17u, tileY)] = vec2<f32>(-centerVelocity.x, centerVelocity.y);
          } else {
            velocityTile[tileIndex(17u, tileY)] = textureLoad(velocitySource, vec2<i32>(globalId.xy) + vec2<i32>(1, 0), 0).xy;
          }
        }
        if (inBounds && localId.y == 0u) {
          if (globalId.y == 0u) {
            velocityTile[tileIndex(tileX, 0u)] = vec2<f32>(centerVelocity.x, -centerVelocity.y);
          } else {
            velocityTile[tileIndex(tileX, 0u)] = textureLoad(velocitySource, vec2<i32>(globalId.xy) + vec2<i32>(0, -1), 0).xy;
          }
        }
        if (inBounds && localId.y == 15u) {
          if (globalId.y == u32(size.y - 1)) {
            velocityTile[tileIndex(tileX, 17u)] = vec2<f32>(centerVelocity.x, -centerVelocity.y);
          } else {
            velocityTile[tileIndex(tileX, 17u)] = textureLoad(velocitySource, vec2<i32>(globalId.xy) + vec2<i32>(0, 1), 0).xy;
          }
        }

        workgroupBarrier();
        if (!inBounds) {
          return;
        }
        let left = velocityTile[tileIndex(tileX - 1u, tileY)];
        let right = velocityTile[tileIndex(tileX + 1u, tileY)];
        let top = velocityTile[tileIndex(tileX, tileY - 1u)];
        let bottom = velocityTile[tileIndex(tileX, tileY + 1u)];
        let curl = 0.5 * ((right.y - left.y) - (bottom.x - top.x));
        textureStore(vorticityDestination, vec2<i32>(globalId.xy), vec4<f32>(curl, 0.0, 0.0, 0.0));
      }
    `;
export const confinementShaderCode = `
      struct SimParams {
        timeDtDissipation: vec4<f32>,
        gridPointer: vec4<f32>,
        strokeStart: vec4<f32>,
        strokeEnd: vec4<f32>,
        injectedVelocityForce: vec4<f32>,
        inkColorExposure: vec4<f32>,
        vortexSettings: vec4<f32>,
      };

      @group(0) @binding(0) var<uniform> params: SimParams;
      @group(0) @binding(1) var velocitySource: texture_2d<f32>;
      @group(0) @binding(2) var vorticitySource: texture_2d<f32>;
      @group(0) @binding(3) var velocityDestination: texture_storage_2d<rg32float, write>;

      fn gridSize() -> vec2<i32> {
        return vec2<i32>(i32(params.gridPointer.x), i32(params.gridPointer.y));
      }

      fn clampCoordinate(coordinate: vec2<i32>) -> vec2<i32> {
        return clamp(coordinate, vec2<i32>(0), gridSize() - vec2<i32>(1));
      }

      fn loadVelocity(coordinate: vec2<i32>) -> vec2<f32> {
        return textureLoad(velocitySource, clampCoordinate(coordinate), 0).xy;
      }

      fn loadVorticity(coordinate: vec2<i32>) -> f32 {
        return textureLoad(vorticitySource, clampCoordinate(coordinate), 0).x;
      }

      @compute @workgroup_size(16, 16, 1)
      fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
        let size = gridSize();
        if (globalId.x >= u32(size.x) || globalId.y >= u32(size.y)) {
          return;
        }

        let coordinate = vec2<i32>(globalId.xy);
        let leftMagnitude = abs(loadVorticity(coordinate + vec2<i32>(-1, 0)));
        let rightMagnitude = abs(loadVorticity(coordinate + vec2<i32>(1, 0)));
        let topMagnitude = abs(loadVorticity(coordinate + vec2<i32>(0, -1)));
        let bottomMagnitude = abs(loadVorticity(coordinate + vec2<i32>(0, 1)));
        let magnitudeGradient = vec2<f32>(rightMagnitude - leftMagnitude, bottomMagnitude - topMagnitude);
        let normal = magnitudeGradient / max(length(magnitudeGradient), 0.0001);
        let curl = loadVorticity(coordinate);
        let confinementForce = params.vortexSettings.x * vec2<f32>(normal.y, -normal.x) * curl;
        var velocity = loadVelocity(coordinate) + confinementForce * params.timeDtDissipation.y;

        if (coordinate.x == 0 || coordinate.x == size.x - 1) {
          velocity = vec2<f32>(0.0, velocity.y);
        }
        if (coordinate.y == 0 || coordinate.y == size.y - 1) {
          velocity = vec2<f32>(velocity.x, 0.0);
        }
        textureStore(velocityDestination, coordinate, vec4<f32>(clamp(velocity, vec2<f32>(-4096.0), vec2<f32>(4096.0)), 0.0, 0.0));
      }
    `;
export const particleComputeShaderCode = `
      struct SimParams {
        timeDtDissipation: vec4<f32>,
        gridPointer: vec4<f32>,
        strokeStart: vec4<f32>,
        strokeEnd: vec4<f32>,
        injectedVelocityForce: vec4<f32>,
        inkColorExposure: vec4<f32>,
        vortexSettings: vec4<f32>,
      };

      struct Particle {
        position: vec2<f32>,
        age: f32,
        seed: f32,
      };

      @group(0) @binding(0) var<uniform> params: SimParams;
      @group(0) @binding(1) var velocityTexture: texture_2d<f32>;
      @group(0) @binding(2) var<storage, read> particleSource: array<Particle>;
      @group(0) @binding(3) var<storage, read_write> particleDestination: array<Particle>;

      fn hash(value: f32) -> f32 {
        return fract(sin(value * 12.9898) * 43758.5453);
      }

      @compute @workgroup_size(64, 1, 1)
      fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
        if (globalId.x >= ${PARTICLE_COUNT}u) {
          return;
        }

        let particle = particleSource[globalId.x];
        let grid = params.gridPointer.xy;
        let coordinate = clamp(vec2<i32>(particle.position), vec2<i32>(0), vec2<i32>(i32(grid.x - 1.0), i32(grid.y - 1.0)));
        let velocity = textureLoad(velocityTexture, coordinate, 0).xy;
        let deltaTime = params.timeDtDissipation.y;
        var nextPosition = particle.position + velocity * deltaTime * 0.5;
        var nextAge = particle.age + deltaTime;
        let lifetime = 2.0 + hash(particle.seed * 31.7) * 5.0;
        let escaped = nextPosition.x < 0.5 || nextPosition.x >= grid.x - 0.5 || nextPosition.y < 0.5 || nextPosition.y >= grid.y - 0.5;

        if (escaped || nextAge >= lifetime) {
          let spawnTime = params.timeDtDissipation.x * 0.17;
          nextPosition = vec2<f32>(
            0.5 + hash(particle.seed * 17.1 + spawnTime) * (grid.x - 1.0),
            0.5 + hash(particle.seed * 29.3 + spawnTime + 4.0) * (grid.y - 1.0)
          );
          nextAge = 0.0;
        }

        particleDestination[globalId.x] = Particle(nextPosition, nextAge, particle.seed);
      }
    `;
export const renderVertexShaderCode = `
      struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) uv: vec2<f32>,
      };

      @vertex
      fn main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
        var positions = array<vec2<f32>, 6>(
          vec2<f32>(-1.0, -1.0),
          vec2<f32>(1.0, -1.0),
          vec2<f32>(-1.0, 1.0),
          vec2<f32>(-1.0, 1.0),
          vec2<f32>(1.0, -1.0),
          vec2<f32>(1.0, 1.0)
        );
        let position = positions[vertexIndex];
        var output: VertexOutput;
        output.position = vec4<f32>(position, 0.0, 1.0);
        output.uv = vec2<f32>((position.x + 1.0) * 0.5, (1.0 - position.y) * 0.5);
        return output;
      }
    `;
export const renderFragmentShaderCode = `
      struct SimParams {
        timeDtDissipation: vec4<f32>,
        gridPointer: vec4<f32>,
        strokeStart: vec4<f32>,
        strokeEnd: vec4<f32>,
        injectedVelocityForce: vec4<f32>,
        inkColorExposure: vec4<f32>,
        vortexSettings: vec4<f32>,
      };

      @group(0) @binding(0) var densityTexture: texture_2d<f32>;
      @group(0) @binding(1) var velocityTexture: texture_2d<f32>;
      @group(0) @binding(2) var pressureTexture: texture_2d<f32>;
      @group(0) @binding(3) var divergenceTexture: texture_2d<f32>;
      @group(0) @binding(4) var vorticityTexture: texture_2d<f32>;
      @group(0) @binding(5) var linearSampler: sampler;
      @group(0) @binding(6) var<uniform> params: SimParams;

      fn signedFieldColor(value: f32, scale: f32) -> vec3<f32> {
        let normalized = clamp(value * scale, -1.0, 1.0);
        let positive = max(normalized, 0.0);
        let negative = max(-normalized, 0.0);
        return vec3<f32>(0.16 + positive * 0.84, 0.08 + (1.0 - abs(normalized)) * 0.12, 0.22 + negative * 0.78);
      }

      @fragment
      fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
        let density = max(textureSample(densityTexture, linearSampler, uv).rgb, vec3<f32>(0.0));
        let intensity = max(max(density.r, density.g), density.b);
        let chroma = density / max(intensity, 0.0001);
        let coordinate = clamp(vec2<i32>(uv * vec2<f32>(params.gridPointer.xy)), vec2<i32>(0), vec2<i32>(i32(params.gridPointer.x) - 1, i32(params.gridPointer.y) - 1));
        let velocity = textureLoad(velocityTexture, coordinate, 0).xy;
        let pressure = textureLoad(pressureTexture, coordinate, 0).x;
        let divergence = textureLoad(divergenceTexture, coordinate, 0).x;
        let vorticity = textureLoad(vorticityTexture, coordinate, 0).x;
        var displayColor = chroma * (1.0 - exp(-intensity * params.inkColorExposure.w));
        if (params.vortexSettings.z > 0.5) {
          let mode = i32(params.vortexSettings.w);
          if (mode == 1) {
            let speed = clamp(length(velocity) * 0.045, 0.0, 1.0);
            displayColor = vec3<f32>(speed * 0.9, speed * 0.28 + 0.04, 1.0 - speed * 0.62);
          } else if (mode == 2) {
            displayColor = signedFieldColor(pressure, 0.08);
          } else if (mode == 3) {
            displayColor = signedFieldColor(divergence, 0.65);
          } else if (mode == 4) {
            let curl = clamp(abs(vorticity) * 0.18, 0.0, 1.0);
            displayColor = vec3<f32>(curl * 0.95, 0.12 + curl * 0.48, 0.32 + (1.0 - curl) * 0.68);
          }
        }
        let mappedDisplay = clamp(displayColor, vec3<f32>(0.0), vec3<f32>(1.0));
        let gammaCorrected = pow(mappedDisplay, vec3<f32>(1.0 / 2.2));
        return vec4<f32>(gammaCorrected, 1.0);
      }
    `;
export const particleVertexShaderCode = `
      struct SimParams {
        timeDtDissipation: vec4<f32>,
        gridPointer: vec4<f32>,
        strokeStart: vec4<f32>,
        strokeEnd: vec4<f32>,
        injectedVelocityForce: vec4<f32>,
        inkColorExposure: vec4<f32>,
        vortexSettings: vec4<f32>,
      };

      struct Particle {
        position: vec2<f32>,
        age: f32,
        seed: f32,
      };

      struct VertexOutput {
        @builtin(position) position: vec4<f32>,
        @location(0) color: vec4<f32>,
      };

      @group(0) @binding(0) var<uniform> params: SimParams;
      @group(0) @binding(1) var<storage, read> particles: array<Particle>;

      @vertex
      fn main(@builtin(vertex_index) vertexIndex: u32, @builtin(instance_index) instanceIndex: u32) -> VertexOutput {
        var positions = array<vec2<f32>, 6>(
          vec2<f32>(-1.0, -1.0),
          vec2<f32>(1.0, -1.0),
          vec2<f32>(-1.0, 1.0),
          vec2<f32>(-1.0, 1.0),
          vec2<f32>(1.0, -1.0),
          vec2<f32>(1.0, 1.0)
        );
        let particle = particles[instanceIndex];
        let grid = params.gridPointer.xy;
        let clipCenter = vec2<f32>(particle.position.x / grid.x * 2.0 - 1.0, 1.0 - particle.position.y / grid.y * 2.0);
        let quadSize = vec2<f32>(3.0 / grid.x, 3.0 / grid.y);
        let hue = fract(particle.seed * 7.31);
        let color = mix(vec3<f32>(0.32, 0.86, 1.0), vec3<f32>(1.0, 0.38, 0.78), hue);
        let fade = 0.14 + 0.18 * (1.0 - smoothstep(0.0, 7.0, particle.age));
        var output: VertexOutput;
        output.position = vec4<f32>(clipCenter + positions[vertexIndex] * quadSize, 0.0, 1.0);
        output.color = vec4<f32>(color, fade);
        return output;
      }
    `;
export const particleFragmentShaderCode = `
      @fragment
      fn main(@location(0) color: vec4<f32>) -> @location(0) vec4<f32> {
        return color;
      }
    `;

export const indirectArgsShaderCode = `
      @group(0) @binding(0) var<storage, read_write> drawArguments: array<u32>;

      @compute @workgroup_size(1, 1, 1)
      fn main(@builtin(global_invocation_id) globalId: vec3<u32>) {
        if (globalId.x != 0u) {
          return;
        }
        // GPU-generated DrawIndirectArguments: vertexCount, instanceCount, firstVertex, firstInstance.
        drawArguments[0] = 6u;
        drawArguments[1] = ${PARTICLE_COUNT}u;
        drawArguments[2] = 0u;
        drawArguments[3] = 0u;
      }
    `;
