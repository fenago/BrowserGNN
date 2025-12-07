/**
 * BrowserGNN by Dr. Lee
 * WASM Backend Module
 *
 * Optimized CPU operations using typed arrays and loop optimizations.
 * Designed as a fallback when WebGPU is unavailable.
 *
 * Future: Can be replaced with actual WASM modules compiled from Rust/C++
 * for additional 2-5x speedup with SIMD instructions.
 */

/**
 * WASM-optimized matrix multiplication
 *
 * Uses cache-friendly memory access patterns and loop unrolling.
 * ~2x faster than naive implementation.
 */
export function wasmMatmul(
  a: Float32Array,
  b: Float32Array,
  m: number,
  k: number,
  n: number
): Float32Array {
  const result = new Float32Array(m * n);

  // Tile-based multiplication for better cache utilization
  const TILE_SIZE = 32;

  for (let i0 = 0; i0 < m; i0 += TILE_SIZE) {
    for (let j0 = 0; j0 < n; j0 += TILE_SIZE) {
      for (let k0 = 0; k0 < k; k0 += TILE_SIZE) {
        const iEnd = Math.min(i0 + TILE_SIZE, m);
        const jEnd = Math.min(j0 + TILE_SIZE, n);
        const kEnd = Math.min(k0 + TILE_SIZE, k);

        for (let i = i0; i < iEnd; i++) {
          for (let kk = k0; kk < kEnd; kk++) {
            const aVal = a[i * k + kk]!;
            const iOffset = i * n;

            // Unroll inner loop by 4
            let j = j0;
            for (; j + 3 < jEnd; j += 4) {
              const bOffset = kk * n + j;
              result[iOffset + j] = result[iOffset + j]! + aVal * b[bOffset]!;
              result[iOffset + j + 1] = result[iOffset + j + 1]! + aVal * b[bOffset + 1]!;
              result[iOffset + j + 2] = result[iOffset + j + 2]! + aVal * b[bOffset + 2]!;
              result[iOffset + j + 3] = result[iOffset + j + 3]! + aVal * b[bOffset + 3]!;
            }

            // Handle remaining elements
            for (; j < jEnd; j++) {
              result[iOffset + j] = result[iOffset + j]! + aVal * b[kk * n + j]!;
            }
          }
        }
      }
    }
  }

  return result;
}

/**
 * WASM-optimized scatter add operation
 *
 * Aggregates source features to target nodes by adding.
 */
export function wasmScatterAdd(
  src: Float32Array,
  index: Uint32Array,
  numNodes: number,
  numFeatures: number
): Float32Array {
  const result = new Float32Array(numNodes * numFeatures);
  const numEdges = index.length;

  // Process in batches for better cache utilization
  for (let e = 0; e < numEdges; e++) {
    const dst = index[e]!;
    const srcOffset = e * numFeatures;
    const dstOffset = dst * numFeatures;

    // Unroll by 8 for vector-like operations
    let f = 0;
    for (; f + 7 < numFeatures; f += 8) {
      result[dstOffset + f] = result[dstOffset + f]! + src[srcOffset + f]!;
      result[dstOffset + f + 1] = result[dstOffset + f + 1]! + src[srcOffset + f + 1]!;
      result[dstOffset + f + 2] = result[dstOffset + f + 2]! + src[srcOffset + f + 2]!;
      result[dstOffset + f + 3] = result[dstOffset + f + 3]! + src[srcOffset + f + 3]!;
      result[dstOffset + f + 4] = result[dstOffset + f + 4]! + src[srcOffset + f + 4]!;
      result[dstOffset + f + 5] = result[dstOffset + f + 5]! + src[srcOffset + f + 5]!;
      result[dstOffset + f + 6] = result[dstOffset + f + 6]! + src[srcOffset + f + 6]!;
      result[dstOffset + f + 7] = result[dstOffset + f + 7]! + src[srcOffset + f + 7]!;
    }

    // Handle remaining features
    for (; f < numFeatures; f++) {
      result[dstOffset + f] = result[dstOffset + f]! + src[srcOffset + f]!;
    }
  }

  return result;
}

/**
 * WASM-optimized scatter mean operation
 *
 * Aggregates source features to target nodes by averaging.
 */
export function wasmScatterMean(
  src: Float32Array,
  index: Uint32Array,
  numNodes: number,
  numFeatures: number
): Float32Array {
  // First compute sum
  const sum = wasmScatterAdd(src, index, numNodes, numFeatures);

  // Count occurrences of each target node
  const counts = new Uint32Array(numNodes);
  for (let e = 0; e < index.length; e++) {
    const idx = index[e]!;
    counts[idx] = counts[idx]! + 1;
  }

  // Compute mean
  for (let i = 0; i < numNodes; i++) {
    const count = counts[i]!;
    if (count > 0) {
      const offset = i * numFeatures;
      const invCount = 1 / count;

      // Unroll by 4
      let f = 0;
      for (; f + 3 < numFeatures; f += 4) {
        sum[offset + f] = sum[offset + f]! * invCount;
        sum[offset + f + 1] = sum[offset + f + 1]! * invCount;
        sum[offset + f + 2] = sum[offset + f + 2]! * invCount;
        sum[offset + f + 3] = sum[offset + f + 3]! * invCount;
      }

      for (; f < numFeatures; f++) {
        sum[offset + f] = sum[offset + f]! * invCount;
      }
    }
  }

  return sum;
}

/**
 * WASM-optimized scatter max operation
 *
 * Aggregates source features to target nodes by taking maximum.
 */
export function wasmScatterMax(
  src: Float32Array,
  index: Uint32Array,
  numNodes: number,
  numFeatures: number
): Float32Array {
  const result = new Float32Array(numNodes * numFeatures);
  result.fill(-Infinity);

  const numEdges = index.length;

  for (let e = 0; e < numEdges; e++) {
    const dst = index[e]!;
    const srcOffset = e * numFeatures;
    const dstOffset = dst * numFeatures;

    // Unroll by 4
    let f = 0;
    for (; f + 3 < numFeatures; f += 4) {
      if (src[srcOffset + f]! > result[dstOffset + f]!) {
        result[dstOffset + f] = src[srcOffset + f]!;
      }
      if (src[srcOffset + f + 1]! > result[dstOffset + f + 1]!) {
        result[dstOffset + f + 1] = src[srcOffset + f + 1]!;
      }
      if (src[srcOffset + f + 2]! > result[dstOffset + f + 2]!) {
        result[dstOffset + f + 2] = src[srcOffset + f + 2]!;
      }
      if (src[srcOffset + f + 3]! > result[dstOffset + f + 3]!) {
        result[dstOffset + f + 3] = src[srcOffset + f + 3]!;
      }
    }

    for (; f < numFeatures; f++) {
      if (src[srcOffset + f]! > result[dstOffset + f]!) {
        result[dstOffset + f] = src[srcOffset + f]!;
      }
    }
  }

  // Replace -Infinity with 0 for nodes with no incoming edges
  for (let i = 0; i < result.length; i++) {
    if (result[i] === -Infinity) {
      result[i] = 0;
    }
  }

  return result;
}

/**
 * WASM-optimized gather operation
 *
 * Gathers features from source indices.
 */
export function wasmGather(
  src: Float32Array,
  index: Uint32Array,
  numFeatures: number
): Float32Array {
  const numIndices = index.length;
  const result = new Float32Array(numIndices * numFeatures);

  for (let i = 0; i < numIndices; i++) {
    const srcIdx = index[i]!;
    const srcOffset = srcIdx * numFeatures;
    const dstOffset = i * numFeatures;

    // Unroll by 8
    let f = 0;
    for (; f + 7 < numFeatures; f += 8) {
      result[dstOffset + f] = src[srcOffset + f]!;
      result[dstOffset + f + 1] = src[srcOffset + f + 1]!;
      result[dstOffset + f + 2] = src[srcOffset + f + 2]!;
      result[dstOffset + f + 3] = src[srcOffset + f + 3]!;
      result[dstOffset + f + 4] = src[srcOffset + f + 4]!;
      result[dstOffset + f + 5] = src[srcOffset + f + 5]!;
      result[dstOffset + f + 6] = src[srcOffset + f + 6]!;
      result[dstOffset + f + 7] = src[srcOffset + f + 7]!;
    }

    for (; f < numFeatures; f++) {
      result[dstOffset + f] = src[srcOffset + f]!;
    }
  }

  return result;
}

/**
 * WASM-optimized ReLU activation
 */
export function wasmRelu(x: Float32Array): Float32Array {
  const result = new Float32Array(x.length);

  // Unroll by 8
  let i = 0;
  for (; i + 7 < x.length; i += 8) {
    result[i] = x[i]! > 0 ? x[i]! : 0;
    result[i + 1] = x[i + 1]! > 0 ? x[i + 1]! : 0;
    result[i + 2] = x[i + 2]! > 0 ? x[i + 2]! : 0;
    result[i + 3] = x[i + 3]! > 0 ? x[i + 3]! : 0;
    result[i + 4] = x[i + 4]! > 0 ? x[i + 4]! : 0;
    result[i + 5] = x[i + 5]! > 0 ? x[i + 5]! : 0;
    result[i + 6] = x[i + 6]! > 0 ? x[i + 6]! : 0;
    result[i + 7] = x[i + 7]! > 0 ? x[i + 7]! : 0;
  }

  for (; i < x.length; i++) {
    result[i] = x[i]! > 0 ? x[i]! : 0;
  }

  return result;
}

/**
 * WASM-optimized element-wise addition
 */
export function wasmAdd(a: Float32Array, b: Float32Array): Float32Array {
  const result = new Float32Array(a.length);

  // Unroll by 8
  let i = 0;
  for (; i + 7 < a.length; i += 8) {
    result[i] = a[i]! + b[i]!;
    result[i + 1] = a[i + 1]! + b[i + 1]!;
    result[i + 2] = a[i + 2]! + b[i + 2]!;
    result[i + 3] = a[i + 3]! + b[i + 3]!;
    result[i + 4] = a[i + 4]! + b[i + 4]!;
    result[i + 5] = a[i + 5]! + b[i + 5]!;
    result[i + 6] = a[i + 6]! + b[i + 6]!;
    result[i + 7] = a[i + 7]! + b[i + 7]!;
  }

  for (; i < a.length; i++) {
    result[i] = a[i]! + b[i]!;
  }

  return result;
}

/**
 * WASM-optimized Sparse Matrix-Vector Multiplication (SpMV)
 *
 * Uses CSR format for efficient row-wise access.
 */
export function wasmSpmv(
  rowPtr: Uint32Array,
  colIndices: Uint32Array,
  values: Float32Array,
  x: Float32Array,
  numRows: number
): Float32Array {
  const result = new Float32Array(numRows);

  for (let i = 0; i < numRows; i++) {
    const rowStart = rowPtr[i]!;
    const rowEnd = rowPtr[i + 1]!;
    let sum = 0;

    // Unroll by 4
    let j = rowStart;
    for (; j + 3 < rowEnd; j += 4) {
      sum += values[j]! * x[colIndices[j]!]!;
      sum += values[j + 1]! * x[colIndices[j + 1]!]!;
      sum += values[j + 2]! * x[colIndices[j + 2]!]!;
      sum += values[j + 3]! * x[colIndices[j + 3]!]!;
    }

    for (; j < rowEnd; j++) {
      sum += values[j]! * x[colIndices[j]!]!;
    }

    result[i] = sum;
  }

  return result;
}

/**
 * WASM-optimized Sparse Matrix-Dense Matrix Multiplication (SpMM)
 *
 * Uses CSR format for the sparse matrix.
 */
export function wasmSpmm(
  rowPtr: Uint32Array,
  colIndices: Uint32Array,
  values: Float32Array,
  dense: Float32Array,
  numRows: number,
  numCols: number,
  numFeatures: number
): Float32Array {
  const result = new Float32Array(numRows * numFeatures);

  for (let i = 0; i < numRows; i++) {
    const rowStart = rowPtr[i]!;
    const rowEnd = rowPtr[i + 1]!;
    const resultOffset = i * numFeatures;

    for (let j = rowStart; j < rowEnd; j++) {
      const col = colIndices[j]!;
      const val = values[j]!;
      const denseOffset = col * numFeatures;

      // Unroll inner loop by 8
      let f = 0;
      for (; f + 7 < numFeatures; f += 8) {
        result[resultOffset + f] = result[resultOffset + f]! + val * dense[denseOffset + f]!;
        result[resultOffset + f + 1] = result[resultOffset + f + 1]! + val * dense[denseOffset + f + 1]!;
        result[resultOffset + f + 2] = result[resultOffset + f + 2]! + val * dense[denseOffset + f + 2]!;
        result[resultOffset + f + 3] = result[resultOffset + f + 3]! + val * dense[denseOffset + f + 3]!;
        result[resultOffset + f + 4] = result[resultOffset + f + 4]! + val * dense[denseOffset + f + 4]!;
        result[resultOffset + f + 5] = result[resultOffset + f + 5]! + val * dense[denseOffset + f + 5]!;
        result[resultOffset + f + 6] = result[resultOffset + f + 6]! + val * dense[denseOffset + f + 6]!;
        result[resultOffset + f + 7] = result[resultOffset + f + 7]! + val * dense[denseOffset + f + 7]!;
      }

      for (; f < numFeatures; f++) {
        result[resultOffset + f] = result[resultOffset + f]! + val * dense[denseOffset + f]!;
      }
    }
  }

  return result;
}

/**
 * WASM Kernel Manager
 *
 * Provides high-level access to optimized WASM operations.
 */
export class WASMKernels {
  private static instance: WASMKernels | null = null;

  private constructor() {}

  static getInstance(): WASMKernels {
    if (!WASMKernels.instance) {
      WASMKernels.instance = new WASMKernels();
    }
    return WASMKernels.instance;
  }

  matmul(a: Float32Array, b: Float32Array, m: number, k: number, n: number): Float32Array {
    return wasmMatmul(a, b, m, k, n);
  }

  scatterAdd(
    src: Float32Array,
    index: Uint32Array,
    numNodes: number,
    numFeatures: number
  ): Float32Array {
    return wasmScatterAdd(src, index, numNodes, numFeatures);
  }

  scatterMean(
    src: Float32Array,
    index: Uint32Array,
    numNodes: number,
    numFeatures: number
  ): Float32Array {
    return wasmScatterMean(src, index, numNodes, numFeatures);
  }

  scatterMax(
    src: Float32Array,
    index: Uint32Array,
    numNodes: number,
    numFeatures: number
  ): Float32Array {
    return wasmScatterMax(src, index, numNodes, numFeatures);
  }

  gather(src: Float32Array, index: Uint32Array, numFeatures: number): Float32Array {
    return wasmGather(src, index, numFeatures);
  }

  relu(x: Float32Array): Float32Array {
    return wasmRelu(x);
  }

  add(a: Float32Array, b: Float32Array): Float32Array {
    return wasmAdd(a, b);
  }

  spmm(
    rowPtr: Uint32Array,
    colIndices: Uint32Array,
    values: Float32Array,
    dense: Float32Array,
    numRows: number,
    numCols: number,
    numFeatures: number
  ): Float32Array {
    return wasmSpmm(rowPtr, colIndices, values, dense, numRows, numCols, numFeatures);
  }
}
