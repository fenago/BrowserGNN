/**
 * BrowserGNN by Dr. Lee
 * GPU-Accelerated Sparse Operations
 *
 * Provides WebGPU-accelerated versions of sparse matrix operations
 * with automatic fallback to CPU when WebGPU is not available.
 */

import { Tensor } from './tensor';
import { SparseCOO, SparseCSR, MessagePassing } from './sparse';
import { backend, getComputeManager } from '../backend';
import type { WebGPUComputeManager } from '../backend/webgpu';

/**
 * GPU-accelerated sparse operations
 *
 * Automatically uses WebGPU when available, falls back to CPU otherwise.
 */
export class GPUSparse {
  /**
   * Sparse matrix-dense matrix multiplication
   *
   * @param sparse Sparse matrix in COO or CSR format
   * @param dense Dense matrix [N, K]
   * @returns Result matrix [M, K]
   */
  static async matmul(sparse: SparseCOO | SparseCSR, dense: Tensor): Promise<Tensor> {
    const computeManager = getComputeManager();

    // Use GPU if available
    if (computeManager && backend.getBackend() === 'webgpu') {
      try {
        return await this.gpuSpMM(sparse, dense, computeManager);
      } catch (error) {
        console.warn('WebGPU SpMM failed, falling back to CPU:', error);
      }
    }

    // CPU fallback
    return sparse.matmul(dense);
  }

  /**
   * GPU-accelerated SpMM
   */
  private static async gpuSpMM(
    sparse: SparseCOO | SparseCSR,
    dense: Tensor,
    computeManager: WebGPUComputeManager
  ): Promise<Tensor> {
    // Convert to CSR for efficient row-parallel processing
    const csr = sparse instanceof SparseCSR ? sparse : sparse.toCSR();

    const [numRows, numCols] = csr.shape;
    const numFeatures = dense.shape[1]!;

    const result = await computeManager.spmm(
      csr.rowPtr,
      csr.colIndices,
      csr.values,
      dense.data,
      numRows,
      numCols,
      numFeatures
    );

    return new Tensor(result, [numRows, numFeatures]);
  }

  /**
   * Scatter-add operation: aggregate values by target indices
   *
   * @param src Source values [numMessages, features]
   * @param index Target indices [numMessages]
   * @param numNodes Number of output nodes
   * @returns Aggregated values [numNodes, features]
   */
  static async scatterAdd(
    src: Tensor,
    index: Uint32Array,
    numNodes: number
  ): Promise<Tensor> {
    const computeManager = getComputeManager();

    // Use GPU if available
    if (computeManager && backend.getBackend() === 'webgpu') {
      try {
        const numFeatures = src.shape[1]!;
        const result = await computeManager.scatterAdd(
          src.data,
          index,
          numNodes,
          numFeatures
        );
        return new Tensor(result, [numNodes, numFeatures]);
      } catch (error) {
        console.warn('WebGPU scatter-add failed, falling back to CPU:', error);
      }
    }

    // CPU fallback
    return MessagePassing.scatterAdd(src, index, numNodes);
  }

  /**
   * Scatter-mean operation: average values by target indices
   *
   * @param src Source values [numMessages, features]
   * @param index Target indices [numMessages]
   * @param numNodes Number of output nodes
   * @returns Averaged values [numNodes, features]
   */
  static async scatterMean(
    src: Tensor,
    index: Uint32Array,
    numNodes: number
  ): Promise<Tensor> {
    // For now, use CPU implementation
    // GPU scatter-mean requires atomic operations which need special handling
    // TODO: Implement GPU scatter-mean with two-pass approach
    return MessagePassing.scatterMean(src, index, numNodes);
  }

  /**
   * Scatter-max operation: max values by target indices
   *
   * @param src Source values [numMessages, features]
   * @param index Target indices [numMessages]
   * @param numNodes Number of output nodes
   * @returns Max values [numNodes, features]
   */
  static async scatterMax(
    src: Tensor,
    index: Uint32Array,
    numNodes: number
  ): Promise<Tensor> {
    const computeManager = getComputeManager();

    // Use GPU if available
    if (computeManager && backend.getBackend() === 'webgpu') {
      try {
        const numFeatures = src.shape[1]!;
        const result = await computeManager.scatterMax(
          src.data,
          index,
          numNodes,
          numFeatures
        );
        return new Tensor(result, [numNodes, numFeatures]);
      } catch (error) {
        console.warn('WebGPU scatter-max failed, falling back to CPU:', error);
      }
    }

    // CPU fallback
    return MessagePassing.scatterMax(src, index, numNodes);
  }

  /**
   * Gather operation: collect values by indices
   *
   * @param src Source values [numNodes, features]
   * @param index Indices to gather [numIndices]
   * @returns Gathered values [numIndices, features]
   */
  static async gather(src: Tensor, index: Uint32Array): Promise<Tensor> {
    const computeManager = getComputeManager();

    // Use GPU if available
    if (computeManager && backend.getBackend() === 'webgpu') {
      try {
        const numFeatures = src.shape[1]!;
        const result = await computeManager.gather(
          src.data,
          index,
          numFeatures
        );
        return new Tensor(result, [index.length, numFeatures]);
      } catch (error) {
        console.warn('WebGPU gather failed, falling back to CPU:', error);
      }
    }

    // CPU fallback
    return MessagePassing.gather(src, index);
  }
}

/**
 * GPU-accelerated tensor operations
 */
export class GPUTensor {
  /**
   * Matrix multiplication: C = A @ B
   *
   * @param a Matrix A [M, K]
   * @param b Matrix B [K, N]
   * @returns Result C [M, N]
   */
  static async matmul(a: Tensor, b: Tensor): Promise<Tensor> {
    if (a.ndim !== 2 || b.ndim !== 2) {
      throw new Error('Both tensors must be 2D');
    }
    if (a.shape[1] !== b.shape[0]) {
      throw new Error(`Shape mismatch: [${a.shape}] @ [${b.shape}]`);
    }

    const computeManager = getComputeManager();

    // Use GPU if available
    if (computeManager && backend.getBackend() === 'webgpu') {
      try {
        const m = a.shape[0]!;
        const k = a.shape[1]!;
        const n = b.shape[1]!;

        const result = await computeManager.matmul(a.data, b.data, m, k, n);
        return new Tensor(result, [m, n]);
      } catch (error) {
        console.warn('WebGPU matmul failed, falling back to CPU:', error);
      }
    }

    // CPU fallback
    return a.matmul(b);
  }

  /**
   * ReLU activation
   *
   * @param x Input tensor
   * @returns Output tensor with ReLU applied
   */
  static async relu(x: Tensor): Promise<Tensor> {
    const computeManager = getComputeManager();

    // Use GPU if available
    if (computeManager && backend.getBackend() === 'webgpu') {
      try {
        const result = await computeManager.relu(new Float32Array(x.data));
        return new Tensor(result, x.shape);
      } catch (error) {
        console.warn('WebGPU relu failed, falling back to CPU:', error);
      }
    }

    // CPU fallback
    return x.relu();
  }

  /**
   * Element-wise addition
   *
   * @param a First tensor
   * @param b Second tensor
   * @returns Sum tensor
   */
  static async add(a: Tensor, b: Tensor): Promise<Tensor> {
    if (a.size !== b.size) {
      throw new Error(`Size mismatch: ${a.size} vs ${b.size}`);
    }

    const computeManager = getComputeManager();

    // Use GPU if available
    if (computeManager && backend.getBackend() === 'webgpu') {
      try {
        const result = await computeManager.add(a.data, b.data);
        return new Tensor(result, a.shape);
      } catch (error) {
        console.warn('WebGPU add failed, falling back to CPU:', error);
      }
    }

    // CPU fallback
    return a.add(b);
  }
}

/**
 * Check if GPU acceleration is available
 */
export function isGPUAvailable(): boolean {
  return backend.getBackend() === 'webgpu' && getComputeManager() !== null;
}

/**
 * Get GPU statistics
 */
export function getGPUStats(): {
  available: boolean;
  backend: string;
  bufferPoolStats?: ReturnType<WebGPUComputeManager['getStats']>;
} {
  const computeManager = getComputeManager();

  return {
    available: isGPUAvailable(),
    backend: backend.getBackend(),
    bufferPoolStats: computeManager?.getStats(),
  };
}
