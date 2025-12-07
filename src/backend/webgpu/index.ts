/**
 * BrowserGNN by Dr. Lee
 * WebGPU Backend Module
 *
 * High-performance compute shaders for graph neural network operations.
 */

export { GPUBufferPool, uploadToBuffer, readFromBuffer } from './buffer-pool';
export type { BufferDescriptor } from './buffer-pool';

export { WebGPUComputeManager } from './compute-pipeline';
export type { ComputePipelineConfig } from './compute-pipeline';

export {
  SHADER_REGISTRY,
  SPMM_SHADER,
  SPMM_ROW_PARALLEL_SHADER,
  SCATTER_ADD_SHADER,
  SCATTER_MEAN_SHADER,
  SCATTER_MAX_SHADER,
  GATHER_SHADER,
  ATTENTION_SCORE_SHADER,
  ATTENTION_SOFTMAX_SHADER,
  MATMUL_SHADER,
  RELU_SHADER,
  ADD_SHADER,
} from './shaders';
export type { ShaderName } from './shaders';
