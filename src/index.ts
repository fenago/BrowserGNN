/**
 * BrowserGNN by Dr. Lee
 *
 * The first comprehensive Graph Neural Network library for the browser.
 * Enables client-side graph learning with WebGPU acceleration and WASM fallback.
 *
 * @packageDocumentation
 */

// Version
export const VERSION = '0.4.0';
export const AUTHOR = 'Dr. Lee';

// Core data structures
export {
  // Tensor operations
  Tensor,
  concat,
  stack,
  type Shape,
  type DataType,
  type TypedArray,
  // Graph data
  GraphData,
  batchGraphs,
  fromEdgeList,
  randomGraph,
  type GraphDataConfig,
  // Sparse operations
  SparseCOO,
  SparseCSR,
  MessagePassing,
  computeGCNNorm,
  // GPU-accelerated operations
  GPUSparse,
  GPUTensor,
  isGPUAvailable,
  getGPUStats,
} from './core';

// Neural network primitives
export {
  Module,
  Sequential,
  type Parameter,
  Linear,
  type LinearConfig,
  // Activation functions
  ReLU,
  LeakyReLU,
  ELU,
  Sigmoid,
  Tanh,
  Softmax,
  LogSoftmax,
  GELU,
  SiLU,
  PReLU,
  Dropout,
} from './nn';

// GNN layers
export {
  GCNConv,
  type GCNConvConfig,
  GATConv,
  type GATConvConfig,
  SAGEConv,
  type SAGEConvConfig,
  type SAGEAggregator,
} from './layers';

// Backend management
export {
  initBrowserGNN,
  getBackend,
  checkCapabilities,
  getBackendInfo,
  type BackendType,
  type BackendCapabilities,
  type BackendConfig,
} from './backend';

// Benchmarks
export {
  benchmarkGCN,
  benchmarkSAGE,
  benchmarkGAT,
  runAllBenchmarks,
  quickBenchmark,
  formatBenchmarkTable,
  type BenchmarkResult,
  type BenchmarkConfig,
} from './benchmarks';

// Autograd and Training
export {
  // Variable with gradient tracking
  Variable,
  type BackwardFn,
  type GradNode,
  // Loss functions
  crossEntropyLoss,
  mseLoss,
  binaryCrossEntropyLoss,
  nllLoss,
  l1Loss,
  smoothL1Loss,
  // Optimizers
  Optimizer,
  SGD,
  Adam,
  Adagrad,
  RMSprop,
  // Learning rate schedulers
  LRScheduler,
  StepLR,
  ExponentialLR,
  CosineAnnealingLR,
  type ParamGroup,
  // Trainer
  Trainer,
  type TrainerConfig,
  type TrainingMetrics,
} from './autograd';

// Convenience function to create and initialize BrowserGNN
export async function createBrowserGNN(config?: {
  preferredBackend?: 'webgpu' | 'wasm' | 'cpu';
  enableProfiling?: boolean;
}): Promise<{
  backend: string;
  info: string;
}> {
  const { initBrowserGNN, getBackendInfo } = await import('./backend');
  const backend = await initBrowserGNN(config);
  return {
    backend,
    info: getBackendInfo(),
  };
}

// Default export for convenient imports
export default {
  VERSION,
  AUTHOR,
  createBrowserGNN,
};
