/**
 * BrowserGNN by Dr. Lee
 *
 * The first comprehensive Graph Neural Network library for the browser.
 * Enables client-side graph learning with WebGPU acceleration and WASM fallback.
 *
 * @packageDocumentation
 */

// Version
export const VERSION = '0.1.0';
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
