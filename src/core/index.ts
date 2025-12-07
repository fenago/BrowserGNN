/**
 * BrowserGNN by Dr. Lee
 * Core Data Structures
 *
 * Export all core data structures and utilities.
 */

export { Tensor, concat, stack, type Shape, type DataType, type TypedArray } from './tensor';

export {
  GraphData,
  batchGraphs,
  fromEdgeList,
  randomGraph,
  type GraphDataConfig,
} from './graph';

export {
  SparseCOO,
  SparseCSR,
  MessagePassing,
  computeGCNNorm,
} from './sparse';
