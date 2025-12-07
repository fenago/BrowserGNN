/**
 * BrowserGNN by Dr. Lee
 * Sparse Tensor Utilities
 *
 * Efficient sparse matrix operations for graph neural networks.
 * Supports COO, CSR, and CSC formats.
 * WASM-accelerated for optimal performance.
 */

import { Tensor } from './tensor';
import {
  wasmScatterAdd,
  wasmScatterMean,
  wasmScatterMax,
  wasmGather,
} from '../backend/wasm';

/**
 * Sparse matrix in COO (Coordinate) format
 * Most flexible format, good for construction
 */
export class SparseCOO {
  readonly rows: Uint32Array;
  readonly cols: Uint32Array;
  readonly values: Float32Array;
  readonly shape: [number, number];
  readonly nnz: number; // Number of non-zeros

  constructor(
    rows: Uint32Array | number[],
    cols: Uint32Array | number[],
    values: Float32Array | number[],
    shape: [number, number]
  ) {
    this.rows = rows instanceof Uint32Array ? rows : new Uint32Array(rows);
    this.cols = cols instanceof Uint32Array ? cols : new Uint32Array(cols);
    this.values = values instanceof Float32Array ? values : new Float32Array(values);
    this.shape = shape;
    this.nnz = this.values.length;

    if (this.rows.length !== this.nnz || this.cols.length !== this.nnz) {
      throw new Error('Row, column, and value arrays must have same length');
    }
  }

  /**
   * Create from dense tensor
   */
  static fromDense(dense: Tensor): SparseCOO {
    if (dense.ndim !== 2) {
      throw new Error('Only 2D tensors supported');
    }

    const [rows, cols] = dense.shape;
    const rowIndices: number[] = [];
    const colIndices: number[] = [];
    const values: number[] = [];

    for (let i = 0; i < rows!; i++) {
      for (let j = 0; j < cols!; j++) {
        const val = dense.get(i, j);
        if (val !== 0) {
          rowIndices.push(i);
          colIndices.push(j);
          values.push(val);
        }
      }
    }

    return new SparseCOO(
      new Uint32Array(rowIndices),
      new Uint32Array(colIndices),
      new Float32Array(values),
      [rows!, cols!]
    );
  }

  /**
   * Create from edge index (adjacency matrix)
   */
  static fromEdgeIndex(
    edgeIndex: Uint32Array,
    numEdges: number,
    numNodes: number,
    values?: Float32Array
  ): SparseCOO {
    const rows = edgeIndex.slice(0, numEdges);
    const cols = edgeIndex.slice(numEdges);
    const vals = values ?? new Float32Array(numEdges).fill(1);

    return new SparseCOO(rows, cols, vals, [numNodes, numNodes]);
  }

  /**
   * Convert to dense tensor
   */
  toDense(): Tensor {
    const result = Tensor.zeros(this.shape);
    for (let i = 0; i < this.nnz; i++) {
      const row = this.rows[i]!;
      const col = this.cols[i]!;
      const val = this.values[i]!;
      const idx = row * this.shape[1] + col;
      result.data[idx] = (result.data[idx] ?? 0) + val;
    }
    return result;
  }

  /**
   * Convert to CSR format
   */
  toCSR(): SparseCSR {
    return SparseCSR.fromCOO(this);
  }

  /**
   * Sparse matrix-vector multiplication
   */
  matvec(x: Tensor): Tensor {
    if (x.ndim !== 1 || x.shape[0] !== this.shape[1]) {
      throw new Error(`Shape mismatch: matrix ${this.shape} @ vector [${x.shape}]`);
    }

    const result = new Float32Array(this.shape[0]);
    for (let i = 0; i < this.nnz; i++) {
      const row = this.rows[i]!;
      const col = this.cols[i]!;
      result[row] = (result[row] ?? 0) + this.values[i]! * x.data[col]!;
    }

    return new Tensor(result, [this.shape[0]]);
  }

  /**
   * Sparse matrix - dense matrix multiplication
   * Result: [M, K] where this is [M, N] and dense is [N, K]
   */
  matmul(dense: Tensor): Tensor {
    if (dense.ndim !== 2 || dense.shape[0] !== this.shape[1]) {
      throw new Error(`Shape mismatch: sparse ${this.shape} @ dense [${dense.shape}]`);
    }

    const [m] = this.shape;
    const k = dense.shape[1]!;
    const result = new Float32Array(m * k);

    for (let i = 0; i < this.nnz; i++) {
      const row = this.rows[i]!;
      const col = this.cols[i]!;
      const val = this.values[i]!;

      for (let j = 0; j < k; j++) {
        const idx = row * k + j;
        result[idx] = (result[idx] ?? 0) + val * dense.data[col * k + j]!;
      }
    }

    return new Tensor(result, [m, k]);
  }

  /**
   * Transpose the sparse matrix
   */
  transpose(): SparseCOO {
    return new SparseCOO(
      new Uint32Array(this.cols),
      new Uint32Array(this.rows),
      new Float32Array(this.values),
      [this.shape[1], this.shape[0]]
    );
  }

  /**
   * Scale all values
   */
  scale(scalar: number): SparseCOO {
    const newValues = new Float32Array(this.nnz);
    for (let i = 0; i < this.nnz; i++) {
      newValues[i] = this.values[i]! * scalar;
    }
    return new SparseCOO(this.rows, this.cols, newValues, this.shape);
  }

  /**
   * Element-wise multiplication with another sparse matrix (same sparsity pattern)
   */
  multiply(other: SparseCOO): SparseCOO {
    if (this.shape[0] !== other.shape[0] || this.shape[1] !== other.shape[1]) {
      throw new Error('Shape mismatch for element-wise multiplication');
    }

    // Build hash map for other
    const otherMap = new Map<string, number>();
    for (let i = 0; i < other.nnz; i++) {
      otherMap.set(`${other.rows[i]},${other.cols[i]}`, other.values[i]!);
    }

    const newRows: number[] = [];
    const newCols: number[] = [];
    const newValues: number[] = [];

    for (let i = 0; i < this.nnz; i++) {
      const key = `${this.rows[i]},${this.cols[i]}`;
      const otherVal = otherMap.get(key);
      if (otherVal !== undefined) {
        newRows.push(this.rows[i]!);
        newCols.push(this.cols[i]!);
        newValues.push(this.values[i]! * otherVal);
      }
    }

    return new SparseCOO(
      new Uint32Array(newRows),
      new Uint32Array(newCols),
      new Float32Array(newValues),
      this.shape
    );
  }

  /**
   * Get sparsity (fraction of zeros)
   */
  get sparsity(): number {
    return 1 - this.nnz / (this.shape[0] * this.shape[1]);
  }

  toString(): string {
    return `SparseCOO(shape=[${this.shape}], nnz=${this.nnz}, sparsity=${(this.sparsity * 100).toFixed(1)}%)`;
  }
}

/**
 * Sparse matrix in CSR (Compressed Sparse Row) format
 * Efficient for row slicing and matrix-vector multiplication
 */
export class SparseCSR {
  readonly rowPtr: Uint32Array; // Row pointers [numRows + 1]
  readonly colIndices: Uint32Array; // Column indices [nnz]
  readonly values: Float32Array; // Values [nnz]
  readonly shape: [number, number];
  readonly nnz: number;

  constructor(
    rowPtr: Uint32Array,
    colIndices: Uint32Array,
    values: Float32Array,
    shape: [number, number]
  ) {
    this.rowPtr = rowPtr;
    this.colIndices = colIndices;
    this.values = values;
    this.shape = shape;
    this.nnz = values.length;
  }

  /**
   * Create from COO format
   */
  static fromCOO(coo: SparseCOO): SparseCSR {
    const [numRows, numCols] = coo.shape;

    // Sort by row then column
    const indices = Array.from({ length: coo.nnz }, (_, i) => i);
    indices.sort((a, b) => {
      if (coo.rows[a]! !== coo.rows[b]!) {
        return coo.rows[a]! - coo.rows[b]!;
      }
      return coo.cols[a]! - coo.cols[b]!;
    });

    const rowPtr = new Uint32Array(numRows + 1);
    const colIndices = new Uint32Array(coo.nnz);
    const values = new Float32Array(coo.nnz);

    let currentRow = 0;
    for (let i = 0; i < coo.nnz; i++) {
      const idx = indices[i]!;
      const row = coo.rows[idx]!;

      while (currentRow <= row) {
        rowPtr[currentRow] = i;
        currentRow++;
      }

      colIndices[i] = coo.cols[idx]!;
      values[i] = coo.values[idx]!;
    }

    // Fill remaining row pointers
    while (currentRow <= numRows) {
      rowPtr[currentRow] = coo.nnz;
      currentRow++;
    }

    return new SparseCSR(rowPtr, colIndices, values, [numRows, numCols]);
  }

  /**
   * Sparse matrix-vector multiplication (efficient in CSR)
   */
  matvec(x: Tensor): Tensor {
    if (x.ndim !== 1 || x.shape[0] !== this.shape[1]) {
      throw new Error(`Shape mismatch: matrix ${this.shape} @ vector [${x.shape}]`);
    }

    const result = new Float32Array(this.shape[0]);

    for (let row = 0; row < this.shape[0]; row++) {
      let sum = 0;
      for (let j = this.rowPtr[row]!; j < this.rowPtr[row + 1]!; j++) {
        sum += this.values[j]! * x.data[this.colIndices[j]!]!;
      }
      result[row] = sum;
    }

    return new Tensor(result, [this.shape[0]]);
  }

  /**
   * Sparse matrix - dense matrix multiplication
   */
  matmul(dense: Tensor): Tensor {
    if (dense.ndim !== 2 || dense.shape[0] !== this.shape[1]) {
      throw new Error(`Shape mismatch: sparse ${this.shape} @ dense [${dense.shape}]`);
    }

    const [m] = this.shape;
    const k = dense.shape[1]!;
    const result = new Float32Array(m * k);

    for (let row = 0; row < m; row++) {
      for (let j = this.rowPtr[row]!; j < this.rowPtr[row + 1]!; j++) {
        const col = this.colIndices[j]!;
        const val = this.values[j]!;

        for (let c = 0; c < k; c++) {
          const idx = row * k + c;
          result[idx] = (result[idx] ?? 0) + val * dense.data[col * k + c]!;
        }
      }
    }

    return new Tensor(result, [m, k]);
  }

  /**
   * Get row as dense array
   */
  getRow(row: number): Float32Array {
    const result = new Float32Array(this.shape[1]);
    for (let j = this.rowPtr[row]!; j < this.rowPtr[row + 1]!; j++) {
      result[this.colIndices[j]!] = this.values[j]!;
    }
    return result;
  }

  /**
   * Convert to COO format
   */
  toCOO(): SparseCOO {
    const rows = new Uint32Array(this.nnz);

    for (let row = 0; row < this.shape[0]; row++) {
      for (let j = this.rowPtr[row]!; j < this.rowPtr[row + 1]!; j++) {
        rows[j] = row;
      }
    }

    return new SparseCOO(rows, this.colIndices, this.values, this.shape);
  }

  /**
   * Convert to dense tensor
   */
  toDense(): Tensor {
    return this.toCOO().toDense();
  }

  toString(): string {
    const sparsity = 1 - this.nnz / (this.shape[0] * this.shape[1]);
    return `SparseCSR(shape=[${this.shape}], nnz=${this.nnz}, sparsity=${(sparsity * 100).toFixed(1)}%)`;
  }
}

/**
 * Message passing utilities for GNNs
 * WASM-accelerated with loop unrolling for optimal performance
 */
export class MessagePassing {
  /**
   * Scatter-add operation: aggregate values by target indices
   * Used for message aggregation in GNNs
   * Uses WASM kernel with 8x loop unrolling
   *
   * @param src Source values [numMessages, features]
   * @param index Target indices [numMessages]
   * @param numNodes Number of output nodes
   * @returns Aggregated values [numNodes, features]
   */
  static scatterAdd(src: Tensor, index: Uint32Array, numNodes: number): Tensor {
    if (src.ndim !== 2) {
      throw new Error('Source must be 2D tensor');
    }
    if (src.shape[0] !== index.length) {
      throw new Error('Source rows must match index length');
    }

    const numFeatures = src.shape[1]!;

    // Use WASM-optimized kernel
    const result = wasmScatterAdd(src.data, index, numNodes, numFeatures);

    return new Tensor(result, [numNodes, numFeatures]);
  }

  /**
   * Scatter-mean operation: average values by target indices
   * Uses WASM kernel with 8x loop unrolling
   */
  static scatterMean(src: Tensor, index: Uint32Array, numNodes: number): Tensor {
    if (src.ndim !== 2) {
      throw new Error('Source must be 2D tensor');
    }
    if (src.shape[0] !== index.length) {
      throw new Error('Source rows must match index length');
    }

    const numFeatures = src.shape[1]!;

    // Use WASM-optimized kernel
    const result = wasmScatterMean(src.data, index, numNodes, numFeatures);

    return new Tensor(result, [numNodes, numFeatures]);
  }

  /**
   * Scatter-max operation: max values by target indices
   * Uses WASM kernel with 8x loop unrolling
   */
  static scatterMax(src: Tensor, index: Uint32Array, numNodes: number): Tensor {
    if (src.ndim !== 2) {
      throw new Error('Source must be 2D tensor');
    }

    const numFeatures = src.shape[1]!;

    // Use WASM-optimized kernel
    const result = wasmScatterMax(src.data, index, numNodes, numFeatures);

    return new Tensor(result, [numNodes, numFeatures]);
  }

  /**
   * Gather operation: select values by indices
   * Used to gather source node features for each edge
   * Uses WASM kernel with 8x loop unrolling
   *
   * @param src Source values [numNodes, features]
   * @param index Indices to gather [numIndices]
   * @returns Gathered values [numIndices, features]
   */
  static gather(src: Tensor, index: Uint32Array): Tensor {
    if (src.ndim !== 2) {
      throw new Error('Source must be 2D tensor');
    }

    const numFeatures = src.shape[1]!;

    // Use WASM-optimized kernel
    const result = wasmGather(src.data, index, numFeatures);

    return new Tensor(result, [index.length, numFeatures]);
  }
}

/**
 * Compute normalized edge weights for GCN
 * Returns D^{-1/2} weights for each edge
 */
export function computeGCNNorm(
  edgeIndex: Uint32Array,
  numEdges: number,
  numNodes: number,
  addSelfLoops: boolean = true
): Float32Array {
  // Calculate degree (including self-loops if specified)
  const degree = new Float32Array(numNodes);

  if (addSelfLoops) {
    degree.fill(1); // Self-loop contribution
  }

  for (let i = 0; i < numEdges; i++) {
    const target = edgeIndex[numEdges + i]!;
    degree[target] = (degree[target] ?? 0) + 1;
  }

  // Calculate D^{-1/2}
  const degInvSqrt = new Float32Array(numNodes);
  for (let i = 0; i < numNodes; i++) {
    degInvSqrt[i] = degree[i]! > 0 ? 1 / Math.sqrt(degree[i]!) : 0;
  }

  // Calculate edge weights: D^{-1/2}[src] * D^{-1/2}[tgt]
  const edgeWeight = new Float32Array(numEdges);
  for (let i = 0; i < numEdges; i++) {
    const src = edgeIndex[i]!;
    const tgt = edgeIndex[numEdges + i]!;
    edgeWeight[i] = degInvSqrt[src]! * degInvSqrt[tgt]!;
  }

  return edgeWeight;
}
