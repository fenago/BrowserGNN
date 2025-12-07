/**
 * BrowserGNN by Dr. Lee
 * Core Tensor Operations
 *
 * Provides fundamental tensor operations for graph neural networks
 * with support for both CPU (TypedArrays) and GPU (WebGPU) backends.
 * WASM-accelerated for optimal performance.
 */

import { wasmMatmul, wasmRelu, wasmAdd } from '../backend/wasm';

export type TypedArray = Float32Array | Float64Array | Int32Array | Uint32Array;
export type DataType = 'float32' | 'float64' | 'int32' | 'uint32';

/**
 * Shape descriptor for multi-dimensional tensors
 */
export type Shape = number[];

/**
 * Tensor class for numerical computations
 * Wraps TypedArrays with shape information and operations
 */
export class Tensor {
  readonly data: Float32Array;
  readonly shape: Shape;
  readonly dtype: DataType;

  constructor(data: Float32Array | number[], shape: Shape, dtype: DataType = 'float32') {
    this.data = data instanceof Float32Array ? data : new Float32Array(data);
    this.shape = shape;
    this.dtype = dtype;

    // Validate shape matches data length
    const expectedLength = shape.reduce((a, b) => a * b, 1);
    if (this.data.length !== expectedLength) {
      throw new Error(
        `Shape ${shape} expects ${expectedLength} elements, but got ${this.data.length}`
      );
    }
  }

  /**
   * Get the number of dimensions
   */
  get ndim(): number {
    return this.shape.length;
  }

  /**
   * Get total number of elements
   */
  get size(): number {
    return this.data.length;
  }

  /**
   * Create a tensor of zeros
   */
  static zeros(shape: Shape): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Float32Array(size), shape);
  }

  /**
   * Create a tensor of ones
   */
  static ones(shape: Shape): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size).fill(1);
    return new Tensor(data, shape);
  }

  /**
   * Create a tensor with random values from uniform distribution [0, 1)
   */
  static rand(shape: Shape): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      data[i] = Math.random();
    }
    return new Tensor(data, shape);
  }

  /**
   * Create a tensor with random values from normal distribution
   * Using Box-Muller transform
   */
  static randn(shape: Shape): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Float32Array(size);
    for (let i = 0; i < size; i += 2) {
      const u1 = Math.random();
      const u2 = Math.random();
      const r = Math.sqrt(-2 * Math.log(u1));
      const theta = 2 * Math.PI * u2;
      data[i] = r * Math.cos(theta);
      if (i + 1 < size) {
        data[i + 1] = r * Math.sin(theta);
      }
    }
    return new Tensor(data, shape);
  }

  /**
   * Create identity matrix
   */
  static eye(n: number): Tensor {
    const data = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
      data[i * n + i] = 1;
    }
    return new Tensor(data, [n, n]);
  }

  /**
   * Reshape tensor to new shape
   */
  reshape(newShape: Shape): Tensor {
    const newSize = newShape.reduce((a, b) => a * b, 1);
    if (newSize !== this.size) {
      throw new Error(`Cannot reshape tensor of size ${this.size} to shape ${newShape}`);
    }
    return new Tensor(this.data, newShape, this.dtype);
  }

  /**
   * Transpose 2D tensor
   */
  transpose(): Tensor {
    if (this.ndim !== 2) {
      throw new Error('Transpose only supported for 2D tensors');
    }
    const rows = this.shape[0]!;
    const cols = this.shape[1]!;
    const result = new Float32Array(this.size);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        result[j * rows + i] = this.data[i * cols + j]!;
      }
    }
    return new Tensor(result, [cols, rows], this.dtype);
  }

  /**
   * Get value at index (for 1D) or [row, col] (for 2D)
   */
  get(...indices: number[]): number {
    if (indices.length !== this.ndim) {
      throw new Error(`Expected ${this.ndim} indices, got ${indices.length}`);
    }
    let idx = 0;
    let stride = 1;
    for (let i = this.ndim - 1; i >= 0; i--) {
      idx += indices[i]! * stride;
      stride *= this.shape[i]!;
    }
    return this.data[idx]!;
  }

  /**
   * Set value at index
   */
  set(value: number, ...indices: number[]): void {
    if (indices.length !== this.ndim) {
      throw new Error(`Expected ${this.ndim} indices, got ${indices.length}`);
    }
    let idx = 0;
    let stride = 1;
    for (let i = this.ndim - 1; i >= 0; i--) {
      idx += indices[i]! * stride;
      stride *= this.shape[i]!;
    }
    this.data[idx] = value;
  }

  /**
   * Clone the tensor
   */
  clone(): Tensor {
    return new Tensor(new Float32Array(this.data), [...this.shape], this.dtype);
  }

  /**
   * Element-wise addition
   * Uses WASM kernel with 8x loop unrolling for tensor-tensor addition
   */
  add(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      const result = new Float32Array(this.size);
      for (let i = 0; i < this.size; i++) {
        result[i] = this.data[i]! + other;
      }
      return new Tensor(result, this.shape, this.dtype);
    }

    if (!this.shapeEquals(other)) {
      throw new Error(`Shape mismatch: ${this.shape} vs ${other.shape}`);
    }
    // Use WASM-optimized kernel for tensor-tensor addition
    const result = wasmAdd(this.data, other.data);
    return new Tensor(result, this.shape, this.dtype);
  }

  /**
   * Element-wise subtraction
   */
  sub(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      const result = new Float32Array(this.size);
      for (let i = 0; i < this.size; i++) {
        result[i] = this.data[i]! - other;
      }
      return new Tensor(result, this.shape, this.dtype);
    }

    if (!this.shapeEquals(other)) {
      throw new Error(`Shape mismatch: ${this.shape} vs ${other.shape}`);
    }
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) {
      result[i] = this.data[i]! - other.data[i]!;
    }
    return new Tensor(result, this.shape, this.dtype);
  }

  /**
   * Element-wise multiplication
   */
  mul(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      const result = new Float32Array(this.size);
      for (let i = 0; i < this.size; i++) {
        result[i] = this.data[i]! * other;
      }
      return new Tensor(result, this.shape, this.dtype);
    }

    if (!this.shapeEquals(other)) {
      throw new Error(`Shape mismatch: ${this.shape} vs ${other.shape}`);
    }
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) {
      result[i] = this.data[i]! * other.data[i]!;
    }
    return new Tensor(result, this.shape, this.dtype);
  }

  /**
   * Element-wise division
   */
  div(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      const result = new Float32Array(this.size);
      for (let i = 0; i < this.size; i++) {
        result[i] = this.data[i]! / other;
      }
      return new Tensor(result, this.shape, this.dtype);
    }

    if (!this.shapeEquals(other)) {
      throw new Error(`Shape mismatch: ${this.shape} vs ${other.shape}`);
    }
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) {
      result[i] = this.data[i]! / other.data[i]!;
    }
    return new Tensor(result, this.shape, this.dtype);
  }

  /**
   * Matrix multiplication for 2D tensors
   * Uses WASM kernel with 4x loop unrolling for optimal performance
   */
  matmul(other: Tensor): Tensor {
    if (this.ndim !== 2 || other.ndim !== 2) {
      throw new Error('Matrix multiplication requires 2D tensors');
    }
    const [m, k1] = this.shape;
    const [k2, n] = other.shape;
    if (k1 !== k2) {
      throw new Error(`Shape mismatch for matmul: ${this.shape} @ ${other.shape}`);
    }

    // Use WASM-optimized kernel
    const result = wasmMatmul(this.data, other.data, m!, k1!, n!);
    return new Tensor(result, [m!, n!], this.dtype);
  }

  /**
   * Sum all elements
   */
  sum(): number {
    return this.data.reduce((a, b) => a + b, 0);
  }

  /**
   * Sum along axis
   */
  sumAxis(axis: number): Tensor {
    if (axis < 0 || axis >= this.ndim) {
      throw new Error(`Invalid axis ${axis} for tensor with ${this.ndim} dimensions`);
    }

    const newShape = this.shape.filter((_, i) => i !== axis);
    if (newShape.length === 0) {
      return new Tensor(new Float32Array([this.sum()]), [1]);
    }

    const newSize = newShape.reduce((a, b) => a * b, 1);
    const result = new Float32Array(newSize);

    // Calculate strides
    const strides: number[] = [];
    let stride = 1;
    for (let i = this.ndim - 1; i >= 0; i--) {
      strides.unshift(stride);
      stride *= this.shape[i]!;
    }

    // Sum along axis
    const axisSize = this.shape[axis]!;
    const axisStride = strides[axis]!;

    for (let i = 0; i < this.size; i++) {
      // Calculate output index (excluding the axis dimension)
      let outIdx = 0;
      let temp = i;
      let outStride = 1;
      for (let d = this.ndim - 1; d >= 0; d--) {
        if (d !== axis) {
          const coord = temp % this.shape[d]!;
          outIdx += coord * outStride;
          outStride *= this.shape[d]!;
        }
        temp = Math.floor(temp / this.shape[d]!);
      }
      // Adjust output index calculation
      outIdx = 0;
      outStride = 1;
      for (let d = this.ndim - 1; d >= 0; d--) {
        if (d !== axis) {
          const dimSize = this.shape[d]!;
          const coord = Math.floor(i / strides[d]!) % dimSize;
          outIdx = coord * outStride + outIdx;
          outStride *= dimSize;
        }
      }
      result[outIdx] = (result[outIdx] ?? 0) + this.data[i]!;
    }

    return new Tensor(result, newShape);
  }

  /**
   * Mean of all elements
   */
  mean(): number {
    return this.sum() / this.size;
  }

  /**
   * Maximum value
   */
  max(): number {
    return Math.max(...this.data);
  }

  /**
   * Minimum value
   */
  min(): number {
    return Math.min(...this.data);
  }

  /**
   * Apply ReLU activation
   * Uses WASM kernel with 8x loop unrolling for optimal performance
   */
  relu(): Tensor {
    // Use WASM-optimized kernel
    const result = wasmRelu(this.data);
    return new Tensor(result, this.shape, this.dtype);
  }

  /**
   * Apply Leaky ReLU activation
   */
  leakyRelu(negativeSlope: number = 0.01): Tensor {
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) {
      const x = this.data[i]!;
      result[i] = x > 0 ? x : negativeSlope * x;
    }
    return new Tensor(result, this.shape, this.dtype);
  }

  /**
   * Apply sigmoid activation
   */
  sigmoid(): Tensor {
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) {
      result[i] = 1 / (1 + Math.exp(-this.data[i]!));
    }
    return new Tensor(result, this.shape, this.dtype);
  }

  /**
   * Apply tanh activation
   */
  tanh(): Tensor {
    const result = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) {
      result[i] = Math.tanh(this.data[i]!);
    }
    return new Tensor(result, this.shape, this.dtype);
  }

  /**
   * Apply softmax along last axis
   */
  softmax(): Tensor {
    if (this.ndim !== 2) {
      throw new Error('Softmax currently only supports 2D tensors');
    }
    const [rows, cols] = this.shape;
    const result = new Float32Array(this.size);

    for (let i = 0; i < rows!; i++) {
      // Find max for numerical stability
      let maxVal = -Infinity;
      for (let j = 0; j < cols!; j++) {
        maxVal = Math.max(maxVal, this.data[i * cols! + j]!);
      }

      // Compute exp and sum
      let sumExp = 0;
      for (let j = 0; j < cols!; j++) {
        const exp = Math.exp(this.data[i * cols! + j]! - maxVal);
        result[i * cols! + j] = exp;
        sumExp += exp;
      }

      // Normalize
      for (let j = 0; j < cols!; j++) {
        result[i * cols! + j] = result[i * cols! + j]! / sumExp;
      }
    }

    return new Tensor(result, this.shape, this.dtype);
  }

  /**
   * Apply log softmax along last axis
   */
  logSoftmax(): Tensor {
    if (this.ndim !== 2) {
      throw new Error('Log softmax currently only supports 2D tensors');
    }
    const [rows, cols] = this.shape;
    const result = new Float32Array(this.size);

    for (let i = 0; i < rows!; i++) {
      // Find max for numerical stability
      let maxVal = -Infinity;
      for (let j = 0; j < cols!; j++) {
        maxVal = Math.max(maxVal, this.data[i * cols! + j]!);
      }

      // Compute log-sum-exp
      let sumExp = 0;
      for (let j = 0; j < cols!; j++) {
        sumExp += Math.exp(this.data[i * cols! + j]! - maxVal);
      }
      const logSumExp = maxVal + Math.log(sumExp);

      // Compute log softmax
      for (let j = 0; j < cols!; j++) {
        result[i * cols! + j] = this.data[i * cols! + j]! - logSumExp;
      }
    }

    return new Tensor(result, this.shape, this.dtype);
  }

  /**
   * Check if shapes are equal
   */
  shapeEquals(other: Tensor): boolean {
    if (this.ndim !== other.ndim) return false;
    return this.shape.every((dim, i) => dim === other.shape[i]);
  }

  /**
   * Convert to string representation
   */
  toString(): string {
    return `Tensor(shape=[${this.shape.join(', ')}], dtype=${this.dtype})`;
  }

  /**
   * Pretty print tensor data
   */
  print(): void {
    console.log(this.toString());
    if (this.ndim === 1) {
      console.log(`[${Array.from(this.data).map(x => x.toFixed(4)).join(', ')}]`);
    } else if (this.ndim === 2) {
      const [rows, cols] = this.shape;
      for (let i = 0; i < rows!; i++) {
        const row = Array.from(this.data.slice(i * cols!, (i + 1) * cols!))
          .map(x => x.toFixed(4).padStart(10))
          .join(' ');
        console.log(`[${row}]`);
      }
    }
  }
}

/**
 * Concatenate tensors along axis
 */
export function concat(tensors: Tensor[], axis: number = 0): Tensor {
  if (tensors.length === 0) {
    throw new Error('Cannot concatenate empty array');
  }

  const first = tensors[0]!;
  const ndim = first.ndim;

  // Validate shapes
  for (const t of tensors) {
    if (t.ndim !== ndim) {
      throw new Error('All tensors must have same number of dimensions');
    }
    for (let i = 0; i < ndim; i++) {
      if (i !== axis && t.shape[i] !== first.shape[i]) {
        throw new Error('All tensors must have same shape except along concat axis');
      }
    }
  }

  // Calculate new shape
  const newShape = [...first.shape];
  newShape[axis] = tensors.reduce((sum, t) => sum + t.shape[axis]!, 0);

  // For 2D tensors
  if (ndim === 2) {
    const [rows, cols] = first.shape;
    if (axis === 0) {
      // Concatenate rows
      const newRows = tensors.reduce((sum, t) => sum + t.shape[0]!, 0);
      const result = new Float32Array(newRows * cols!);
      let offset = 0;
      for (const t of tensors) {
        result.set(t.data, offset);
        offset += t.data.length;
      }
      return new Tensor(result, [newRows, cols!]);
    } else {
      // Concatenate columns
      const newCols = tensors.reduce((sum, t) => sum + t.shape[1]!, 0);
      const result = new Float32Array(rows! * newCols);
      for (let i = 0; i < rows!; i++) {
        let colOffset = 0;
        for (const t of tensors) {
          const tCols = t.shape[1]!;
          for (let j = 0; j < tCols; j++) {
            result[i * newCols + colOffset + j] = t.data[i * tCols + j]!;
          }
          colOffset += tCols;
        }
      }
      return new Tensor(result, [rows!, newCols]);
    }
  }

  throw new Error('Concatenation only supported for 2D tensors currently');
}

/**
 * Stack tensors along new axis
 */
export function stack(tensors: Tensor[], axis: number = 0): Tensor {
  if (tensors.length === 0) {
    throw new Error('Cannot stack empty array');
  }

  const first = tensors[0]!;

  // Validate all tensors have same shape
  for (const t of tensors) {
    if (!t.shapeEquals(first)) {
      throw new Error('All tensors must have same shape for stacking');
    }
  }

  // Create new shape with extra dimension
  const newShape = [...first.shape];
  newShape.splice(axis, 0, tensors.length);

  const result = new Float32Array(tensors.length * first.size);
  for (let i = 0; i < tensors.length; i++) {
    result.set(tensors[i]!.data, i * first.size);
  }

  return new Tensor(result, newShape);
}
