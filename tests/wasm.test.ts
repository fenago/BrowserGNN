/**
 * BrowserGNN by Dr. Lee
 * WASM Kernel Tests
 */

import { describe, it, expect } from 'vitest';
import {
  wasmMatmul,
  wasmScatterAdd,
  wasmScatterMean,
  wasmScatterMax,
  wasmGather,
  wasmRelu,
  wasmAdd,
  wasmSpmv,
  wasmSpmm,
  WASMKernels,
} from '../src/backend/wasm';

describe('WASM Kernels', () => {
  describe('wasmMatmul', () => {
    it('should perform matrix multiplication correctly', () => {
      // 2x3 * 3x2 = 2x2
      const a = new Float32Array([1, 2, 3, 4, 5, 6]); // 2x3
      const b = new Float32Array([1, 2, 3, 4, 5, 6]); // 3x2

      const result = wasmMatmul(a, b, 2, 3, 2);

      // Expected:
      // [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
      // [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
      expect(result.length).toBe(4);
      expect(result[0]).toBeCloseTo(22, 5);
      expect(result[1]).toBeCloseTo(28, 5);
      expect(result[2]).toBeCloseTo(49, 5);
      expect(result[3]).toBeCloseTo(64, 5);
    });

    it('should handle identity matrix', () => {
      const a = new Float32Array([1, 0, 0, 1]); // 2x2 identity
      const b = new Float32Array([5, 6, 7, 8]); // 2x2

      const result = wasmMatmul(a, b, 2, 2, 2);

      expect(result[0]).toBeCloseTo(5, 5);
      expect(result[1]).toBeCloseTo(6, 5);
      expect(result[2]).toBeCloseTo(7, 5);
      expect(result[3]).toBeCloseTo(8, 5);
    });
  });

  describe('wasmScatterAdd', () => {
    it('should scatter-add features correctly', () => {
      // Source features: 3 edges x 2 features
      const src = new Float32Array([1, 2, 3, 4, 5, 6]);
      // Target indices: edge 0->node 0, edge 1->node 1, edge 2->node 0
      const index = new Uint32Array([0, 1, 0]);

      const result = wasmScatterAdd(src, index, 2, 2);

      // Node 0: [1,2] + [5,6] = [6,8]
      // Node 1: [3,4]
      expect(result.length).toBe(4);
      expect(result[0]).toBeCloseTo(6, 5);
      expect(result[1]).toBeCloseTo(8, 5);
      expect(result[2]).toBeCloseTo(3, 5);
      expect(result[3]).toBeCloseTo(4, 5);
    });
  });

  describe('wasmScatterMean', () => {
    it('should scatter-mean features correctly', () => {
      // Source features: 4 edges x 2 features
      const src = new Float32Array([2, 4, 6, 8, 4, 6, 10, 12]);
      // Target indices: edges 0,2->node 0, edges 1,3->node 1
      const index = new Uint32Array([0, 1, 0, 1]);

      const result = wasmScatterMean(src, index, 2, 2);

      // Node 0: mean([2,4], [4,6]) = [3, 5]
      // Node 1: mean([6,8], [10,12]) = [8, 10]
      expect(result.length).toBe(4);
      expect(result[0]).toBeCloseTo(3, 5);
      expect(result[1]).toBeCloseTo(5, 5);
      expect(result[2]).toBeCloseTo(8, 5);
      expect(result[3]).toBeCloseTo(10, 5);
    });
  });

  describe('wasmScatterMax', () => {
    it('should scatter-max features correctly', () => {
      // Source features: 3 edges x 2 features
      const src = new Float32Array([1, 5, 3, 2, 2, 6]);
      // Target indices: edges 0,2->node 0, edge 1->node 1
      const index = new Uint32Array([0, 1, 0]);

      const result = wasmScatterMax(src, index, 2, 2);

      // Node 0: max([1,5], [2,6]) = [2, 6]
      // Node 1: [3,2]
      expect(result.length).toBe(4);
      expect(result[0]).toBeCloseTo(2, 5);
      expect(result[1]).toBeCloseTo(6, 5);
      expect(result[2]).toBeCloseTo(3, 5);
      expect(result[3]).toBeCloseTo(2, 5);
    });
  });

  describe('wasmGather', () => {
    it('should gather features correctly', () => {
      // Source features: 3 nodes x 2 features
      const src = new Float32Array([1, 2, 3, 4, 5, 6]);
      // Gather from nodes [2, 0, 1]
      const index = new Uint32Array([2, 0, 1]);

      const result = wasmGather(src, index, 2);

      expect(result.length).toBe(6);
      expect(result[0]).toBeCloseTo(5, 5); // node 2
      expect(result[1]).toBeCloseTo(6, 5);
      expect(result[2]).toBeCloseTo(1, 5); // node 0
      expect(result[3]).toBeCloseTo(2, 5);
      expect(result[4]).toBeCloseTo(3, 5); // node 1
      expect(result[5]).toBeCloseTo(4, 5);
    });
  });

  describe('wasmRelu', () => {
    it('should apply ReLU correctly', () => {
      const x = new Float32Array([-2, -1, 0, 1, 2]);
      const result = wasmRelu(x);

      expect(result[0]).toBe(0);
      expect(result[1]).toBe(0);
      expect(result[2]).toBe(0);
      expect(result[3]).toBe(1);
      expect(result[4]).toBe(2);
    });
  });

  describe('wasmAdd', () => {
    it('should add arrays element-wise', () => {
      const a = new Float32Array([1, 2, 3, 4]);
      const b = new Float32Array([5, 6, 7, 8]);

      const result = wasmAdd(a, b);

      expect(result[0]).toBe(6);
      expect(result[1]).toBe(8);
      expect(result[2]).toBe(10);
      expect(result[3]).toBe(12);
    });
  });

  describe('wasmSpmv', () => {
    it('should perform sparse matrix-vector multiplication', () => {
      // CSR matrix:
      // [1 2 0]
      // [0 3 4]
      // [5 0 6]
      const rowPtr = new Uint32Array([0, 2, 4, 6]);
      const colIndices = new Uint32Array([0, 1, 1, 2, 0, 2]);
      const values = new Float32Array([1, 2, 3, 4, 5, 6]);
      const x = new Float32Array([1, 2, 3]);

      const result = wasmSpmv(rowPtr, colIndices, values, x, 3);

      // Expected: [1*1+2*2, 3*2+4*3, 5*1+6*3] = [5, 18, 23]
      expect(result[0]).toBeCloseTo(5, 5);
      expect(result[1]).toBeCloseTo(18, 5);
      expect(result[2]).toBeCloseTo(23, 5);
    });
  });

  describe('wasmSpmm', () => {
    it('should perform sparse matrix-dense matrix multiplication', () => {
      // CSR matrix (2x2):
      // [1 2]
      // [3 4]
      const rowPtr = new Uint32Array([0, 2, 4]);
      const colIndices = new Uint32Array([0, 1, 0, 1]);
      const values = new Float32Array([1, 2, 3, 4]);

      // Dense matrix (2x2):
      // [1 2]
      // [3 4]
      const dense = new Float32Array([1, 2, 3, 4]);

      const result = wasmSpmm(rowPtr, colIndices, values, dense, 2, 2, 2);

      // Expected: [[1*1+2*3, 1*2+2*4], [3*1+4*3, 3*2+4*4]] = [[7,10],[15,22]]
      expect(result[0]).toBeCloseTo(7, 5);
      expect(result[1]).toBeCloseTo(10, 5);
      expect(result[2]).toBeCloseTo(15, 5);
      expect(result[3]).toBeCloseTo(22, 5);
    });
  });

  describe('WASMKernels class', () => {
    it('should be a singleton', () => {
      const instance1 = WASMKernels.getInstance();
      const instance2 = WASMKernels.getInstance();
      expect(instance1).toBe(instance2);
    });

    it('should have all kernel methods', () => {
      const kernels = WASMKernels.getInstance();
      expect(typeof kernels.matmul).toBe('function');
      expect(typeof kernels.scatterAdd).toBe('function');
      expect(typeof kernels.scatterMean).toBe('function');
      expect(typeof kernels.scatterMax).toBe('function');
      expect(typeof kernels.gather).toBe('function');
      expect(typeof kernels.relu).toBe('function');
      expect(typeof kernels.add).toBe('function');
      expect(typeof kernels.spmm).toBe('function');
    });
  });
});
