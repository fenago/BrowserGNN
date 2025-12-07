/**
 * BrowserGNN by Dr. Lee
 * Tensor Tests
 */

import { describe, it, expect } from 'vitest';
import { Tensor, concat, stack } from '../src/core/tensor';

describe('Tensor', () => {
  describe('creation', () => {
    it('should create a tensor from Float32Array', () => {
      const data = new Float32Array([1, 2, 3, 4, 5, 6]);
      const tensor = new Tensor(data, [2, 3]);

      expect(tensor.shape).toEqual([2, 3]);
      expect(tensor.size).toBe(6);
      expect(tensor.ndim).toBe(2);
    });

    it('should create a tensor from number array', () => {
      const tensor = new Tensor([1, 2, 3, 4], [2, 2]);

      expect(tensor.shape).toEqual([2, 2]);
      expect(tensor.data[0]).toBe(1);
    });

    it('should throw on shape mismatch', () => {
      expect(() => {
        new Tensor([1, 2, 3], [2, 2]);
      }).toThrow();
    });
  });

  describe('static constructors', () => {
    it('should create zeros tensor', () => {
      const tensor = Tensor.zeros([3, 4]);

      expect(tensor.shape).toEqual([3, 4]);
      expect(tensor.sum()).toBe(0);
    });

    it('should create ones tensor', () => {
      const tensor = Tensor.ones([2, 3]);

      expect(tensor.shape).toEqual([2, 3]);
      expect(tensor.sum()).toBe(6);
    });

    it('should create random tensor', () => {
      const tensor = Tensor.rand([10, 10]);

      expect(tensor.shape).toEqual([10, 10]);
      expect(tensor.min()).toBeGreaterThanOrEqual(0);
      expect(tensor.max()).toBeLessThan(1);
    });

    it('should create identity matrix', () => {
      const eye = Tensor.eye(3);

      expect(eye.shape).toEqual([3, 3]);
      expect(eye.get(0, 0)).toBe(1);
      expect(eye.get(1, 1)).toBe(1);
      expect(eye.get(0, 1)).toBe(0);
    });
  });

  describe('operations', () => {
    it('should add tensors', () => {
      const a = new Tensor([1, 2, 3, 4], [2, 2]);
      const b = new Tensor([5, 6, 7, 8], [2, 2]);

      const result = a.add(b);

      expect(result.data[0]).toBe(6);
      expect(result.data[3]).toBe(12);
    });

    it('should add scalar', () => {
      const a = new Tensor([1, 2, 3], [3]);
      const result = a.add(10);

      expect(result.data[0]).toBe(11);
      expect(result.data[2]).toBe(13);
    });

    it('should multiply tensors', () => {
      const a = new Tensor([1, 2, 3, 4], [2, 2]);
      const b = new Tensor([2, 2, 2, 2], [2, 2]);

      const result = a.mul(b);

      expect(result.data[0]).toBe(2);
      expect(result.data[3]).toBe(8);
    });

    it('should perform matrix multiplication', () => {
      const a = new Tensor([1, 2, 3, 4], [2, 2]);
      const b = new Tensor([5, 6, 7, 8], [2, 2]);

      const result = a.matmul(b);

      expect(result.shape).toEqual([2, 2]);
      expect(result.get(0, 0)).toBe(1 * 5 + 2 * 7); // 19
      expect(result.get(0, 1)).toBe(1 * 6 + 2 * 8); // 22
    });

    it('should transpose 2D tensor', () => {
      const a = new Tensor([1, 2, 3, 4, 5, 6], [2, 3]);
      const t = a.transpose();

      expect(t.shape).toEqual([3, 2]);
      expect(t.get(0, 0)).toBe(1);
      expect(t.get(0, 1)).toBe(4);
      expect(t.get(1, 0)).toBe(2);
    });
  });

  describe('activations', () => {
    it('should apply ReLU', () => {
      const a = new Tensor([-1, 0, 1, 2], [4]);
      const result = a.relu();

      expect(result.data[0]).toBe(0);
      expect(result.data[1]).toBe(0);
      expect(result.data[2]).toBe(1);
      expect(result.data[3]).toBe(2);
    });

    it('should apply sigmoid', () => {
      const a = new Tensor([0], [1]);
      const result = a.sigmoid();

      expect(result.data[0]).toBeCloseTo(0.5);
    });

    it('should apply softmax', () => {
      const a = new Tensor([1, 2, 3], [1, 3]);
      const result = a.softmax();

      // Sum should be 1
      const sum = result.data[0]! + result.data[1]! + result.data[2]!;
      expect(sum).toBeCloseTo(1);

      // Values should be in order
      expect(result.data[0]).toBeLessThan(result.data[1]!);
      expect(result.data[1]).toBeLessThan(result.data[2]!);
    });
  });

  describe('aggregations', () => {
    it('should compute sum', () => {
      const a = new Tensor([1, 2, 3, 4], [2, 2]);
      expect(a.sum()).toBe(10);
    });

    it('should compute mean', () => {
      const a = new Tensor([2, 4, 6, 8], [4]);
      expect(a.mean()).toBe(5);
    });

    it('should find max', () => {
      const a = new Tensor([1, 5, 3, 2], [4]);
      expect(a.max()).toBe(5);
    });
  });
});

describe('concat', () => {
  it('should concatenate along axis 0', () => {
    const a = new Tensor([1, 2, 3, 4], [2, 2]);
    const b = new Tensor([5, 6, 7, 8], [2, 2]);

    const result = concat([a, b], 0);

    expect(result.shape).toEqual([4, 2]);
  });

  it('should concatenate along axis 1', () => {
    const a = new Tensor([1, 2, 3, 4], [2, 2]);
    const b = new Tensor([5, 6, 7, 8], [2, 2]);

    const result = concat([a, b], 1);

    expect(result.shape).toEqual([2, 4]);
  });
});

describe('stack', () => {
  it('should stack tensors', () => {
    const a = new Tensor([1, 2], [2]);
    const b = new Tensor([3, 4], [2]);
    const c = new Tensor([5, 6], [2]);

    const result = stack([a, b, c], 0);

    expect(result.shape).toEqual([3, 2]);
  });
});
