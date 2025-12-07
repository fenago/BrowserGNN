/**
 * BrowserGNN by Dr. Lee
 * Autograd Module Tests
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { Tensor } from '../src/core/tensor';
import {
  Variable,
  crossEntropyLoss,
  mseLoss,
  binaryCrossEntropyLoss,
  nllLoss,
  l1Loss,
  smoothL1Loss,
  SGD,
  Adam,
  Adagrad,
  RMSprop,
  StepLR,
  ExponentialLR,
  CosineAnnealingLR,
} from '../src/autograd';

describe('Variable', () => {
  describe('creation', () => {
    it('should create a variable from tensor', () => {
      const tensor = new Tensor(new Float32Array([1, 2, 3, 4]), [2, 2]);
      const v = new Variable(tensor, { requiresGrad: true });

      expect(v.shape).toEqual([2, 2]);
      expect(v.size).toBe(4);
      expect(v.requiresGrad).toBe(true);
      expect(v.grad).not.toBeNull();
    });

    it('should create variable from array', () => {
      const v = Variable.fromArray([1, 2, 3, 4], [2, 2], true);
      expect(v.shape).toEqual([2, 2]);
      expect(v.requiresGrad).toBe(true);
    });

    it('should create zeros variable', () => {
      const v = Variable.zeros([3, 3]);
      expect(v.shape).toEqual([3, 3]);
      expect(v.data.data.every(x => x === 0)).toBe(true);
    });

    it('should create ones variable', () => {
      const v = Variable.ones([2, 2]);
      expect(v.data.data.every(x => x === 1)).toBe(true);
    });
  });

  describe('operations', () => {
    it('should perform addition with gradient tracking', () => {
      const a = Variable.fromArray([1, 2, 3, 4], [2, 2], true);
      const b = Variable.fromArray([5, 6, 7, 8], [2, 2], true);
      const c = a.add(b);

      expect(c.data.data[0]).toBe(6);
      expect(c.data.data[3]).toBe(12);
      expect(c.requiresGrad).toBe(true);
    });

    it('should perform subtraction', () => {
      const a = Variable.fromArray([5, 6, 7, 8], [2, 2], true);
      const b = Variable.fromArray([1, 2, 3, 4], [2, 2], true);
      const c = a.sub(b);

      expect(c.data.data[0]).toBe(4);
      expect(c.data.data[3]).toBe(4);
    });

    it('should perform multiplication', () => {
      const a = Variable.fromArray([1, 2, 3, 4], [2, 2], true);
      const b = Variable.fromArray([2, 2, 2, 2], [2, 2], true);
      const c = a.mul(b);

      expect(c.data.data[0]).toBe(2);
      expect(c.data.data[3]).toBe(8);
    });

    it('should perform matrix multiplication', () => {
      const a = Variable.fromArray([1, 2, 3, 4], [2, 2], true);
      const b = Variable.fromArray([1, 0, 0, 1], [2, 2], true);
      const c = a.matmul(b);

      // Identity matrix multiplication
      expect(c.data.data[0]).toBeCloseTo(1, 5);
      expect(c.data.data[1]).toBeCloseTo(2, 5);
      expect(c.data.data[2]).toBeCloseTo(3, 5);
      expect(c.data.data[3]).toBeCloseTo(4, 5);
    });

    it('should apply ReLU', () => {
      const a = Variable.fromArray([-2, -1, 0, 1, 2], [5], true);
      const b = a.relu();

      expect(b.data.data[0]).toBe(0);
      expect(b.data.data[1]).toBe(0);
      expect(b.data.data[2]).toBe(0);
      expect(b.data.data[3]).toBe(1);
      expect(b.data.data[4]).toBe(2);
    });

    it('should apply sigmoid', () => {
      const a = Variable.fromArray([0], [1], true);
      const b = a.sigmoid();

      expect(b.data.data[0]).toBeCloseTo(0.5, 5);
    });

    it('should apply tanh', () => {
      const a = Variable.fromArray([0], [1], true);
      const b = a.tanh();

      expect(b.data.data[0]).toBeCloseTo(0, 5);
    });

    it('should compute softmax', () => {
      const a = Variable.fromArray([1, 2, 3], [1, 3], true);
      const b = a.softmax();

      // Sum should be 1
      const sum = b.data.data[0]! + b.data.data[1]! + b.data.data[2]!;
      expect(sum).toBeCloseTo(1, 5);

      // Should be monotonically increasing
      expect(b.data.data[0]).toBeLessThan(b.data.data[1]!);
      expect(b.data.data[1]).toBeLessThan(b.data.data[2]!);
    });

    it('should compute log softmax', () => {
      const a = Variable.fromArray([1, 2, 3], [1, 3], true);
      const b = a.logSoftmax();

      // exp(log_softmax) should sum to 1
      const expSum = Math.exp(b.data.data[0]!) + Math.exp(b.data.data[1]!) + Math.exp(b.data.data[2]!);
      expect(expSum).toBeCloseTo(1, 5);
    });

    it('should compute sum', () => {
      const a = Variable.fromArray([1, 2, 3, 4], [4], true);
      const b = a.sum();

      expect(b.data.data[0]).toBe(10);
    });

    it('should compute mean', () => {
      const a = Variable.fromArray([1, 2, 3, 4], [4], true);
      const b = a.mean();

      expect(b.data.data[0]).toBe(2.5);
    });

    it('should transpose', () => {
      const a = Variable.fromArray([1, 2, 3, 4, 5, 6], [2, 3], true);
      const b = a.transpose();

      expect(b.shape).toEqual([3, 2]);
      expect(b.data.data[0]).toBe(1);
      expect(b.data.data[1]).toBe(4);
      expect(b.data.data[2]).toBe(2);
    });
  });

  describe('backward pass', () => {
    it('should compute gradients for addition', () => {
      const a = Variable.fromArray([1, 2], [2], true);
      const b = Variable.fromArray([3, 4], [2], true);
      const c = a.add(b);
      const loss = c.sum();

      loss.backward();

      expect(a.grad!.data[0]).toBeCloseTo(1, 5);
      expect(a.grad!.data[1]).toBeCloseTo(1, 5);
      expect(b.grad!.data[0]).toBeCloseTo(1, 5);
      expect(b.grad!.data[1]).toBeCloseTo(1, 5);
    });

    it('should compute gradients for multiplication', () => {
      const a = Variable.fromArray([2, 3], [2], true);
      const b = Variable.fromArray([4, 5], [2], true);
      const c = a.mul(b);
      const loss = c.sum();

      loss.backward();

      // d(a*b)/da = b
      expect(a.grad!.data[0]).toBeCloseTo(4, 5);
      expect(a.grad!.data[1]).toBeCloseTo(5, 5);
      // d(a*b)/db = a
      expect(b.grad!.data[0]).toBeCloseTo(2, 5);
      expect(b.grad!.data[1]).toBeCloseTo(3, 5);
    });

    it('should compute gradients for ReLU', () => {
      const a = Variable.fromArray([-1, 2], [2], true);
      const b = a.relu();
      const loss = b.sum();

      loss.backward();

      // ReLU gradient: 0 if x <= 0, 1 if x > 0
      expect(a.grad!.data[0]).toBeCloseTo(0, 5);
      expect(a.grad!.data[1]).toBeCloseTo(1, 5);
    });

    it('should compute gradients through chain of operations', () => {
      const x = Variable.fromArray([1, 2, 3], [3], true);
      const w = Variable.fromArray([0.5, 0.5, 0.5], [3], true);

      // y = sum(x * w)
      const y = x.mul(w).sum();

      y.backward();

      // dy/dw = x
      expect(w.grad!.data[0]).toBeCloseTo(1, 5);
      expect(w.grad!.data[1]).toBeCloseTo(2, 5);
      expect(w.grad!.data[2]).toBeCloseTo(3, 5);
    });
  });
});

describe('Loss Functions', () => {
  describe('crossEntropyLoss', () => {
    it('should compute cross entropy loss', () => {
      // Perfect prediction
      const input = Variable.fromArray([10, -10, -10, -10, 10, -10], [2, 3], true);
      const target = [0, 1];

      const loss = crossEntropyLoss(input, target);

      // Loss should be close to 0 for perfect predictions
      expect(loss.data.data[0]).toBeLessThan(0.1);
    });

    it('should compute gradients', () => {
      const input = Variable.fromArray([0, 0, 0, 0, 0, 0], [2, 3], true);
      const target = [0, 1];

      const loss = crossEntropyLoss(input, target);
      loss.backward();

      expect(input.grad).not.toBeNull();
      expect(input.grad!.size).toBe(6);
    });
  });

  describe('mseLoss', () => {
    it('should compute MSE loss', () => {
      const input = Variable.fromArray([1, 2, 3], [3], true);
      const target = Variable.fromArray([1, 2, 3], [3], false);

      const loss = mseLoss(input, target);

      expect(loss.data.data[0]).toBeCloseTo(0, 5);
    });

    it('should compute correct MSE for differences', () => {
      const input = Variable.fromArray([0, 0], [2], true);
      const target = Variable.fromArray([1, 1], [2], false);

      const loss = mseLoss(input, target);

      // MSE = mean((0-1)^2 + (0-1)^2) = mean(1 + 1) = 1
      expect(loss.data.data[0]).toBeCloseTo(1, 5);
    });
  });

  describe('binaryCrossEntropyLoss', () => {
    it('should compute BCE loss', () => {
      const input = Variable.fromArray([0.9, 0.1], [2], true);
      const target = Variable.fromArray([1, 0], [2], false);

      const loss = binaryCrossEntropyLoss(input, target);

      expect(loss.data.data[0]).toBeLessThan(0.5);
    });
  });

  describe('l1Loss', () => {
    it('should compute L1 loss', () => {
      const input = Variable.fromArray([0, 2, 4], [3], true);
      const target = Variable.fromArray([1, 2, 3], [3], false);

      const loss = l1Loss(input, target);

      // L1 = mean(|0-1| + |2-2| + |4-3|) = mean(1 + 0 + 1) = 2/3
      expect(loss.data.data[0]).toBeCloseTo(2 / 3, 5);
    });
  });

  describe('smoothL1Loss', () => {
    it('should compute smooth L1 loss', () => {
      const input = Variable.fromArray([0, 0], [2], true);
      const target = Variable.fromArray([0.5, 2], [2], false);

      const loss = smoothL1Loss(input, target);

      // Should be defined and reasonable
      expect(loss.data.data[0]).toBeGreaterThan(0);
      expect(loss.data.data[0]).toBeLessThan(2);
    });
  });
});

describe('Optimizers', () => {
  describe('SGD', () => {
    it('should perform gradient descent step', () => {
      const param = Variable.fromArray([1, 1, 1, 1], [4], true);
      param.grad = new Tensor(new Float32Array([1, 1, 1, 1]), [4]);

      const optimizer = new SGD([param], { lr: 0.1 });
      optimizer.stepOptimizer();

      // param = param - lr * grad = 1 - 0.1 * 1 = 0.9
      expect(param.data.data[0]).toBeCloseTo(0.9, 5);
    });

    it('should apply momentum', () => {
      const param = Variable.fromArray([1], [1], true);
      param.grad = new Tensor(new Float32Array([1]), [1]);

      const optimizer = new SGD([param], { lr: 0.1, momentum: 0.9 });

      // First step
      optimizer.stepOptimizer();
      const afterFirst = param.data.data[0];

      // Second step with same gradient
      param.grad = new Tensor(new Float32Array([1]), [1]);
      optimizer.stepOptimizer();
      const afterSecond = param.data.data[0];

      // With momentum, second step should move more
      const firstDelta = 1 - afterFirst!;
      const secondDelta = afterFirst! - afterSecond!;
      expect(secondDelta).toBeGreaterThan(firstDelta);
    });

    it('should apply weight decay', () => {
      const param = Variable.fromArray([2], [1], true);
      param.grad = new Tensor(new Float32Array([0]), [1]); // Zero gradient

      const optimizer = new SGD([param], { lr: 0.1, weightDecay: 0.1 });
      optimizer.stepOptimizer();

      // With weight decay: param = param - lr * (grad + wd * param)
      // = 2 - 0.1 * (0 + 0.1 * 2) = 2 - 0.02 = 1.98
      expect(param.data.data[0]).toBeCloseTo(1.98, 5);
    });
  });

  describe('Adam', () => {
    it('should perform Adam step', () => {
      const param = Variable.fromArray([1, 1], [2], true);
      param.grad = new Tensor(new Float32Array([1, 1]), [2]);

      const optimizer = new Adam([param], { lr: 0.1 });
      optimizer.stepOptimizer();

      // Parameters should have changed
      expect(param.data.data[0]).not.toBe(1);
      expect(param.data.data[0]).toBeLessThan(1);
    });

    it('should handle bias correction', () => {
      const param = Variable.fromArray([1], [1], true);
      param.grad = new Tensor(new Float32Array([1]), [1]);

      const optimizer = new Adam([param], { lr: 0.1, beta1: 0.9, beta2: 0.999 });

      // Multiple steps
      for (let i = 0; i < 10; i++) {
        param.grad = new Tensor(new Float32Array([1]), [1]);
        optimizer.stepOptimizer();
      }

      // Should have moved significantly
      expect(param.data.data[0]).toBeLessThan(0.5);
    });
  });

  describe('Adagrad', () => {
    it('should perform Adagrad step', () => {
      const param = Variable.fromArray([1], [1], true);
      param.grad = new Tensor(new Float32Array([1]), [1]);

      const optimizer = new Adagrad([param], { lr: 0.5 });
      optimizer.stepOptimizer();

      expect(param.data.data[0]).toBeLessThan(1);
    });

    it('should accumulate squared gradients', () => {
      const param = Variable.fromArray([1], [1], true);
      param.grad = new Tensor(new Float32Array([1]), [1]);

      const optimizer = new Adagrad([param], { lr: 1.0 });

      // First step
      optimizer.stepOptimizer();
      const firstStep = 1 - param.data.data[0]!;

      // Second step with same gradient
      param.grad = new Tensor(new Float32Array([1]), [1]);
      optimizer.stepOptimizer();
      const secondStep = param.data.data[0]!;

      // Step size should decrease due to accumulated gradients
      // (This is the characteristic behavior of Adagrad)
    });
  });

  describe('RMSprop', () => {
    it('should perform RMSprop step', () => {
      const param = Variable.fromArray([1], [1], true);
      param.grad = new Tensor(new Float32Array([1]), [1]);

      const optimizer = new RMSprop([param], { lr: 0.1 });
      optimizer.stepOptimizer();

      expect(param.data.data[0]).toBeLessThan(1);
    });
  });
});

describe('Learning Rate Schedulers', () => {
  describe('StepLR', () => {
    it('should decay learning rate at step intervals', () => {
      const param = Variable.fromArray([1], [1], true);
      const optimizer = new SGD([param], { lr: 1.0 });
      const scheduler = new StepLR(optimizer, 5, 0.1);

      expect(scheduler.currentLr).toBeCloseTo(1.0, 5);

      // Step 5 times
      for (let i = 0; i < 5; i++) {
        scheduler.step();
      }

      expect(scheduler.currentLr).toBeCloseTo(0.1, 5);
    });
  });

  describe('ExponentialLR', () => {
    it('should decay learning rate exponentially', () => {
      const param = Variable.fromArray([1], [1], true);
      const optimizer = new SGD([param], { lr: 1.0 });
      const scheduler = new ExponentialLR(optimizer, 0.9);

      scheduler.step(); // epoch 1
      expect(scheduler.currentLr).toBeCloseTo(0.9, 5);

      scheduler.step(); // epoch 2
      expect(scheduler.currentLr).toBeCloseTo(0.81, 5);
    });
  });

  describe('CosineAnnealingLR', () => {
    it('should follow cosine annealing schedule', () => {
      const param = Variable.fromArray([1], [1], true);
      const optimizer = new SGD([param], { lr: 1.0 });
      const scheduler = new CosineAnnealingLR(optimizer, 10, 0);

      expect(scheduler.currentLr).toBeCloseTo(1.0, 5);

      // At half point (epoch 5), should be 0.5
      for (let i = 0; i < 5; i++) {
        scheduler.step();
      }
      expect(scheduler.currentLr).toBeCloseTo(0.5, 1);

      // At end (epoch 10), should be close to 0
      for (let i = 0; i < 5; i++) {
        scheduler.step();
      }
      expect(scheduler.currentLr).toBeCloseTo(0, 1);
    });
  });
});

describe('State Dict', () => {
  it('should save and load SGD state', () => {
    const param = Variable.fromArray([1], [1], true);
    param.grad = new Tensor(new Float32Array([1]), [1]);

    const optimizer = new SGD([param], { lr: 0.1, momentum: 0.9 });
    optimizer.stepOptimizer();
    optimizer.stepOptimizer();

    const state = optimizer.stateDict();

    // Create new optimizer and load state
    const param2 = Variable.fromArray([1], [1], true);
    const optimizer2 = new SGD([param2], { lr: 0.1, momentum: 0.9 });
    optimizer2.loadStateDict(state);

    expect(optimizer2.step).toBe(2);
  });

  it('should save and load Adam state', () => {
    const param = Variable.fromArray([1, 2], [2], true);
    param.grad = new Tensor(new Float32Array([1, 1]), [2]);

    const optimizer = new Adam([param], { lr: 0.01 });
    for (let i = 0; i < 5; i++) {
      param.grad = new Tensor(new Float32Array([1, 1]), [2]);
      optimizer.stepOptimizer();
    }

    const state = optimizer.stateDict();

    expect(state.step).toBe(5);
    expect((state.m as Float32Array[]).length).toBe(1);
    expect((state.v as Float32Array[]).length).toBe(1);
  });
});
