/**
 * BrowserGNN by Dr. Lee
 * Activation Functions
 *
 * Common activation functions as Module wrappers.
 */

import { Tensor } from '../core/tensor';
import { GraphData } from '../core/graph';
import { Module } from './module';

/**
 * ReLU activation: max(0, x)
 */
export class ReLU extends Module {
  forward(input: Tensor | GraphData): Tensor | GraphData {
    if (input instanceof GraphData) {
      const newX = input.x.relu();
      return input.withFeatures(newX);
    }
    return input.relu();
  }
}

/**
 * Leaky ReLU activation: max(negative_slope * x, x)
 */
export class LeakyReLU extends Module {
  private negativeSlope: number;

  constructor(negativeSlope: number = 0.01) {
    super();
    this.negativeSlope = negativeSlope;
  }

  forward(input: Tensor | GraphData): Tensor | GraphData {
    if (input instanceof GraphData) {
      const newX = input.x.leakyRelu(this.negativeSlope);
      return input.withFeatures(newX);
    }
    return input.leakyRelu(this.negativeSlope);
  }
}

/**
 * ELU activation: x if x > 0, alpha * (exp(x) - 1) otherwise
 */
export class ELU extends Module {
  private alpha: number;

  constructor(alpha: number = 1.0) {
    super();
    this.alpha = alpha;
  }

  forward(input: Tensor | GraphData): Tensor | GraphData {
    const applyELU = (tensor: Tensor): Tensor => {
      const result = new Float32Array(tensor.size);
      for (let i = 0; i < tensor.size; i++) {
        const x = tensor.data[i]!;
        result[i] = x > 0 ? x : this.alpha * (Math.exp(x) - 1);
      }
      return new Tensor(result, tensor.shape);
    };

    if (input instanceof GraphData) {
      return input.withFeatures(applyELU(input.x));
    }
    return applyELU(input);
  }
}

/**
 * Sigmoid activation: 1 / (1 + exp(-x))
 */
export class Sigmoid extends Module {
  forward(input: Tensor | GraphData): Tensor | GraphData {
    if (input instanceof GraphData) {
      const newX = input.x.sigmoid();
      return input.withFeatures(newX);
    }
    return input.sigmoid();
  }
}

/**
 * Tanh activation
 */
export class Tanh extends Module {
  forward(input: Tensor | GraphData): Tensor | GraphData {
    if (input instanceof GraphData) {
      const newX = input.x.tanh();
      return input.withFeatures(newX);
    }
    return input.tanh();
  }
}

/**
 * Softmax activation (along last axis)
 */
export class Softmax extends Module {
  forward(input: Tensor | GraphData): Tensor | GraphData {
    if (input instanceof GraphData) {
      const newX = input.x.softmax();
      return input.withFeatures(newX);
    }
    return input.softmax();
  }
}

/**
 * Log Softmax activation (along last axis)
 */
export class LogSoftmax extends Module {
  forward(input: Tensor | GraphData): Tensor | GraphData {
    if (input instanceof GraphData) {
      const newX = input.x.logSoftmax();
      return input.withFeatures(newX);
    }
    return input.logSoftmax();
  }
}

/**
 * GELU activation: x * Φ(x) where Φ is the CDF of standard normal
 * Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
 */
export class GELU extends Module {
  forward(input: Tensor | GraphData): Tensor | GraphData {
    const applyGELU = (tensor: Tensor): Tensor => {
      const result = new Float32Array(tensor.size);
      const sqrt2OverPi = Math.sqrt(2 / Math.PI);

      for (let i = 0; i < tensor.size; i++) {
        const x = tensor.data[i]!;
        const inner = sqrt2OverPi * (x + 0.044715 * x * x * x);
        result[i] = 0.5 * x * (1 + Math.tanh(inner));
      }
      return new Tensor(result, tensor.shape);
    };

    if (input instanceof GraphData) {
      return input.withFeatures(applyGELU(input.x));
    }
    return applyGELU(input);
  }
}

/**
 * SiLU/Swish activation: x * sigmoid(x)
 */
export class SiLU extends Module {
  forward(input: Tensor | GraphData): Tensor | GraphData {
    const applySiLU = (tensor: Tensor): Tensor => {
      const result = new Float32Array(tensor.size);
      for (let i = 0; i < tensor.size; i++) {
        const x = tensor.data[i]!;
        result[i] = x / (1 + Math.exp(-x));
      }
      return new Tensor(result, tensor.shape);
    };

    if (input instanceof GraphData) {
      return input.withFeatures(applySiLU(input.x));
    }
    return applySiLU(input);
  }
}

/**
 * PReLU activation: max(0, x) + a * min(0, x)
 * Learnable parameter 'a'
 */
export class PReLU extends Module {
  constructor(numParameters: number = 1, init: number = 0.25) {
    super();
    const weight = new Tensor(new Float32Array(numParameters).fill(init), [numParameters]);
    this.registerParameter('weight', weight);
  }

  forward(input: Tensor | GraphData): Tensor | GraphData {
    const weight = this._parameters.get('weight')!.tensor;

    const applyPReLU = (tensor: Tensor): Tensor => {
      const result = new Float32Array(tensor.size);
      const numParams = weight.size;

      for (let i = 0; i < tensor.size; i++) {
        const x = tensor.data[i]!;
        const a = numParams === 1 ? weight.data[0]! : weight.data[i % numParams]!;
        result[i] = x > 0 ? x : a * x;
      }
      return new Tensor(result, tensor.shape);
    };

    if (input instanceof GraphData) {
      return input.withFeatures(applyPReLU(input.x));
    }
    return applyPReLU(input);
  }
}

/**
 * Dropout layer
 */
export class Dropout extends Module {
  private p: number;

  constructor(p: number = 0.5) {
    super();
    if (p < 0 || p > 1) {
      throw new Error('Dropout probability must be between 0 and 1');
    }
    this.p = p;
  }

  forward(input: Tensor | GraphData): Tensor | GraphData {
    // Only apply dropout during training
    if (!this._training || this.p === 0) {
      if (input instanceof GraphData) {
        return input;
      }
      return input;
    }

    const applyDropout = (tensor: Tensor): Tensor => {
      const result = new Float32Array(tensor.size);
      const scale = 1 / (1 - this.p);

      for (let i = 0; i < tensor.size; i++) {
        if (Math.random() > this.p) {
          result[i] = tensor.data[i]! * scale;
        } else {
          result[i] = 0;
        }
      }
      return new Tensor(result, tensor.shape);
    };

    if (input instanceof GraphData) {
      return input.withFeatures(applyDropout(input.x));
    }
    return applyDropout(input);
  }
}
