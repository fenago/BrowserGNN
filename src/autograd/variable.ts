/**
 * BrowserGNN by Dr. Lee
 * Autograd Variable System
 *
 * Implements automatic differentiation with computational graph tracking.
 * Inspired by PyTorch's autograd system.
 */

import { Tensor } from '../core/tensor';

/**
 * Backward function type - computes gradients with respect to inputs
 */
export type BackwardFn = (grad: Tensor) => Tensor[];

/**
 * Computational graph node representing an operation
 */
export interface GradNode {
  inputs: Variable[];
  backward: BackwardFn;
  name: string;
}

/**
 * Variable - A tensor wrapper with gradient tracking
 *
 * Variables form a computational graph during forward pass,
 * which is then traversed backward to compute gradients.
 */
export class Variable {
  readonly data: Tensor;
  grad: Tensor | null = null;
  readonly requiresGrad: boolean;
  readonly gradNode: GradNode | null;
  readonly name: string;

  // Track if this variable has been used in backward
  private _gradAccumulated = false;

  constructor(
    data: Tensor,
    options: {
      requiresGrad?: boolean;
      gradNode?: GradNode | null;
      name?: string;
    } = {}
  ) {
    this.data = data;
    this.requiresGrad = options.requiresGrad ?? false;
    this.gradNode = options.gradNode ?? null;
    this.name = options.name ?? 'var';

    // Initialize gradient tensor if tracking
    if (this.requiresGrad) {
      this.grad = Tensor.zeros(data.shape);
    }
  }

  /**
   * Get tensor shape
   */
  get shape(): number[] {
    return this.data.shape;
  }

  /**
   * Get tensor size
   */
  get size(): number {
    return this.data.size;
  }

  /**
   * Create a variable from raw data
   */
  static fromArray(
    data: number[] | Float32Array,
    shape: number[],
    requiresGrad = false,
    name = 'var'
  ): Variable {
    const tensor = new Tensor(
      data instanceof Float32Array ? data : new Float32Array(data),
      shape
    );
    return new Variable(tensor, { requiresGrad, name });
  }

  /**
   * Create zeros variable
   */
  static zeros(shape: number[], requiresGrad = false): Variable {
    return new Variable(Tensor.zeros(shape), { requiresGrad });
  }

  /**
   * Create ones variable
   */
  static ones(shape: number[], requiresGrad = false): Variable {
    return new Variable(Tensor.ones(shape), { requiresGrad });
  }

  /**
   * Create random variable (uniform 0-1)
   */
  static rand(shape: number[], requiresGrad = false): Variable {
    return new Variable(Tensor.rand(shape), { requiresGrad });
  }

  /**
   * Create random variable (normal distribution)
   */
  static randn(shape: number[], requiresGrad = false): Variable {
    return new Variable(Tensor.randn(shape), { requiresGrad });
  }

  /**
   * Detach from computational graph
   */
  detach(): Variable {
    return new Variable(this.data, { requiresGrad: false, name: this.name + '_detached' });
  }

  /**
   * Zero gradients
   */
  zeroGrad(): void {
    if (this.grad) {
      this.grad.data.fill(0);
    }
    this._gradAccumulated = false;
  }

  /**
   * Accumulate gradient
   */
  accumulateGrad(grad: Tensor): void {
    if (!this.requiresGrad) return;

    if (this.grad === null) {
      this.grad = Tensor.zeros(this.data.shape);
    }

    // Add gradient (allows accumulation from multiple paths)
    for (let i = 0; i < this.grad.size; i++) {
      this.grad.data[i]! += grad.data[i]!;
    }
    this._gradAccumulated = true;
  }

  /**
   * Backward pass - compute gradients through the graph
   *
   * @param grad Optional upstream gradient (defaults to ones for scalar loss)
   */
  backward(grad?: Tensor): void {
    if (!this.requiresGrad && !this.gradNode) {
      throw new Error('Cannot call backward on a variable that does not require gradients');
    }

    // Initialize gradient for this node
    const upstreamGrad = grad ?? Tensor.ones(this.data.shape);

    // Topological sort of the computational graph
    const topoOrder: Variable[] = [];
    const visited = new Set<Variable>();

    const buildTopoOrder = (v: Variable) => {
      if (visited.has(v)) return;
      visited.add(v);

      if (v.gradNode) {
        for (const input of v.gradNode.inputs) {
          if (input.requiresGrad || input.gradNode) {
            buildTopoOrder(input);
          }
        }
      }
      topoOrder.push(v);
    };

    buildTopoOrder(this);

    // Initialize gradient for the output node
    this.accumulateGrad(upstreamGrad);

    // Reverse topological order for backward pass
    for (let i = topoOrder.length - 1; i >= 0; i--) {
      const v = topoOrder[i]!;

      if (v.gradNode && v.grad) {
        // Compute gradients for inputs
        const inputGrads = v.gradNode.backward(v.grad);

        // Accumulate gradients to inputs
        for (let j = 0; j < v.gradNode.inputs.length; j++) {
          const input = v.gradNode.inputs[j]!;
          const inputGrad = inputGrads[j];
          if (input.requiresGrad && inputGrad) {
            input.accumulateGrad(inputGrad);
          }
        }
      }
    }
  }

  // ==================== Operations ====================

  /**
   * Matrix multiplication: this @ other
   */
  matmul(other: Variable): Variable {
    const result = this.data.matmul(other.data);
    const needsGrad = this.requiresGrad || other.requiresGrad;

    const gradNode: GradNode | null = needsGrad
      ? {
          inputs: [this, other],
          backward: (grad: Tensor) => {
            // d(A @ B) / dA = grad @ B^T
            // d(A @ B) / dB = A^T @ grad
            const gradA = this.requiresGrad
              ? grad.matmul(other.data.transpose())
              : Tensor.zeros(this.data.shape);

            const gradB = other.requiresGrad
              ? this.data.transpose().matmul(grad)
              : Tensor.zeros(other.data.shape);

            return [gradA, gradB];
          },
          name: 'matmul',
        }
      : null;

    return new Variable(result, { requiresGrad: needsGrad, gradNode });
  }

  /**
   * Element-wise addition
   */
  add(other: Variable): Variable {
    const result = this.data.add(other.data);
    const needsGrad = this.requiresGrad || other.requiresGrad;

    const gradNode: GradNode | null = needsGrad
      ? {
          inputs: [this, other],
          backward: (grad: Tensor) => {
            // d(A + B) / dA = grad
            // d(A + B) / dB = grad (with broadcasting support)
            let gradA = grad;
            let gradB = grad;

            // Handle broadcasting for gradA
            if (this.data.shape.length !== grad.shape.length ||
                !this.data.shape.every((d, i) => d === grad.shape[i])) {
              gradA = this._reduceBroadcast(grad, this.data.shape);
            }

            // Handle broadcasting for gradB
            if (other.data.shape.length !== grad.shape.length ||
                !other.data.shape.every((d, i) => d === grad.shape[i])) {
              gradB = this._reduceBroadcast(grad, other.data.shape);
            }

            return [gradA, gradB];
          },
          name: 'add',
        }
      : null;

    return new Variable(result, { requiresGrad: needsGrad, gradNode });
  }

  /**
   * Element-wise subtraction
   */
  sub(other: Variable): Variable {
    const result = this.data.sub(other.data);
    const needsGrad = this.requiresGrad || other.requiresGrad;

    const gradNode: GradNode | null = needsGrad
      ? {
          inputs: [this, other],
          backward: (grad: Tensor) => {
            let gradA = grad;
            let gradB = grad.neg();

            if (!this.data.shape.every((d, i) => d === grad.shape[i])) {
              gradA = this._reduceBroadcast(grad, this.data.shape);
            }
            if (!other.data.shape.every((d, i) => d === grad.shape[i])) {
              gradB = this._reduceBroadcast(grad.neg(), other.data.shape);
            }

            return [gradA, gradB];
          },
          name: 'sub',
        }
      : null;

    return new Variable(result, { requiresGrad: needsGrad, gradNode });
  }

  /**
   * Element-wise multiplication
   */
  mul(other: Variable): Variable {
    const result = this.data.mul(other.data);
    const needsGrad = this.requiresGrad || other.requiresGrad;

    const gradNode: GradNode | null = needsGrad
      ? {
          inputs: [this, other],
          backward: (grad: Tensor) => {
            // d(A * B) / dA = grad * B
            // d(A * B) / dB = grad * A
            const gradA = grad.mul(other.data);
            const gradB = grad.mul(this.data);
            return [gradA, gradB];
          },
          name: 'mul',
        }
      : null;

    return new Variable(result, { requiresGrad: needsGrad, gradNode });
  }

  /**
   * Element-wise division
   */
  div(other: Variable): Variable {
    const result = this.data.div(other.data);
    const needsGrad = this.requiresGrad || other.requiresGrad;

    const gradNode: GradNode | null = needsGrad
      ? {
          inputs: [this, other],
          backward: (grad: Tensor) => {
            // d(A / B) / dA = grad / B
            // d(A / B) / dB = -grad * A / B^2
            const gradA = grad.div(other.data);
            const gradB = grad.mul(this.data).div(other.data.mul(other.data)).neg();
            return [gradA, gradB];
          },
          name: 'div',
        }
      : null;

    return new Variable(result, { requiresGrad: needsGrad, gradNode });
  }

  /**
   * ReLU activation
   */
  relu(): Variable {
    const result = this.data.relu();
    const needsGrad = this.requiresGrad;

    const gradNode: GradNode | null = needsGrad
      ? {
          inputs: [this],
          backward: (grad: Tensor) => {
            // d(relu(x)) / dx = 1 if x > 0, else 0
            const gradData = new Float32Array(grad.size);
            for (let i = 0; i < grad.size; i++) {
              gradData[i] = this.data.data[i]! > 0 ? grad.data[i]! : 0;
            }
            return [new Tensor(gradData, grad.shape)];
          },
          name: 'relu',
        }
      : null;

    return new Variable(result, { requiresGrad: needsGrad, gradNode });
  }

  /**
   * Sigmoid activation
   */
  sigmoid(): Variable {
    const resultData = new Float32Array(this.data.size);
    for (let i = 0; i < this.data.size; i++) {
      resultData[i] = 1 / (1 + Math.exp(-this.data.data[i]!));
    }
    const result = new Tensor(resultData, this.data.shape);
    const needsGrad = this.requiresGrad;

    const gradNode: GradNode | null = needsGrad
      ? {
          inputs: [this],
          backward: (grad: Tensor) => {
            // d(sigmoid(x)) / dx = sigmoid(x) * (1 - sigmoid(x))
            const gradData = new Float32Array(grad.size);
            for (let i = 0; i < grad.size; i++) {
              const s = resultData[i]!;
              gradData[i] = grad.data[i]! * s * (1 - s);
            }
            return [new Tensor(gradData, grad.shape)];
          },
          name: 'sigmoid',
        }
      : null;

    return new Variable(result, { requiresGrad: needsGrad, gradNode });
  }

  /**
   * Tanh activation
   */
  tanh(): Variable {
    const resultData = new Float32Array(this.data.size);
    for (let i = 0; i < this.data.size; i++) {
      resultData[i] = Math.tanh(this.data.data[i]!);
    }
    const result = new Tensor(resultData, this.data.shape);
    const needsGrad = this.requiresGrad;

    const gradNode: GradNode | null = needsGrad
      ? {
          inputs: [this],
          backward: (grad: Tensor) => {
            // d(tanh(x)) / dx = 1 - tanh(x)^2
            const gradData = new Float32Array(grad.size);
            for (let i = 0; i < grad.size; i++) {
              const t = resultData[i]!;
              gradData[i] = grad.data[i]! * (1 - t * t);
            }
            return [new Tensor(gradData, grad.shape)];
          },
          name: 'tanh',
        }
      : null;

    return new Variable(result, { requiresGrad: needsGrad, gradNode });
  }

  /**
   * Softmax (along last dimension)
   */
  softmax(): Variable {
    const [rows, cols] = this.data.shape.length === 2
      ? this.data.shape as [number, number]
      : [1, this.data.size];

    const resultData = new Float32Array(this.data.size);

    for (let i = 0; i < rows; i++) {
      // Find max for numerical stability
      let max = -Infinity;
      for (let j = 0; j < cols; j++) {
        max = Math.max(max, this.data.data[i * cols + j]!);
      }

      // Compute exp and sum
      let sum = 0;
      for (let j = 0; j < cols; j++) {
        const exp = Math.exp(this.data.data[i * cols + j]! - max);
        resultData[i * cols + j] = exp;
        sum += exp;
      }

      // Normalize
      for (let j = 0; j < cols; j++) {
        resultData[i * cols + j]! /= sum;
      }
    }

    const result = new Tensor(resultData, this.data.shape);
    const needsGrad = this.requiresGrad;

    const gradNode: GradNode | null = needsGrad
      ? {
          inputs: [this],
          backward: (grad: Tensor) => {
            // Softmax Jacobian: diag(s) - s @ s^T
            // For efficiency, use: grad_input[i] = s[i] * (grad[i] - sum(grad * s))
            const gradData = new Float32Array(grad.size);

            for (let i = 0; i < rows; i++) {
              // Compute sum(grad * softmax)
              let dotProduct = 0;
              for (let j = 0; j < cols; j++) {
                dotProduct += grad.data[i * cols + j]! * resultData[i * cols + j]!;
              }

              // Compute gradient
              for (let j = 0; j < cols; j++) {
                const s = resultData[i * cols + j]!;
                gradData[i * cols + j] = s * (grad.data[i * cols + j]! - dotProduct);
              }
            }

            return [new Tensor(gradData, grad.shape)];
          },
          name: 'softmax',
        }
      : null;

    return new Variable(result, { requiresGrad: needsGrad, gradNode });
  }

  /**
   * Log softmax (more numerically stable for cross-entropy)
   */
  logSoftmax(): Variable {
    const [rows, cols] = this.data.shape.length === 2
      ? this.data.shape as [number, number]
      : [1, this.data.size];

    const resultData = new Float32Array(this.data.size);

    for (let i = 0; i < rows; i++) {
      // Find max for numerical stability
      let max = -Infinity;
      for (let j = 0; j < cols; j++) {
        max = Math.max(max, this.data.data[i * cols + j]!);
      }

      // Compute log-sum-exp
      let logSumExp = 0;
      for (let j = 0; j < cols; j++) {
        logSumExp += Math.exp(this.data.data[i * cols + j]! - max);
      }
      logSumExp = max + Math.log(logSumExp);

      // Compute log-softmax
      for (let j = 0; j < cols; j++) {
        resultData[i * cols + j] = this.data.data[i * cols + j]! - logSumExp;
      }
    }

    const result = new Tensor(resultData, this.data.shape);
    const needsGrad = this.requiresGrad;

    const gradNode: GradNode | null = needsGrad
      ? {
          inputs: [this],
          backward: (grad: Tensor) => {
            // d(log_softmax) / dx = grad - softmax * sum(grad)
            const gradData = new Float32Array(grad.size);

            for (let i = 0; i < rows; i++) {
              // Sum of upstream gradients
              let gradSum = 0;
              for (let j = 0; j < cols; j++) {
                gradSum += grad.data[i * cols + j]!;
              }

              // Compute gradient
              for (let j = 0; j < cols; j++) {
                const softmax = Math.exp(resultData[i * cols + j]!);
                gradData[i * cols + j] = grad.data[i * cols + j]! - softmax * gradSum;
              }
            }

            return [new Tensor(gradData, grad.shape)];
          },
          name: 'log_softmax',
        }
      : null;

    return new Variable(result, { requiresGrad: needsGrad, gradNode });
  }

  /**
   * Sum all elements
   */
  sum(): Variable {
    let total = 0;
    for (let i = 0; i < this.data.size; i++) {
      total += this.data.data[i]!;
    }
    const result = new Tensor(new Float32Array([total]), [1]);
    const needsGrad = this.requiresGrad;

    const gradNode: GradNode | null = needsGrad
      ? {
          inputs: [this],
          backward: (grad: Tensor) => {
            // d(sum(x)) / dx = ones * grad
            const gradData = new Float32Array(this.data.size);
            gradData.fill(grad.data[0]!);
            return [new Tensor(gradData, this.data.shape)];
          },
          name: 'sum',
        }
      : null;

    return new Variable(result, { requiresGrad: needsGrad, gradNode });
  }

  /**
   * Mean of all elements
   */
  mean(): Variable {
    let total = 0;
    for (let i = 0; i < this.data.size; i++) {
      total += this.data.data[i]!;
    }
    const result = new Tensor(new Float32Array([total / this.data.size]), [1]);
    const needsGrad = this.requiresGrad;

    const gradNode: GradNode | null = needsGrad
      ? {
          inputs: [this],
          backward: (grad: Tensor) => {
            // d(mean(x)) / dx = ones / n * grad
            const gradData = new Float32Array(this.data.size);
            gradData.fill(grad.data[0]! / this.data.size);
            return [new Tensor(gradData, this.data.shape)];
          },
          name: 'mean',
        }
      : null;

    return new Variable(result, { requiresGrad: needsGrad, gradNode });
  }

  /**
   * Negation
   */
  neg(): Variable {
    const result = this.data.neg();
    const needsGrad = this.requiresGrad;

    const gradNode: GradNode | null = needsGrad
      ? {
          inputs: [this],
          backward: (grad: Tensor) => {
            return [grad.neg()];
          },
          name: 'neg',
        }
      : null;

    return new Variable(result, { requiresGrad: needsGrad, gradNode });
  }

  /**
   * Transpose (2D only)
   */
  transpose(): Variable {
    const result = this.data.transpose();
    const needsGrad = this.requiresGrad;

    const gradNode: GradNode | null = needsGrad
      ? {
          inputs: [this],
          backward: (grad: Tensor) => {
            return [grad.transpose()];
          },
          name: 'transpose',
        }
      : null;

    return new Variable(result, { requiresGrad: needsGrad, gradNode });
  }

  /**
   * Helper: Reduce gradient for broadcasting
   */
  private _reduceBroadcast(grad: Tensor, targetShape: number[]): Tensor {
    if (targetShape.length === 0 || (targetShape.length === 1 && targetShape[0] === 1)) {
      // Scalar - sum all
      let sum = 0;
      for (let i = 0; i < grad.size; i++) {
        sum += grad.data[i]!;
      }
      return new Tensor(new Float32Array([sum]), [1]);
    }

    // For bias [outChannels] broadcast to [batchSize, outChannels]
    if (targetShape.length === 1 && grad.shape.length === 2) {
      const [rows, cols] = grad.shape;
      const reduced = new Float32Array(targetShape[0]!);
      for (let i = 0; i < rows!; i++) {
        for (let j = 0; j < cols!; j++) {
          reduced[j]! += grad.data[i * cols! + j]!;
        }
      }
      return new Tensor(reduced, targetShape);
    }

    return grad;
  }
}
