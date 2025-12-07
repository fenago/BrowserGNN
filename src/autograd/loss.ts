/**
 * BrowserGNN by Dr. Lee
 * Loss Functions for Training
 */

import { Tensor } from '../core/tensor';
import { Variable } from './variable';

/**
 * Cross-Entropy Loss
 *
 * For classification tasks. Combines log-softmax and NLL loss.
 * loss = -sum(target * log(softmax(input)))
 */
export function crossEntropyLoss(
  input: Variable,
  target: Uint32Array | number[],
  options: { reduction?: 'mean' | 'sum' | 'none' } = {}
): Variable {
  const reduction = options.reduction ?? 'mean';
  const [batchSize, numClasses] = input.shape.length === 2
    ? input.shape as [number, number]
    : [1, input.shape[0]!];

  // Compute log-softmax for numerical stability
  const logProbs = input.logSoftmax();

  // Gather the log probabilities of the target classes
  const lossData = new Float32Array(batchSize);
  for (let i = 0; i < batchSize; i++) {
    const targetClass = target[i]!;
    lossData[i] = -logProbs.data.data[i * numClasses + targetClass]!;
  }

  // Create loss variable with gradient tracking
  let lossValue: number;
  if (reduction === 'mean') {
    lossValue = lossData.reduce((a, b) => a + b, 0) / batchSize;
  } else if (reduction === 'sum') {
    lossValue = lossData.reduce((a, b) => a + b, 0);
  } else {
    // Return unreduced loss
    const lossTensor = new Tensor(lossData, [batchSize]);
    return new Variable(lossTensor, {
      requiresGrad: input.requiresGrad,
      gradNode: input.requiresGrad
        ? {
            inputs: [logProbs],
            backward: (grad: Tensor) => {
              const gradData = new Float32Array(input.size);
              const softmax = new Float32Array(input.size);

              // Compute softmax from log-softmax
              for (let i = 0; i < input.size; i++) {
                softmax[i] = Math.exp(logProbs.data.data[i]!);
              }

              // Gradient: softmax - one_hot(target)
              for (let i = 0; i < batchSize; i++) {
                for (let j = 0; j < numClasses; j++) {
                  const idx = i * numClasses + j;
                  gradData[idx] = grad.data[i]! * softmax[idx]!;
                  if (j === target[i]) {
                    gradData[idx] -= grad.data[i]!;
                  }
                }
              }

              return [new Tensor(gradData, input.shape)];
            },
            name: 'cross_entropy_none',
          }
        : null,
    });
  }

  const lossTensor = new Tensor(new Float32Array([lossValue]), [1]);

  return new Variable(lossTensor, {
    requiresGrad: input.requiresGrad,
    gradNode: input.requiresGrad
      ? {
          inputs: [logProbs],
          backward: (_grad: Tensor) => {
            const gradData = new Float32Array(input.size);
            const softmax = new Float32Array(input.size);

            // Compute softmax from log-softmax
            for (let i = 0; i < input.size; i++) {
              softmax[i] = Math.exp(logProbs.data.data[i]!);
            }

            // Gradient: (softmax - one_hot(target)) / batch_size (for mean)
            const scale = reduction === 'mean' ? 1 / batchSize : 1;
            for (let i = 0; i < batchSize; i++) {
              for (let j = 0; j < numClasses; j++) {
                const idx = i * numClasses + j;
                gradData[idx] = scale * softmax[idx]!;
                if (j === target[i]) {
                  gradData[idx] -= scale;
                }
              }
            }

            return [new Tensor(gradData, input.shape)];
          },
          name: 'cross_entropy',
        }
      : null,
  });
}

/**
 * Mean Squared Error Loss
 *
 * For regression tasks.
 * loss = mean((input - target)^2)
 */
export function mseLoss(
  input: Variable,
  target: Variable,
  options: { reduction?: 'mean' | 'sum' | 'none' } = {}
): Variable {
  const reduction = options.reduction ?? 'mean';

  // Compute squared differences
  const diff = input.sub(target);
  const squared = diff.mul(diff);

  if (reduction === 'mean') {
    return squared.mean();
  } else if (reduction === 'sum') {
    return squared.sum();
  } else {
    return squared;
  }
}

/**
 * Binary Cross-Entropy Loss
 *
 * For binary classification with sigmoid output.
 * loss = -mean(target * log(input) + (1 - target) * log(1 - input))
 */
export function binaryCrossEntropyLoss(
  input: Variable,
  target: Variable,
  options: { reduction?: 'mean' | 'sum' | 'none'; eps?: number } = {}
): Variable {
  const reduction = options.reduction ?? 'mean';
  const eps = options.eps ?? 1e-7;

  // Clamp input for numerical stability
  const clampedData = new Float32Array(input.size);
  for (let i = 0; i < input.size; i++) {
    clampedData[i] = Math.max(eps, Math.min(1 - eps, input.data.data[i]!));
  }
  const clamped = new Tensor(clampedData, input.shape);

  // Compute BCE: -target * log(input) - (1 - target) * log(1 - input)
  const lossData = new Float32Array(input.size);
  for (let i = 0; i < input.size; i++) {
    const p = clampedData[i]!;
    const t = target.data.data[i]!;
    lossData[i] = -t * Math.log(p) - (1 - t) * Math.log(1 - p);
  }

  let lossValue: number;
  if (reduction === 'mean') {
    lossValue = lossData.reduce((a, b) => a + b, 0) / input.size;
  } else if (reduction === 'sum') {
    lossValue = lossData.reduce((a, b) => a + b, 0);
  } else {
    return new Variable(new Tensor(lossData, input.shape), {
      requiresGrad: input.requiresGrad,
      gradNode: input.requiresGrad
        ? {
            inputs: [input],
            backward: (grad: Tensor) => {
              const gradData = new Float32Array(input.size);
              for (let i = 0; i < input.size; i++) {
                const p = clampedData[i]!;
                const t = target.data.data[i]!;
                gradData[i] = grad.data[i]! * (-t / p + (1 - t) / (1 - p));
              }
              return [new Tensor(gradData, input.shape)];
            },
            name: 'bce_none',
          }
        : null,
    });
  }

  const lossTensor = new Tensor(new Float32Array([lossValue]), [1]);
  const scale = reduction === 'mean' ? 1 / input.size : 1;

  return new Variable(lossTensor, {
    requiresGrad: input.requiresGrad,
    gradNode: input.requiresGrad
      ? {
          inputs: [input],
          backward: (_grad: Tensor) => {
            const gradData = new Float32Array(input.size);
            for (let i = 0; i < input.size; i++) {
              const p = clampedData[i]!;
              const t = target.data.data[i]!;
              gradData[i] = scale * (-t / p + (1 - t) / (1 - p));
            }
            return [new Tensor(gradData, input.shape)];
          },
          name: 'bce',
        }
      : null,
  });
}

/**
 * Negative Log Likelihood Loss
 *
 * Use after log_softmax for classification.
 */
export function nllLoss(
  input: Variable,
  target: Uint32Array | number[],
  options: { reduction?: 'mean' | 'sum' | 'none' } = {}
): Variable {
  const reduction = options.reduction ?? 'mean';
  const [batchSize, numClasses] = input.shape.length === 2
    ? input.shape as [number, number]
    : [1, input.shape[0]!];

  // Gather the log probabilities of the target classes
  const lossData = new Float32Array(batchSize);
  for (let i = 0; i < batchSize; i++) {
    const targetClass = target[i]!;
    lossData[i] = -input.data.data[i * numClasses + targetClass]!;
  }

  let lossValue: number;
  if (reduction === 'mean') {
    lossValue = lossData.reduce((a, b) => a + b, 0) / batchSize;
  } else if (reduction === 'sum') {
    lossValue = lossData.reduce((a, b) => a + b, 0);
  } else {
    return new Variable(new Tensor(lossData, [batchSize]), {
      requiresGrad: input.requiresGrad,
      gradNode: input.requiresGrad
        ? {
            inputs: [input],
            backward: (grad: Tensor) => {
              const gradData = new Float32Array(input.size);
              for (let i = 0; i < batchSize; i++) {
                const targetClass = target[i]!;
                gradData[i * numClasses + targetClass] = -grad.data[i]!;
              }
              return [new Tensor(gradData, input.shape)];
            },
            name: 'nll_none',
          }
        : null,
    });
  }

  const lossTensor = new Tensor(new Float32Array([lossValue]), [1]);
  const scale = reduction === 'mean' ? 1 / batchSize : 1;

  return new Variable(lossTensor, {
    requiresGrad: input.requiresGrad,
    gradNode: input.requiresGrad
      ? {
          inputs: [input],
          backward: (_grad: Tensor) => {
            const gradData = new Float32Array(input.size);
            for (let i = 0; i < batchSize; i++) {
              const targetClass = target[i]!;
              gradData[i * numClasses + targetClass] = -scale;
            }
            return [new Tensor(gradData, input.shape)];
          },
          name: 'nll',
        }
      : null,
  });
}

/**
 * L1 Loss (Mean Absolute Error)
 */
export function l1Loss(
  input: Variable,
  target: Variable,
  options: { reduction?: 'mean' | 'sum' | 'none' } = {}
): Variable {
  const reduction = options.reduction ?? 'mean';

  const diffData = new Float32Array(input.size);
  const absData = new Float32Array(input.size);
  for (let i = 0; i < input.size; i++) {
    diffData[i] = input.data.data[i]! - target.data.data[i]!;
    absData[i] = Math.abs(diffData[i]!);
  }

  let lossValue: number;
  if (reduction === 'mean') {
    lossValue = absData.reduce((a, b) => a + b, 0) / input.size;
  } else if (reduction === 'sum') {
    lossValue = absData.reduce((a, b) => a + b, 0);
  } else {
    return new Variable(new Tensor(absData, input.shape), {
      requiresGrad: input.requiresGrad,
      gradNode: input.requiresGrad
        ? {
            inputs: [input],
            backward: (grad: Tensor) => {
              const gradData = new Float32Array(input.size);
              for (let i = 0; i < input.size; i++) {
                gradData[i] = grad.data[i]! * Math.sign(diffData[i]!);
              }
              return [new Tensor(gradData, input.shape)];
            },
            name: 'l1_none',
          }
        : null,
    });
  }

  const lossTensor = new Tensor(new Float32Array([lossValue]), [1]);
  const scale = reduction === 'mean' ? 1 / input.size : 1;

  return new Variable(lossTensor, {
    requiresGrad: input.requiresGrad,
    gradNode: input.requiresGrad
      ? {
          inputs: [input],
          backward: (_grad: Tensor) => {
            const gradData = new Float32Array(input.size);
            for (let i = 0; i < input.size; i++) {
              gradData[i] = scale * Math.sign(diffData[i]!);
            }
            return [new Tensor(gradData, input.shape)];
          },
          name: 'l1',
        }
      : null,
  });
}

/**
 * Smooth L1 Loss (Huber Loss)
 *
 * Less sensitive to outliers than MSE.
 */
export function smoothL1Loss(
  input: Variable,
  target: Variable,
  options: { reduction?: 'mean' | 'sum' | 'none'; beta?: number } = {}
): Variable {
  const reduction = options.reduction ?? 'mean';
  const beta = options.beta ?? 1.0;

  const lossData = new Float32Array(input.size);
  const diffData = new Float32Array(input.size);

  for (let i = 0; i < input.size; i++) {
    const diff = input.data.data[i]! - target.data.data[i]!;
    diffData[i] = diff;
    const absDiff = Math.abs(diff);
    if (absDiff < beta) {
      lossData[i] = 0.5 * diff * diff / beta;
    } else {
      lossData[i] = absDiff - 0.5 * beta;
    }
  }

  let lossValue: number;
  if (reduction === 'mean') {
    lossValue = lossData.reduce((a, b) => a + b, 0) / input.size;
  } else if (reduction === 'sum') {
    lossValue = lossData.reduce((a, b) => a + b, 0);
  } else {
    return new Variable(new Tensor(lossData, input.shape), {
      requiresGrad: input.requiresGrad,
      gradNode: input.requiresGrad
        ? {
            inputs: [input],
            backward: (grad: Tensor) => {
              const gradData = new Float32Array(input.size);
              for (let i = 0; i < input.size; i++) {
                const diff = diffData[i]!;
                const absDiff = Math.abs(diff);
                if (absDiff < beta) {
                  gradData[i] = grad.data[i]! * diff / beta;
                } else {
                  gradData[i] = grad.data[i]! * Math.sign(diff);
                }
              }
              return [new Tensor(gradData, input.shape)];
            },
            name: 'smooth_l1_none',
          }
        : null,
    });
  }

  const lossTensor = new Tensor(new Float32Array([lossValue]), [1]);
  const scale = reduction === 'mean' ? 1 / input.size : 1;

  return new Variable(lossTensor, {
    requiresGrad: input.requiresGrad,
    gradNode: input.requiresGrad
      ? {
          inputs: [input],
          backward: (_grad: Tensor) => {
            const gradData = new Float32Array(input.size);
            for (let i = 0; i < input.size; i++) {
              const diff = diffData[i]!;
              const absDiff = Math.abs(diff);
              if (absDiff < beta) {
                gradData[i] = scale * diff / beta;
              } else {
                gradData[i] = scale * Math.sign(diff);
              }
            }
            return [new Tensor(gradData, input.shape)];
          },
          name: 'smooth_l1',
        }
      : null,
  });
}
