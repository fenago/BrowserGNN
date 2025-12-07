/**
 * BrowserGNN by Dr. Lee
 * Linear Layer
 *
 * Fully connected linear transformation layer.
 */

import { Tensor } from '../core/tensor';
import { Module } from './module';

export interface LinearConfig {
  inFeatures: number;
  outFeatures: number;
  bias?: boolean;
}

/**
 * Linear layer: y = xW^T + b
 */
export class Linear extends Module {
  readonly inFeatures: number;
  readonly outFeatures: number;
  readonly useBias: boolean;

  constructor(config: LinearConfig) {
    super();
    this.inFeatures = config.inFeatures;
    this.outFeatures = config.outFeatures;
    this.useBias = config.bias ?? true;

    // Initialize weights using Xavier/Glorot initialization
    const stdv = Math.sqrt(2.0 / (this.inFeatures + this.outFeatures));
    const weightData = new Float32Array(this.outFeatures * this.inFeatures);
    for (let i = 0; i < weightData.length; i++) {
      // Box-Muller transform for normal distribution
      const u1 = Math.random();
      const u2 = Math.random();
      weightData[i] = stdv * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    const weight = new Tensor(weightData, [this.outFeatures, this.inFeatures]);
    this.registerParameter('weight', weight);

    // Initialize bias to zeros
    if (this.useBias) {
      const bias = Tensor.zeros([this.outFeatures]);
      this.registerParameter('bias', bias);
    }
  }

  forward(input: Tensor): Tensor {
    // Input shape: [batch, inFeatures] or [inFeatures]
    // Output shape: [batch, outFeatures] or [outFeatures]

    const weight = this._parameters.get('weight')!.tensor;

    let output: Tensor;

    if (input.ndim === 1) {
      // Single sample: [inFeatures] -> [outFeatures]
      if (input.shape[0] !== this.inFeatures) {
        throw new Error(`Expected input size ${this.inFeatures}, got ${input.shape[0]}`);
      }

      // y = Wx
      const result = new Float32Array(this.outFeatures);
      for (let i = 0; i < this.outFeatures; i++) {
        let sum = 0;
        for (let j = 0; j < this.inFeatures; j++) {
          sum += weight.data[i * this.inFeatures + j]! * input.data[j]!;
        }
        result[i] = sum;
      }
      output = new Tensor(result, [this.outFeatures]);
    } else if (input.ndim === 2) {
      // Batch: [batch, inFeatures] -> [batch, outFeatures]
      const [batch, inFeat] = input.shape;
      if (inFeat !== this.inFeatures) {
        throw new Error(`Expected input size ${this.inFeatures}, got ${inFeat}`);
      }

      // Y = XW^T
      const result = new Float32Array(batch! * this.outFeatures);
      for (let b = 0; b < batch!; b++) {
        for (let i = 0; i < this.outFeatures; i++) {
          let sum = 0;
          for (let j = 0; j < this.inFeatures; j++) {
            sum += input.data[b * this.inFeatures + j]! * weight.data[i * this.inFeatures + j]!;
          }
          result[b * this.outFeatures + i] = sum;
        }
      }
      output = new Tensor(result, [batch!, this.outFeatures]);
    } else {
      throw new Error(`Linear layer expects 1D or 2D input, got ${input.ndim}D`);
    }

    // Add bias
    if (this.useBias) {
      const bias = this._parameters.get('bias')!.tensor;
      if (output.ndim === 1) {
        output = output.add(bias);
      } else {
        // Broadcast bias across batch
        const [batch] = output.shape;
        for (let b = 0; b < batch!; b++) {
          for (let i = 0; i < this.outFeatures; i++) {
            output.data[b * this.outFeatures + i] = (output.data[b * this.outFeatures + i] ?? 0) + (bias.data[i] ?? 0);
          }
        }
      }
    }

    return output;
  }
}
