/**
 * BrowserGNN by Dr. Lee
 * GraphSAGE Layer
 *
 * Implementation of GraphSAGE from:
 * "Inductive Representation Learning on Large Graphs"
 * by Hamilton et al. (2017)
 *
 * Supports multiple aggregation methods:
 * - mean: Average neighbor features
 * - max: Element-wise max of neighbor features
 * - sum: Sum neighbor features
 * - lstm: LSTM-based aggregation (sequential)
 */

import { Tensor } from '../core/tensor';
import { GraphData } from '../core/graph';
import { MessagePassing } from '../core/sparse';
import { GPUSparse, GPUTensor, isGPUAvailable } from '../core/gpu-sparse';
import { Module } from '../nn/module';

export type SAGEAggregator = 'mean' | 'max' | 'sum' | 'pool';

export interface SAGEConvConfig {
  inChannels: number;
  outChannels: number;
  aggregator?: SAGEAggregator;
  normalize?: boolean;
  rootWeight?: boolean;
  bias?: boolean;
  projectNeighbors?: boolean;
}

/**
 * GraphSAGE Convolution Layer
 *
 * Aggregates neighbor features and combines with self features:
 * h_i' = W · [h_i || AGG({h_j : j ∈ N(i)})]
 *
 * Or with root_weight=false:
 * h_i' = W · AGG({h_j : j ∈ N(i)})
 */
export class SAGEConv extends Module {
  readonly inChannels: number;
  readonly outChannels: number;
  readonly aggregator: SAGEAggregator;
  readonly normalize: boolean;
  readonly rootWeight: boolean;
  readonly useBias: boolean;
  readonly projectNeighbors: boolean;

  constructor(config: SAGEConvConfig) {
    super();

    this.inChannels = config.inChannels;
    this.outChannels = config.outChannels;
    this.aggregator = config.aggregator ?? 'mean';
    this.normalize = config.normalize ?? false;
    this.rootWeight = config.rootWeight ?? true;
    this.useBias = config.bias ?? true;
    this.projectNeighbors = config.projectNeighbors ?? false;

    // Linear transformation for neighbors
    const linNeighborIn = this.projectNeighbors ? this.inChannels : this.inChannels;
    const stdv = Math.sqrt(2.0 / (linNeighborIn + this.outChannels));

    const linNeighborData = new Float32Array(linNeighborIn * this.outChannels);
    for (let i = 0; i < linNeighborData.length; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      linNeighborData[i] = stdv * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    const linNeighbor = new Tensor(linNeighborData, [linNeighborIn, this.outChannels]);
    this.registerParameter('lin_neighbor', linNeighbor);

    // Linear transformation for self (root node)
    if (this.rootWeight) {
      const linSelfData = new Float32Array(this.inChannels * this.outChannels);
      for (let i = 0; i < linSelfData.length; i++) {
        const u1 = Math.random();
        const u2 = Math.random();
        linSelfData[i] = stdv * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      const linSelf = new Tensor(linSelfData, [this.inChannels, this.outChannels]);
      this.registerParameter('lin_self', linSelf);
    }

    // Bias
    if (this.useBias) {
      const bias = Tensor.zeros([this.outChannels]);
      this.registerParameter('bias', bias);
    }

    // Pool aggregator: additional MLP for neighbor projection
    if (this.aggregator === 'pool') {
      const poolData = new Float32Array(this.inChannels * this.inChannels);
      for (let i = 0; i < poolData.length; i++) {
        const u1 = Math.random();
        const u2 = Math.random();
        poolData[i] = stdv * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      const poolWeight = new Tensor(poolData, [this.inChannels, this.inChannels]);
      this.registerParameter('pool_weight', poolWeight);

      const poolBias = Tensor.zeros([this.inChannels]);
      this.registerParameter('pool_bias', poolBias);
    }
  }

  /**
   * Forward pass
   */
  forward(input: GraphData): GraphData {
    const { x, edgeIndex, numNodes, numEdges } = input;

    if (x.shape[1] !== this.inChannels) {
      throw new Error(`SAGEConv: Expected ${this.inChannels} input channels, got ${x.shape[1]}`);
    }

    const srcNodes = edgeIndex.slice(0, numEdges);
    const dstNodes = edgeIndex.slice(numEdges);

    // Step 1: Gather source node features for each edge
    const srcFeatures = MessagePassing.gather(x, srcNodes);

    // Step 2: Apply aggregator-specific transformation
    let aggregatedFeatures: Tensor;

    switch (this.aggregator) {
      case 'mean':
        aggregatedFeatures = MessagePassing.scatterMean(srcFeatures, dstNodes, numNodes);
        break;

      case 'max':
        aggregatedFeatures = MessagePassing.scatterMax(srcFeatures, dstNodes, numNodes);
        break;

      case 'sum':
        aggregatedFeatures = MessagePassing.scatterAdd(srcFeatures, dstNodes, numNodes);
        break;

      case 'pool': {
        // Apply MLP to neighbors before max pooling
        const poolWeight = this._parameters.get('pool_weight')!.tensor;
        const poolBias = this._parameters.get('pool_bias')!.tensor;

        // Project: srcFeatures @ poolWeight + poolBias
        const projected = new Float32Array(srcFeatures.size);
        for (let i = 0; i < numEdges; i++) {
          for (let j = 0; j < this.inChannels; j++) {
            let sum = poolBias.data[j]!;
            for (let k = 0; k < this.inChannels; k++) {
              sum += srcFeatures.data[i * this.inChannels + k]! * poolWeight.data[k * this.inChannels + j]!;
            }
            // ReLU activation
            projected[i * this.inChannels + j] = Math.max(0, sum);
          }
        }
        const projectedTensor = new Tensor(projected, srcFeatures.shape);

        // Max pooling
        aggregatedFeatures = MessagePassing.scatterMax(projectedTensor, dstNodes, numNodes);
        break;
      }

      default:
        throw new Error(`Unknown aggregator: ${this.aggregator}`);
    }

    // Step 3: Transform aggregated features
    const linNeighbor = this._parameters.get('lin_neighbor')!.tensor;
    const neighborTransformed = aggregatedFeatures.matmul(linNeighbor);

    // Step 4: Combine with self features (if rootWeight)
    let output: Tensor;

    if (this.rootWeight) {
      const linSelf = this._parameters.get('lin_self')!.tensor;
      const selfTransformed = x.matmul(linSelf);

      // Add self and neighbor transformations
      output = selfTransformed.add(neighborTransformed);
    } else {
      output = neighborTransformed;
    }

    // Step 5: Add bias
    if (this.useBias) {
      const bias = this._parameters.get('bias')!.tensor;
      for (let i = 0; i < numNodes; i++) {
        for (let f = 0; f < this.outChannels; f++) {
          const idx = i * this.outChannels + f;
          output.data[idx] = (output.data[idx] ?? 0) + bias.data[f]!;
        }
      }
    }

    // Step 6: L2 normalize output (optional)
    if (this.normalize) {
      for (let i = 0; i < numNodes; i++) {
        let norm = 0;
        for (let f = 0; f < this.outChannels; f++) {
          const val = output.data[i * this.outChannels + f]!;
          norm += val * val;
        }
        norm = Math.sqrt(norm);
        if (norm > 0) {
          for (let f = 0; f < this.outChannels; f++) {
            const idx = i * this.outChannels + f;
            output.data[idx] = (output.data[idx] ?? 0) / norm;
          }
        }
      }
    }

    return input.withFeatures(output);
  }

  /**
   * Async forward pass with GPU acceleration
   *
   * Uses WebGPU compute shaders when available.
   */
  async forwardAsync(input: GraphData): Promise<GraphData> {
    if (!isGPUAvailable()) {
      return this.forward(input);
    }

    const { x, edgeIndex, numNodes, numEdges } = input;

    if (x.shape[1] !== this.inChannels) {
      throw new Error(`SAGEConv: Expected ${this.inChannels} input channels, got ${x.shape[1]}`);
    }

    const srcNodes = edgeIndex.slice(0, numEdges);
    const dstNodes = edgeIndex.slice(numEdges);

    // Step 1: GPU-accelerated gather
    const srcFeatures = await GPUSparse.gather(x, srcNodes);

    // Step 2: Apply aggregator
    let aggregatedFeatures: Tensor;

    switch (this.aggregator) {
      case 'mean':
        aggregatedFeatures = await GPUSparse.scatterMean(srcFeatures, dstNodes, numNodes);
        break;

      case 'max':
        aggregatedFeatures = await GPUSparse.scatterMax(srcFeatures, dstNodes, numNodes);
        break;

      case 'sum':
        aggregatedFeatures = await GPUSparse.scatterAdd(srcFeatures, dstNodes, numNodes);
        break;

      case 'pool': {
        // Pool aggregator uses CPU for MLP (typically small operation)
        const poolWeight = this._parameters.get('pool_weight')!.tensor;
        const poolBias = this._parameters.get('pool_bias')!.tensor;

        const projected = new Float32Array(srcFeatures.size);
        for (let i = 0; i < numEdges; i++) {
          for (let j = 0; j < this.inChannels; j++) {
            let sum = poolBias.data[j]!;
            for (let k = 0; k < this.inChannels; k++) {
              sum += srcFeatures.data[i * this.inChannels + k]! * poolWeight.data[k * this.inChannels + j]!;
            }
            projected[i * this.inChannels + j] = Math.max(0, sum);
          }
        }
        const projectedTensor = new Tensor(projected, srcFeatures.shape);
        aggregatedFeatures = await GPUSparse.scatterMax(projectedTensor, dstNodes, numNodes);
        break;
      }

      default:
        throw new Error(`Unknown aggregator: ${this.aggregator}`);
    }

    // Step 3: GPU-accelerated matrix multiplications
    const linNeighbor = this._parameters.get('lin_neighbor')!.tensor;
    const neighborTransformed = await GPUTensor.matmul(aggregatedFeatures, linNeighbor);

    // Step 4: Combine with self features
    let output: Tensor;

    if (this.rootWeight) {
      const linSelf = this._parameters.get('lin_self')!.tensor;
      const selfTransformed = await GPUTensor.matmul(x, linSelf);
      output = await GPUTensor.add(selfTransformed, neighborTransformed);
    } else {
      output = neighborTransformed;
    }

    // Step 5: Add bias
    if (this.useBias) {
      const bias = this._parameters.get('bias')!.tensor;
      for (let i = 0; i < numNodes; i++) {
        for (let f = 0; f < this.outChannels; f++) {
          const idx = i * this.outChannels + f;
          output.data[idx] = (output.data[idx] ?? 0) + bias.data[f]!;
        }
      }
    }

    // Step 6: L2 normalize (CPU - typically fast)
    if (this.normalize) {
      for (let i = 0; i < numNodes; i++) {
        let norm = 0;
        for (let f = 0; f < this.outChannels; f++) {
          const val = output.data[i * this.outChannels + f]!;
          norm += val * val;
        }
        norm = Math.sqrt(norm);
        if (norm > 0) {
          for (let f = 0; f < this.outChannels; f++) {
            const idx = i * this.outChannels + f;
            output.data[idx] = (output.data[idx] ?? 0) / norm;
          }
        }
      }
    }

    return input.withFeatures(output);
  }

  /**
   * Reset parameters
   */
  resetParameters(): void {
    const stdv = Math.sqrt(2.0 / (this.inChannels + this.outChannels));

    const linNeighbor = this._parameters.get('lin_neighbor')!.tensor;
    for (let i = 0; i < linNeighbor.size; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      linNeighbor.data[i] = stdv * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    if (this.rootWeight) {
      const linSelf = this._parameters.get('lin_self')!.tensor;
      for (let i = 0; i < linSelf.size; i++) {
        const u1 = Math.random();
        const u2 = Math.random();
        linSelf.data[i] = stdv * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
    }

    if (this.useBias) {
      const bias = this._parameters.get('bias')!.tensor;
      bias.data.fill(0);
    }

    if (this.aggregator === 'pool') {
      const poolWeight = this._parameters.get('pool_weight')!.tensor;
      for (let i = 0; i < poolWeight.size; i++) {
        const u1 = Math.random();
        const u2 = Math.random();
        poolWeight.data[i] = stdv * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      const poolBias = this._parameters.get('pool_bias')!.tensor;
      poolBias.data.fill(0);
    }
  }

  /**
   * Get layer description
   */
  toString(): string {
    return `SAGEConv(${this.inChannels} -> ${this.outChannels}, aggregator=${this.aggregator})`;
  }
}
