/**
 * BrowserGNN by Dr. Lee
 * Graph Isomorphism Network (GIN) Layer
 *
 * Implementation of GIN layer from:
 * "How Powerful are Graph Neural Networks?"
 * by Xu et al. (2019)
 *
 * Formula: h_v' = MLP((1 + ε) · h_v + Σ_{u∈N(v)} h_u)
 * where ε is a learnable parameter (or fixed at 0)
 *
 * GIN is one of the most powerful GNN architectures for graph isomorphism testing
 * and is widely used for graph-level tasks like molecule classification.
 */

import { Tensor } from '../core/tensor';
import { GraphData } from '../core/graph';
import { MessagePassing } from '../core/sparse';
import { GPUSparse, GPUTensor, isGPUAvailable } from '../core/gpu-sparse';
import { Module } from '../nn/module';

export interface GINConvConfig {
  inChannels: number;
  outChannels: number;
  hiddenChannels?: number;      // MLP hidden layer size (default: outChannels)
  epsilon?: number;             // Initial epsilon value (default: 0)
  trainEpsilon?: boolean;       // Whether to learn epsilon (default: true)
  bias?: boolean;               // Use bias in MLP layers (default: true)
  numLayers?: number;           // Number of MLP layers (default: 2)
}

/**
 * Graph Isomorphism Network Layer
 *
 * GIN aggregates neighbor features and applies an MLP:
 * x_i' = MLP((1 + ε) · x_i + Σ_{j∈N(i)} x_j)
 *
 * This is proven to be as powerful as the Weisfeiler-Lehman test
 * for distinguishing non-isomorphic graphs.
 */
export class GINConv extends Module {
  readonly inChannels: number;
  readonly outChannels: number;
  readonly hiddenChannels: number;
  readonly trainEpsilon: boolean;
  readonly useBias: boolean;
  readonly numLayers: number;

  constructor(config: GINConvConfig) {
    super();

    this.inChannels = config.inChannels;
    this.outChannels = config.outChannels;
    this.hiddenChannels = config.hiddenChannels ?? config.outChannels;
    this.trainEpsilon = config.trainEpsilon ?? true;
    this.useBias = config.bias ?? true;
    this.numLayers = config.numLayers ?? 2;

    // Initialize epsilon (learnable or fixed)
    const epsilonData = new Float32Array([config.epsilon ?? 0]);
    const epsilon = new Tensor(epsilonData, [1]);
    if (this.trainEpsilon) {
      this.registerParameter('epsilon', epsilon);
    } else {
      // Store as buffer (non-trainable)
      this.registerBuffer('epsilon', epsilon);
    }

    // Initialize MLP weights
    // Layer 1: inChannels -> hiddenChannels
    const stdv1 = Math.sqrt(2.0 / (this.inChannels + this.hiddenChannels));
    const weight1Data = new Float32Array(this.inChannels * this.hiddenChannels);
    for (let i = 0; i < weight1Data.length; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      weight1Data[i] = stdv1 * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    const weight1 = new Tensor(weight1Data, [this.inChannels, this.hiddenChannels]);
    this.registerParameter('mlp_weight1', weight1);

    if (this.useBias) {
      const bias1 = Tensor.zeros([this.hiddenChannels]);
      this.registerParameter('mlp_bias1', bias1);
    }

    // Layer 2: hiddenChannels -> outChannels
    const stdv2 = Math.sqrt(2.0 / (this.hiddenChannels + this.outChannels));
    const weight2Data = new Float32Array(this.hiddenChannels * this.outChannels);
    for (let i = 0; i < weight2Data.length; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      weight2Data[i] = stdv2 * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    const weight2 = new Tensor(weight2Data, [this.hiddenChannels, this.outChannels]);
    this.registerParameter('mlp_weight2', weight2);

    if (this.useBias) {
      const bias2 = Tensor.zeros([this.outChannels]);
      this.registerParameter('mlp_bias2', bias2);
    }
  }

  /**
   * Apply ReLU activation in-place
   */
  private relu(tensor: Tensor): Tensor {
    const result = new Float32Array(tensor.size);
    for (let i = 0; i < tensor.size; i++) {
      result[i] = Math.max(0, tensor.data[i] ?? 0);
    }
    return new Tensor(result, tensor.shape);
  }

  /**
   * Forward pass
   *
   * @param input GraphData with node features
   * @returns GraphData with updated node features
   */
  forward(input: GraphData): GraphData {
    const { x, edgeIndex, numEdges, numNodes } = input;

    // Validate input dimensions
    if (x.shape[1] !== this.inChannels) {
      throw new Error(
        `GINConv: Expected ${this.inChannels} input channels, got ${x.shape[1]}`
      );
    }

    // Step 1: Gather source node features
    const srcFeatures = MessagePassing.gather(
      x,
      edgeIndex.slice(0, numEdges)
    );

    // Step 2: Aggregate neighbor features using sum
    const aggregated = MessagePassing.scatterAdd(
      srcFeatures,
      edgeIndex.slice(numEdges),
      numNodes
    );

    // Step 3: Apply (1 + ε) scaling to self-features and add aggregated neighbors
    // combined = (1 + ε) · x + aggregated
    const epsilon = this.trainEpsilon
      ? this._parameters.get('epsilon')!.tensor
      : this._buffers.get('epsilon')!;
    const eps = epsilon.data[0] ?? 0;

    const combined = new Float32Array(numNodes * this.inChannels);
    for (let i = 0; i < numNodes; i++) {
      for (let f = 0; f < this.inChannels; f++) {
        const idx = i * this.inChannels + f;
        combined[idx] = (1 + eps) * (x.data[idx] ?? 0) + (aggregated.data[idx] ?? 0);
      }
    }
    const combinedTensor = new Tensor(combined, [numNodes, this.inChannels]);

    // Step 4: Apply MLP
    // Layer 1: Linear + ReLU
    const weight1 = this._parameters.get('mlp_weight1')!.tensor;
    let hidden = combinedTensor.matmul(weight1);

    if (this.useBias) {
      const bias1 = this._parameters.get('mlp_bias1')!.tensor;
      for (let i = 0; i < numNodes; i++) {
        for (let f = 0; f < this.hiddenChannels; f++) {
          const idx = i * this.hiddenChannels + f;
          hidden.data[idx] = (hidden.data[idx] ?? 0) + (bias1.data[f] ?? 0);
        }
      }
    }

    hidden = this.relu(hidden);

    // Layer 2: Linear
    const weight2 = this._parameters.get('mlp_weight2')!.tensor;
    let output = hidden.matmul(weight2);

    if (this.useBias) {
      const bias2 = this._parameters.get('mlp_bias2')!.tensor;
      for (let i = 0; i < numNodes; i++) {
        for (let f = 0; f < this.outChannels; f++) {
          const idx = i * this.outChannels + f;
          output.data[idx] = (output.data[idx] ?? 0) + (bias2.data[f] ?? 0);
        }
      }
    }

    // Return new GraphData with updated features
    return input.withFeatures(output);
  }

  /**
   * Async forward pass with GPU acceleration
   *
   * Uses WebGPU compute shaders when available for acceleration.
   * Falls back to CPU otherwise.
   *
   * @param input GraphData with node features
   * @returns Promise<GraphData> with updated node features
   */
  async forwardAsync(input: GraphData): Promise<GraphData> {
    // If GPU not available, use synchronous CPU path
    if (!isGPUAvailable()) {
      return this.forward(input);
    }

    const { x, edgeIndex, numEdges, numNodes } = input;

    // Validate input dimensions
    if (x.shape[1] !== this.inChannels) {
      throw new Error(
        `GINConv: Expected ${this.inChannels} input channels, got ${x.shape[1]}`
      );
    }

    // Step 1: GPU-accelerated gather
    const srcFeatures = await GPUSparse.gather(
      x,
      edgeIndex.slice(0, numEdges)
    );

    // Step 2: GPU-accelerated scatter-add
    const aggregated = await GPUSparse.scatterAdd(
      srcFeatures,
      edgeIndex.slice(numEdges),
      numNodes
    );

    // Step 3: Apply (1 + ε) scaling and combine (CPU - element-wise)
    const epsilon = this.trainEpsilon
      ? this._parameters.get('epsilon')!.tensor
      : this._buffers.get('epsilon')!;
    const eps = epsilon.data[0] ?? 0;

    const combined = new Float32Array(numNodes * this.inChannels);
    for (let i = 0; i < numNodes; i++) {
      for (let f = 0; f < this.inChannels; f++) {
        const idx = i * this.inChannels + f;
        combined[idx] = (1 + eps) * (x.data[idx] ?? 0) + (aggregated.data[idx] ?? 0);
      }
    }
    const combinedTensor = new Tensor(combined, [numNodes, this.inChannels]);

    // Step 4: GPU-accelerated MLP
    const weight1 = this._parameters.get('mlp_weight1')!.tensor;
    let hidden = await GPUTensor.matmul(combinedTensor, weight1);

    if (this.useBias) {
      const bias1 = this._parameters.get('mlp_bias1')!.tensor;
      for (let i = 0; i < numNodes; i++) {
        for (let f = 0; f < this.hiddenChannels; f++) {
          const idx = i * this.hiddenChannels + f;
          hidden.data[idx] = (hidden.data[idx] ?? 0) + (bias1.data[f] ?? 0);
        }
      }
    }

    hidden = this.relu(hidden);

    const weight2 = this._parameters.get('mlp_weight2')!.tensor;
    let output = await GPUTensor.matmul(hidden, weight2);

    if (this.useBias) {
      const bias2 = this._parameters.get('mlp_bias2')!.tensor;
      for (let i = 0; i < numNodes; i++) {
        for (let f = 0; f < this.outChannels; f++) {
          const idx = i * this.outChannels + f;
          output.data[idx] = (output.data[idx] ?? 0) + (bias2.data[f] ?? 0);
        }
      }
    }

    return input.withFeatures(output);
  }

  /**
   * Reset parameters to initial values
   */
  resetParameters(): void {
    // Reset epsilon
    if (this.trainEpsilon) {
      const epsilon = this._parameters.get('epsilon')!.tensor;
      epsilon.data[0] = 0;
    }

    // Reset MLP weights with Xavier initialization
    const stdv1 = Math.sqrt(2.0 / (this.inChannels + this.hiddenChannels));
    const weight1 = this._parameters.get('mlp_weight1')!.tensor;
    for (let i = 0; i < weight1.size; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      weight1.data[i] = stdv1 * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    const stdv2 = Math.sqrt(2.0 / (this.hiddenChannels + this.outChannels));
    const weight2 = this._parameters.get('mlp_weight2')!.tensor;
    for (let i = 0; i < weight2.size; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      weight2.data[i] = stdv2 * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    if (this.useBias) {
      const bias1 = this._parameters.get('mlp_bias1')!.tensor;
      bias1.data.fill(0);
      const bias2 = this._parameters.get('mlp_bias2')!.tensor;
      bias2.data.fill(0);
    }
  }

  /**
   * Get layer description
   */
  toString(): string {
    return `GINConv(${this.inChannels} -> ${this.outChannels}, hidden=${this.hiddenChannels}, trainEps=${this.trainEpsilon})`;
  }
}
