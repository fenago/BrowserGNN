/**
 * BrowserGNN by Dr. Lee
 * Graph Attention Network (GAT) Layer
 *
 * Implementation of GAT layer from:
 * "Graph Attention Networks" by Velickovic et al. (2018)
 *
 * Uses attention mechanism to weight neighbor contributions:
 * α_ij = softmax_j(LeakyReLU(a^T [Wh_i || Wh_j]))
 * h_i' = σ(Σ_j α_ij W h_j)
 */

import { Tensor } from '../core/tensor';
import { GraphData } from '../core/graph';
import { MessagePassing } from '../core/sparse';
import { Module } from '../nn/module';

export interface GATConvConfig {
  inChannels: number;
  outChannels: number;
  heads?: number;
  concat?: boolean;
  negativeSlope?: number;
  dropout?: number;
  addSelfLoops?: boolean;
  bias?: boolean;
}

/**
 * Graph Attention Layer
 *
 * Multi-head attention mechanism for graph neural networks.
 * Each head learns different attention patterns.
 */
export class GATConv extends Module {
  readonly inChannels: number;
  readonly outChannels: number;
  readonly heads: number;
  readonly concat: boolean;
  readonly negativeSlope: number;
  readonly dropout: number;
  readonly addSelfLoops: boolean;
  readonly useBias: boolean;

  constructor(config: GATConvConfig) {
    super();

    this.inChannels = config.inChannels;
    this.outChannels = config.outChannels;
    this.heads = config.heads ?? 1;
    this.concat = config.concat ?? true;
    this.negativeSlope = config.negativeSlope ?? 0.2;
    this.dropout = config.dropout ?? 0;
    this.addSelfLoops = config.addSelfLoops ?? true;
    this.useBias = config.bias ?? true;

    // Initialize weight matrix: [inChannels, heads * outChannels]
    const stdv = Math.sqrt(2.0 / (this.inChannels + this.outChannels));
    const weightData = new Float32Array(this.inChannels * this.heads * this.outChannels);
    for (let i = 0; i < weightData.length; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      weightData[i] = stdv * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    const weight = new Tensor(weightData, [this.inChannels, this.heads * this.outChannels]);
    this.registerParameter('weight', weight);

    // Attention parameters: [heads, 2 * outChannels]
    // Split into attention_src and attention_dst for efficiency
    const attData = new Float32Array(this.heads * 2 * this.outChannels);
    const attStdv = Math.sqrt(2.0 / (2 * this.outChannels));
    for (let i = 0; i < attData.length; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      attData[i] = attStdv * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    const attSrc = new Tensor(attData.slice(0, this.heads * this.outChannels), [
      this.heads,
      this.outChannels,
    ]);
    const attDst = new Tensor(attData.slice(this.heads * this.outChannels), [
      this.heads,
      this.outChannels,
    ]);
    this.registerParameter('att_src', attSrc);
    this.registerParameter('att_dst', attDst);

    // Bias
    if (this.useBias) {
      const biasSize = this.concat ? this.heads * this.outChannels : this.outChannels;
      const bias = Tensor.zeros([biasSize]);
      this.registerParameter('bias', bias);
    }
  }

  /**
   * Get output dimension
   */
  get outputDim(): number {
    return this.concat ? this.heads * this.outChannels : this.outChannels;
  }

  /**
   * Forward pass
   */
  forward(input: GraphData): GraphData {
    const { x, edgeIndex, numNodes } = input;

    if (x.shape[1] !== this.inChannels) {
      throw new Error(`GATConv: Expected ${this.inChannels} input channels, got ${x.shape[1]}`);
    }

    // Add self-loops if needed
    let processedGraph = input;
    if (this.addSelfLoops && !input.hasSelfLoops()) {
      processedGraph = input.addSelfLoops();
    }

    const weight = this._parameters.get('weight')!.tensor;
    const attSrc = this._parameters.get('att_src')!.tensor;
    const attDst = this._parameters.get('att_dst')!.tensor;

    const numEdges = processedGraph.numEdges;
    const srcNodes = processedGraph.edgeIndex.slice(0, numEdges);
    const dstNodes = processedGraph.edgeIndex.slice(numEdges);

    // Step 1: Linear transformation
    // H = X @ W, shape: [numNodes, heads * outChannels]
    const H = x.matmul(weight);

    // Reshape to [numNodes, heads, outChannels] conceptually
    // We'll work with the flat representation for efficiency

    // Step 2: Compute attention scores for each edge and head
    const alpha = new Float32Array(numEdges * this.heads);

    for (let e = 0; e < numEdges; e++) {
      const src = srcNodes[e]!;
      const dst = dstNodes[e]!;

      for (let h = 0; h < this.heads; h++) {
        // Get transformed features for this head
        const headOffset = h * this.outChannels;

        // Compute attention: a_src^T @ h_src + a_dst^T @ h_dst
        let score = 0;

        for (let f = 0; f < this.outChannels; f++) {
          const hSrc = H.data[src * this.heads * this.outChannels + headOffset + f]!;
          const hDst = H.data[dst * this.heads * this.outChannels + headOffset + f]!;
          score +=
            attSrc.data[h * this.outChannels + f]! * hSrc +
            attDst.data[h * this.outChannels + f]! * hDst;
        }

        // Apply LeakyReLU
        if (score < 0) {
          score *= this.negativeSlope;
        }

        alpha[e * this.heads + h] = score;
      }
    }

    // Step 3: Softmax attention per target node
    // Group edges by target node and apply softmax
    const alphaExp = new Float32Array(alpha.length);
    const alphaSum = new Float32Array(numNodes * this.heads);

    // Compute exp and sum per target node
    for (let e = 0; e < numEdges; e++) {
      const dst = dstNodes[e]!;
      for (let h = 0; h < this.heads; h++) {
        const expVal = Math.exp(alpha[e * this.heads + h]! - 10); // Subtract max for stability
        alphaExp[e * this.heads + h] = expVal;
        const sumIdx = dst * this.heads + h;
        alphaSum[sumIdx] = (alphaSum[sumIdx] ?? 0) + expVal;
      }
    }

    // Normalize
    for (let e = 0; e < numEdges; e++) {
      const dst = dstNodes[e]!;
      for (let h = 0; h < this.heads; h++) {
        const sum = alphaSum[dst * this.heads + h]!;
        if (sum > 0) {
          const expIdx = e * this.heads + h;
          alphaExp[expIdx] = (alphaExp[expIdx] ?? 0) / sum;
        }
      }
    }

    // Apply dropout to attention (during training)
    if (this._training && this.dropout > 0) {
      for (let i = 0; i < alphaExp.length; i++) {
        if (Math.random() < this.dropout) {
          alphaExp[i] = 0;
        } else {
          alphaExp[i] = (alphaExp[i] ?? 0) / (1 - this.dropout);
        }
      }
    }

    // Step 4: Weighted aggregation
    const outputDim = this.concat ? this.heads * this.outChannels : this.outChannels;
    const output = new Float32Array(numNodes * outputDim);

    if (this.concat) {
      // Concatenate heads: [numNodes, heads * outChannels]
      for (let e = 0; e < numEdges; e++) {
        const src = srcNodes[e]!;
        const dst = dstNodes[e]!;

        for (let h = 0; h < this.heads; h++) {
          const attn = alphaExp[e * this.heads + h]!;
          const headOffset = h * this.outChannels;

          for (let f = 0; f < this.outChannels; f++) {
            const srcVal = H.data[src * this.heads * this.outChannels + headOffset + f]!;
            const outIdx = dst * outputDim + headOffset + f;
            output[outIdx] = (output[outIdx] ?? 0) + attn * srcVal;
          }
        }
      }
    } else {
      // Average heads: [numNodes, outChannels]
      for (let e = 0; e < numEdges; e++) {
        const src = srcNodes[e]!;
        const dst = dstNodes[e]!;

        for (let h = 0; h < this.heads; h++) {
          const attn = alphaExp[e * this.heads + h]!;
          const headOffset = h * this.outChannels;

          for (let f = 0; f < this.outChannels; f++) {
            const srcVal = H.data[src * this.heads * this.outChannels + headOffset + f]!;
            const outIdx = dst * this.outChannels + f;
            output[outIdx] = (output[outIdx] ?? 0) + attn * srcVal / this.heads;
          }
        }
      }
    }

    // Step 5: Add bias
    if (this.useBias) {
      const bias = this._parameters.get('bias')!.tensor;
      for (let i = 0; i < numNodes; i++) {
        for (let f = 0; f < outputDim; f++) {
          const outIdx = i * outputDim + f;
          output[outIdx] = (output[outIdx] ?? 0) + bias.data[f]!;
        }
      }
    }

    const outputTensor = new Tensor(output, [numNodes, outputDim]);
    return input.withFeatures(outputTensor);
  }

  /**
   * Reset parameters
   */
  resetParameters(): void {
    const stdv = Math.sqrt(2.0 / (this.inChannels + this.outChannels));

    const weight = this._parameters.get('weight')!.tensor;
    for (let i = 0; i < weight.size; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      weight.data[i] = stdv * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    const attStdv = Math.sqrt(2.0 / (2 * this.outChannels));
    const attSrc = this._parameters.get('att_src')!.tensor;
    const attDst = this._parameters.get('att_dst')!.tensor;

    for (let i = 0; i < attSrc.size; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      attSrc.data[i] = attStdv * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    for (let i = 0; i < attDst.size; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      attDst.data[i] = attStdv * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    if (this.useBias) {
      const bias = this._parameters.get('bias')!.tensor;
      bias.data.fill(0);
    }
  }

  /**
   * Get layer description
   */
  toString(): string {
    return `GATConv(${this.inChannels} -> ${this.outChannels}, heads=${this.heads}, concat=${this.concat})`;
  }
}
