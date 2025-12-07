/**
 * BrowserGNN by Dr. Lee
 * Graph Convolutional Network (GCN) Layer
 *
 * Implementation of GCN layer from:
 * "Semi-Supervised Classification with Graph Convolutional Networks"
 * by Kipf & Welling (2017)
 *
 * Formula: H' = σ(D̃^{-1/2} Ã D̃^{-1/2} H W)
 * where Ã = A + I (adjacency with self-loops)
 * and D̃ is the degree matrix of Ã
 */

import { Tensor } from '../core/tensor';
import { GraphData } from '../core/graph';
import { MessagePassing, computeGCNNorm } from '../core/sparse';
import { Module } from '../nn/module';

export interface GCNConvConfig {
  inChannels: number;
  outChannels: number;
  bias?: boolean;
  addSelfLoops?: boolean;
  normalize?: boolean;
}

/**
 * Graph Convolutional Layer
 *
 * Performs message passing with symmetric normalization:
 * x_i' = W · Σ_{j∈N(i)∪{i}} (1/√(d_i·d_j)) · x_j
 */
export class GCNConv extends Module {
  readonly inChannels: number;
  readonly outChannels: number;
  readonly addSelfLoops: boolean;
  readonly normalize: boolean;
  readonly useBias: boolean;

  // Cached normalization coefficients
  private cachedNorm: Float32Array | null = null;
  private cachedGraphId: string | null = null;

  constructor(config: GCNConvConfig) {
    super();

    this.inChannels = config.inChannels;
    this.outChannels = config.outChannels;
    this.addSelfLoops = config.addSelfLoops ?? true;
    this.normalize = config.normalize ?? true;
    this.useBias = config.bias ?? true;

    // Initialize weight matrix using Xavier initialization
    const stdv = Math.sqrt(2.0 / (this.inChannels + this.outChannels));
    const weightData = new Float32Array(this.inChannels * this.outChannels);
    for (let i = 0; i < weightData.length; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      weightData[i] = stdv * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    const weight = new Tensor(weightData, [this.inChannels, this.outChannels]);
    this.registerParameter('weight', weight);

    // Initialize bias
    if (this.useBias) {
      const bias = Tensor.zeros([this.outChannels]);
      this.registerParameter('bias', bias);
    }
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
        `GCNConv: Expected ${this.inChannels} input channels, got ${x.shape[1]}`
      );
    }

    // Add self-loops if needed
    let processedGraph = input;
    if (this.addSelfLoops && !input.hasSelfLoops()) {
      processedGraph = input.addSelfLoops();
    }

    const weight = this._parameters.get('weight')!.tensor;

    // Step 1: Transform features: H_transformed = X @ W
    // X: [numNodes, inChannels], W: [inChannels, outChannels]
    const transformed = x.matmul(weight);

    // Step 2: Message passing with normalization
    let output: Tensor;

    if (this.normalize) {
      // Compute normalization coefficients
      const norm = computeGCNNorm(
        processedGraph.edgeIndex,
        processedGraph.numEdges,
        numNodes,
        false // Already added self-loops above
      );

      // Gather source node features
      const srcFeatures = MessagePassing.gather(
        transformed,
        processedGraph.edgeIndex.slice(0, processedGraph.numEdges)
      );

      // Apply normalization to messages
      const normalizedMessages = new Float32Array(srcFeatures.size);
      for (let i = 0; i < processedGraph.numEdges; i++) {
        for (let f = 0; f < this.outChannels; f++) {
          normalizedMessages[i * this.outChannels + f] =
            srcFeatures.data[i * this.outChannels + f]! * norm[i]!;
        }
      }
      const normalizedTensor = new Tensor(normalizedMessages, srcFeatures.shape);

      // Aggregate to target nodes
      output = MessagePassing.scatterAdd(
        normalizedTensor,
        processedGraph.edgeIndex.slice(processedGraph.numEdges),
        numNodes
      );
    } else {
      // Simple sum aggregation without normalization
      const srcFeatures = MessagePassing.gather(
        transformed,
        processedGraph.edgeIndex.slice(0, processedGraph.numEdges)
      );
      output = MessagePassing.scatterAdd(
        srcFeatures,
        processedGraph.edgeIndex.slice(processedGraph.numEdges),
        numNodes
      );
    }

    // Step 3: Add bias
    if (this.useBias) {
      const bias = this._parameters.get('bias')!.tensor;
      for (let i = 0; i < numNodes; i++) {
        for (let f = 0; f < this.outChannels; f++) {
          const idx = i * this.outChannels + f;
          output.data[idx] = (output.data[idx] ?? 0) + (bias.data[f] ?? 0);
        }
      }
    }

    // Return new GraphData with updated features
    return input.withFeatures(output);
  }

  /**
   * Reset parameters to initial values
   */
  resetParameters(): void {
    const stdv = Math.sqrt(2.0 / (this.inChannels + this.outChannels));
    const weight = this._parameters.get('weight')!.tensor;

    for (let i = 0; i < weight.size; i++) {
      const u1 = Math.random();
      const u2 = Math.random();
      weight.data[i] = stdv * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    if (this.useBias) {
      const bias = this._parameters.get('bias')!.tensor;
      bias.data.fill(0);
    }

    this.cachedNorm = null;
    this.cachedGraphId = null;
  }

  /**
   * Get layer description
   */
  toString(): string {
    return `GCNConv(${this.inChannels} -> ${this.outChannels}, bias=${this.useBias}, normalize=${this.normalize})`;
  }
}
