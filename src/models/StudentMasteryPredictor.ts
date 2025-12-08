/**
 * BrowserGNN by Dr. Lee
 * Student Mastery Predictor
 *
 * A GNN model for predicting student mastery levels in a knowledge graph.
 * Designed for integration with learning platforms like LearningScience.ai.
 *
 * Use Cases:
 * - Knowledge tracing: Predict student mastery of concepts
 * - Learning analytics: Identify knowledge gaps
 * - Personalized learning: Recommend focus areas
 *
 * Graph Structure:
 * - Nodes: Concepts/skills in a curriculum
 * - Node Features: Concept embeddings, difficulty, student performance history
 * - Edges: Prerequisite relationships, semantic similarity
 * - Output: Mastery probability for each concept (0-1)
 */

import { Tensor } from '../core/tensor';
import { GraphData } from '../core/graph';
import { Module, Sequential } from '../nn';
import { Linear, ReLU, Sigmoid, Dropout } from '../nn';
import { GCNConv } from '../layers/gcn';
import { GATConv } from '../layers/gat';

export interface StudentMasteryPredictorConfig {
  /**
   * Number of input features per concept node.
   * Includes: concept embedding, difficulty, historical performance, etc.
   */
  inputFeatures: number;

  /**
   * Hidden dimension for GNN layers.
   * Default: 64
   */
  hiddenDim?: number;

  /**
   * Number of GNN layers.
   * Default: 2
   */
  numLayers?: number;

  /**
   * Whether to use attention mechanism (GAT) instead of GCN.
   * Default: false
   */
  useAttention?: boolean;

  /**
   * Number of attention heads (if useAttention=true).
   * Default: 4
   */
  numHeads?: number;

  /**
   * Dropout rate for regularization.
   * Default: 0.1
   */
  dropout?: number;

  /**
   * Whether to use residual connections.
   * Default: true
   */
  residual?: boolean;
}

/**
 * Student Mastery Predictor Model
 *
 * Architecture:
 * 1. Input projection: Linear(inputFeatures -> hiddenDim)
 * 2. GNN layers (GCN or GAT) for message passing
 * 3. Output head: Linear(hiddenDim -> 1) + Sigmoid
 *
 * The model propagates mastery information through the knowledge graph,
 * allowing prerequisites to influence predicted mastery.
 */
export class StudentMasteryPredictor extends Module {
  readonly config: Required<StudentMasteryPredictorConfig>;

  private inputProj: Linear;
  private gnnLayers: Module[];
  private dropout: Dropout;
  private outputHead: Sequential;

  constructor(config: StudentMasteryPredictorConfig) {
    super();

    // Apply defaults
    this.config = {
      inputFeatures: config.inputFeatures,
      hiddenDim: config.hiddenDim ?? 64,
      numLayers: config.numLayers ?? 2,
      useAttention: config.useAttention ?? false,
      numHeads: config.numHeads ?? 4,
      dropout: config.dropout ?? 0.1,
      residual: config.residual ?? true,
    };

    // Input projection
    this.inputProj = new Linear({
      inFeatures: this.config.inputFeatures,
      outFeatures: this.config.hiddenDim,
    });
    this.registerModule('inputProj', this.inputProj);

    // GNN layers
    this.gnnLayers = [];
    for (let i = 0; i < this.config.numLayers; i++) {
      let layer: Module;

      if (this.config.useAttention) {
        // GAT layer
        const outChannels = Math.floor(this.config.hiddenDim / this.config.numHeads);
        layer = new GATConv({
          inChannels: this.config.hiddenDim,
          outChannels,
          heads: this.config.numHeads,
          concat: true, // Concatenate heads
          dropout: this.config.dropout,
        });
      } else {
        // GCN layer
        layer = new GCNConv({
          inChannels: this.config.hiddenDim,
          outChannels: this.config.hiddenDim,
        });
      }

      this.gnnLayers.push(layer);
      this.registerModule(`gnn_${i}`, layer);
    }

    // Dropout
    this.dropout = new Dropout(this.config.dropout);
    this.registerModule('dropout', this.dropout);

    // Output head: Linear -> Sigmoid (mastery probability)
    this.outputHead = new Sequential([
      new Linear({
        inFeatures: this.config.hiddenDim,
        outFeatures: 1,
      }),
      new Sigmoid(),
    ]);
    this.registerModule('outputHead', this.outputHead);
  }

  /**
   * Forward pass to predict mastery levels.
   *
   * @param graph Knowledge graph with concept nodes
   * @returns Tensor of shape [numNodes, 1] with mastery probabilities
   */
  async forward(graph: GraphData): Promise<Tensor> {
    // Input projection
    let h = (await this.inputProj.forward(graph.x)) as Tensor;
    h = new ReLU().forward(h) as Tensor;

    // Create working graph
    let workingGraph = graph.withFeatures(h);

    // GNN layers with optional residual connections
    for (let i = 0; i < this.gnnLayers.length; i++) {
      const layer = this.gnnLayers[i]!;
      const prevH = workingGraph.x;

      // GNN forward
      if (layer instanceof GCNConv || layer instanceof GATConv) {
        workingGraph = layer.forward(workingGraph);
      }

      // Apply activation
      let newH = new ReLU().forward(workingGraph.x) as Tensor;
      newH = this.dropout.forward(newH) as Tensor;

      // Residual connection
      if (this.config.residual && prevH.shape[1] === newH.shape[1]) {
        const residualData = new Float32Array(newH.size);
        for (let j = 0; j < newH.size; j++) {
          residualData[j] = (newH.data[j] ?? 0) + (prevH.data[j] ?? 0);
        }
        newH = new Tensor(residualData, newH.shape);
      }

      workingGraph = workingGraph.withFeatures(newH);
    }

    // Output head
    const mastery = (await this.outputHead.forward(workingGraph.x)) as Tensor;

    return mastery;
  }

  /**
   * Predict mastery for all concepts in the graph.
   *
   * @param graph Knowledge graph
   * @returns Object with mastery scores and predictions
   */
  async predict(graph: GraphData): Promise<{
    mastery: number[];
    predictions: { nodeIndex: number; score: number; mastered: boolean }[];
  }> {
    this.eval(); // Set to evaluation mode
    const output = await this.forward(graph);

    const mastery: number[] = [];
    const predictions: { nodeIndex: number; score: number; mastered: boolean }[] = [];

    for (let i = 0; i < graph.numNodes; i++) {
      const score = output.data[i] ?? 0;
      mastery.push(score);
      predictions.push({
        nodeIndex: i,
        score,
        mastered: score >= 0.7, // Threshold for mastery
      });
    }

    return { mastery, predictions };
  }

  /**
   * Get concepts that need the most attention.
   *
   * @param graph Knowledge graph
   * @param topK Number of concepts to return
   * @returns Indices and scores of concepts needing attention
   */
  async getWeakConcepts(
    graph: GraphData,
    topK: number = 5
  ): Promise<{ nodeIndex: number; score: number }[]> {
    const { predictions } = await this.predict(graph);

    // Sort by score (ascending - lowest mastery first)
    predictions.sort((a, b) => a.score - b.score);

    return predictions.slice(0, topK).map(p => ({
      nodeIndex: p.nodeIndex,
      score: p.score,
    }));
  }

  /**
   * Create a sample knowledge graph for testing.
   * Represents a simple math curriculum.
   */
  static createSampleGraph(numConcepts: number = 10): GraphData {
    // Feature dimensions: [embedding(8), difficulty(1), attempts(1), successRate(1)] = 11
    const featureDim = 11;
    const features = new Float32Array(numConcepts * featureDim);

    // Create random features
    for (let i = 0; i < numConcepts; i++) {
      // Random embedding
      for (let j = 0; j < 8; j++) {
        features[i * featureDim + j] = Math.random() * 2 - 1;
      }
      // Difficulty (0-1)
      features[i * featureDim + 8] = Math.random();
      // Attempts (normalized)
      features[i * featureDim + 9] = Math.random();
      // Success rate (0-1)
      features[i * featureDim + 10] = Math.random();
    }

    // Create prerequisite edges (simple chain + some cross-links)
    const srcEdges: number[] = [];
    const tgtEdges: number[] = [];
    for (let i = 0; i < numConcepts - 1; i++) {
      // Chain: i -> i+1
      srcEdges.push(i);
      tgtEdges.push(i + 1);
      // Add reverse for message passing
      srcEdges.push(i + 1);
      tgtEdges.push(i);
    }

    // Add some cross-links
    for (let i = 0; i < numConcepts; i += 3) {
      if (i + 2 < numConcepts) {
        srcEdges.push(i);
        tgtEdges.push(i + 2);
        srcEdges.push(i + 2);
        tgtEdges.push(i);
      }
    }

    const edgeIndex = new Uint32Array([...srcEdges, ...tgtEdges]);

    return new GraphData({
      x: features,
      numNodes: numConcepts,
      numFeatures: featureDim,
      edgeIndex,
      numEdges: srcEdges.length,
    });
  }

  toString(): string {
    const gnnType = this.config.useAttention ? 'GAT' : 'GCN';
    return `StudentMasteryPredictor(${this.config.inputFeatures} -> ${gnnType}x${this.config.numLayers} -> mastery)`;
  }
}
