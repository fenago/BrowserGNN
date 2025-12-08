/**
 * BrowserGNN by Dr. Lee
 * Learning Path Recommender
 *
 * A GNN model for recommending optimal learning paths through a curriculum.
 * Uses graph structure to find the best sequence of concepts to learn.
 *
 * Use Cases:
 * - Curriculum sequencing: Determine optimal order of topics
 * - Adaptive learning: Personalize paths based on student state
 * - Gap filling: Identify missing prerequisite knowledge
 *
 * Graph Structure:
 * - Nodes: Concepts/skills in a curriculum
 * - Node Features: Concept embeddings, difficulty, learning time, mastery level
 * - Edges: Prerequisite relationships (directed), semantic similarity
 * - Edge Features: Strength of prerequisite relationship
 * - Output: Priority scores for next concepts to learn
 */

import { Tensor } from '../core/tensor';
import { GraphData } from '../core/graph';
import { Module, Sequential } from '../nn';
import { Linear, ReLU, LeakyReLU, Softmax, Dropout } from '../nn';
import { SAGEConv } from '../layers/sage';
import { GINConv } from '../layers/gin';

export interface LearningPathRecommenderConfig {
  /**
   * Number of input features per concept node.
   */
  inputFeatures: number;

  /**
   * Hidden dimension for GNN layers.
   * Default: 64
   */
  hiddenDim?: number;

  /**
   * Number of GNN layers.
   * Default: 3
   */
  numLayers?: number;

  /**
   * Aggregation method for GraphSAGE.
   * Default: 'mean'
   */
  aggregator?: 'mean' | 'max' | 'sum' | 'pool';

  /**
   * Whether to use GIN layers instead of SAGE.
   * GIN is more expressive for graph isomorphism tasks.
   * Default: false
   */
  useGIN?: boolean;

  /**
   * Dropout rate for regularization.
   * Default: 0.1
   */
  dropout?: number;

  /**
   * Whether to normalize output scores (softmax).
   * Default: true
   */
  normalizeScores?: boolean;
}

/**
 * Learning Path Recommender Model
 *
 * Architecture:
 * 1. Input projection with LeakyReLU
 * 2. GraphSAGE or GIN layers for neighborhood aggregation
 * 3. MLP scoring head
 * 4. Optional softmax normalization
 *
 * The model learns to score concepts based on:
 * - Current mastery levels of prerequisites
 * - Structural position in the knowledge graph
 * - Concept difficulty and learning time
 */
export class LearningPathRecommender extends Module {
  readonly config: Required<LearningPathRecommenderConfig>;

  private inputProj: Sequential;
  private gnnLayers: Module[];
  private dropout: Dropout;
  private scoringHead: Sequential;
  private softmax: Softmax;

  constructor(config: LearningPathRecommenderConfig) {
    super();

    // Apply defaults
    this.config = {
      inputFeatures: config.inputFeatures,
      hiddenDim: config.hiddenDim ?? 64,
      numLayers: config.numLayers ?? 3,
      aggregator: config.aggregator ?? 'mean',
      useGIN: config.useGIN ?? false,
      dropout: config.dropout ?? 0.1,
      normalizeScores: config.normalizeScores ?? true,
    };

    // Input projection
    this.inputProj = new Sequential([
      new Linear({
        inFeatures: this.config.inputFeatures,
        outFeatures: this.config.hiddenDim,
      }),
      new LeakyReLU(0.1),
    ]);
    this.registerModule('inputProj', this.inputProj);

    // GNN layers
    this.gnnLayers = [];
    for (let i = 0; i < this.config.numLayers; i++) {
      let layer: Module;

      if (this.config.useGIN) {
        layer = new GINConv({
          inChannels: this.config.hiddenDim,
          outChannels: this.config.hiddenDim,
          hiddenChannels: this.config.hiddenDim * 2,
          trainEpsilon: true,
        });
      } else {
        layer = new SAGEConv({
          inChannels: this.config.hiddenDim,
          outChannels: this.config.hiddenDim,
          aggregator: this.config.aggregator,
          normalize: true,
        });
      }

      this.gnnLayers.push(layer);
      this.registerModule(`gnn_${i}`, layer);
    }

    // Dropout
    this.dropout = new Dropout(this.config.dropout);
    this.registerModule('dropout', this.dropout);

    // Scoring head: MLP -> single score per node
    this.scoringHead = new Sequential([
      new Linear({
        inFeatures: this.config.hiddenDim,
        outFeatures: this.config.hiddenDim / 2,
      }),
      new ReLU(),
      new Linear({
        inFeatures: Math.floor(this.config.hiddenDim / 2),
        outFeatures: 1,
      }),
    ]);
    this.registerModule('scoringHead', this.scoringHead);

    // Softmax for normalization
    this.softmax = new Softmax();
  }

  /**
   * Forward pass to compute priority scores.
   *
   * @param graph Knowledge graph with concept nodes
   * @param maskMastered Optional boolean array to mask already mastered concepts
   * @returns Tensor of shape [numNodes, 1] with priority scores
   */
  async forward(graph: GraphData, maskMastered?: boolean[]): Promise<Tensor> {
    // Input projection
    let h = (await this.inputProj.forward(graph.x)) as Tensor;

    // Create working graph
    let workingGraph = graph.withFeatures(h);

    // GNN layers
    for (let i = 0; i < this.gnnLayers.length; i++) {
      const layer = this.gnnLayers[i]!;

      // GNN forward
      if (layer instanceof SAGEConv || layer instanceof GINConv) {
        workingGraph = layer.forward(workingGraph);
      }

      // Apply activation and dropout (except last layer)
      if (i < this.gnnLayers.length - 1) {
        let newH = new LeakyReLU(0.1).forward(workingGraph.x) as Tensor;
        newH = this.dropout.forward(newH) as Tensor;
        workingGraph = workingGraph.withFeatures(newH);
      }
    }

    // Scoring head
    let scores = (await this.scoringHead.forward(workingGraph.x)) as Tensor;

    // Mask mastered concepts (set score to -infinity before softmax)
    if (maskMastered) {
      const maskedScores = new Float32Array(scores.size);
      for (let i = 0; i < graph.numNodes; i++) {
        if (maskMastered[i]) {
          maskedScores[i] = -1e9; // Very negative score
        } else {
          maskedScores[i] = scores.data[i] ?? 0;
        }
      }
      scores = new Tensor(maskedScores, scores.shape);
    }

    // Normalize scores if configured
    if (this.config.normalizeScores) {
      // Reshape to [1, numNodes] for softmax, then back to [numNodes, 1]
      const flat = new Tensor(scores.data, [1, graph.numNodes]);
      const normalized = this.softmax.forward(flat) as Tensor;
      scores = new Tensor(normalized.data, [graph.numNodes, 1]);
    }

    return scores;
  }

  /**
   * Recommend next concepts to learn.
   *
   * @param graph Knowledge graph
   * @param masteredNodes Indices of already mastered nodes
   * @param topK Number of recommendations
   * @returns Ranked list of concept recommendations
   */
  async recommend(
    graph: GraphData,
    masteredNodes: number[] = [],
    topK: number = 5
  ): Promise<{
    recommendations: { nodeIndex: number; score: number; rank: number }[];
    allScores: number[];
  }> {
    this.eval();

    // Create mastery mask
    const mask = new Array(graph.numNodes).fill(false);
    for (const idx of masteredNodes) {
      mask[idx] = true;
    }

    // Get scores
    const scores = await this.forward(graph, mask);

    // Convert to array with indices
    const scored: { nodeIndex: number; score: number }[] = [];
    const allScores: number[] = [];
    for (let i = 0; i < graph.numNodes; i++) {
      const score = scores.data[i] ?? 0;
      allScores.push(score);
      if (!mask[i]) {
        scored.push({ nodeIndex: i, score });
      }
    }

    // Sort by score (descending)
    scored.sort((a, b) => b.score - a.score);

    // Return top-k with ranks
    const recommendations = scored.slice(0, topK).map((item, rank) => ({
      ...item,
      rank: rank + 1,
    }));

    return { recommendations, allScores };
  }

  /**
   * Get a complete learning path from current state to goal.
   *
   * @param graph Knowledge graph
   * @param masteredNodes Currently mastered nodes
   * @param goalNodes Target nodes to reach
   * @param maxSteps Maximum path length
   * @returns Ordered learning path
   */
  async getLearningPath(
    graph: GraphData,
    masteredNodes: number[],
    goalNodes: number[],
    maxSteps: number = 20
  ): Promise<{ path: number[]; scores: number[] }> {
    const path: number[] = [];
    const pathScores: number[] = [];
    const mastered = new Set(masteredNodes);
    const goals = new Set(goalNodes);

    // Iteratively recommend until goals are reached or max steps
    for (let step = 0; step < maxSteps; step++) {
      // Check if all goals are mastered
      const allGoalsReached = [...goals].every(g => mastered.has(g));
      if (allGoalsReached) {
        break;
      }

      // Get next recommendation
      const { recommendations } = await this.recommend(
        graph,
        [...mastered],
        1
      );

      if (recommendations.length === 0) {
        break;
      }

      const next = recommendations[0]!;
      path.push(next.nodeIndex);
      pathScores.push(next.score);
      mastered.add(next.nodeIndex);
    }

    return { path, scores: pathScores };
  }

  /**
   * Create a sample curriculum graph for testing.
   */
  static createSampleCurriculum(numConcepts: number = 15): GraphData {
    // Feature dimensions: [embedding(8), difficulty(1), time(1), mastery(1)] = 11
    const featureDim = 11;
    const features = new Float32Array(numConcepts * featureDim);

    // Create random features with increasing difficulty
    for (let i = 0; i < numConcepts; i++) {
      // Random embedding
      for (let j = 0; j < 8; j++) {
        features[i * featureDim + j] = Math.random() * 2 - 1;
      }
      // Difficulty (increases with index)
      features[i * featureDim + 8] = i / numConcepts;
      // Learning time (normalized)
      features[i * featureDim + 9] = 0.3 + Math.random() * 0.5;
      // Current mastery (random, mostly low)
      features[i * featureDim + 10] = Math.random() * 0.3;
    }

    // Create prerequisite edges (tree-like structure)
    const srcEdges: number[] = [];
    const tgtEdges: number[] = [];

    // Create a tree structure with some cross-links
    for (let i = 1; i < numConcepts; i++) {
      // Each node connects to a parent (creates tree)
      const parent = Math.floor((i - 1) / 2);
      srcEdges.push(parent);
      tgtEdges.push(i);
      // Reverse edge for message passing
      srcEdges.push(i);
      tgtEdges.push(parent);

      // Add sibling connections for concepts in same level
      const sibling = i + (i % 2 === 0 ? -1 : 1);
      if (sibling > 0 && sibling < numConcepts) {
        srcEdges.push(i);
        tgtEdges.push(sibling);
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
    const gnnType = this.config.useGIN ? 'GIN' : 'SAGE';
    return `LearningPathRecommender(${this.config.inputFeatures} -> ${gnnType}x${this.config.numLayers} -> scores)`;
  }
}
