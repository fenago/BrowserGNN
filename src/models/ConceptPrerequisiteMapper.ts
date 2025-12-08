/**
 * BrowserGNN by Dr. Lee
 * Concept Prerequisite Mapper
 *
 * A GNN model for discovering and predicting prerequisite relationships
 * between concepts in an educational domain.
 *
 * Use Cases:
 * - Curriculum design: Automatically discover prerequisite structure
 * - Knowledge gap detection: Find missing prerequisite connections
 * - Transfer learning: Map concepts across different curricula
 * - Dependency analysis: Identify critical path concepts
 *
 * Graph Structure:
 * - Nodes: Concepts/skills
 * - Node Features: Concept embeddings, metadata
 * - Edges: Known prerequisite relationships (can be sparse/incomplete)
 * - Output: Link prediction scores for potential prerequisite edges
 */

import { Tensor } from '../core/tensor';
import { GraphData } from '../core/graph';
import { Module, Sequential } from '../nn';
import { Linear, ReLU, Sigmoid, Dropout, ELU } from '../nn';
import { GCNConv } from '../layers/gcn';
import { GATConv } from '../layers/gat';

export interface ConceptPrerequisiteMapperConfig {
  /**
   * Number of input features per concept node.
   */
  inputFeatures: number;

  /**
   * Hidden dimension for GNN layers and embeddings.
   * Default: 64
   */
  hiddenDim?: number;

  /**
   * Number of GNN layers for encoding.
   * Default: 2
   */
  numLayers?: number;

  /**
   * Whether to use attention mechanism.
   * Default: true (GAT better captures asymmetric relationships)
   */
  useAttention?: boolean;

  /**
   * Number of attention heads.
   * Default: 4
   */
  numHeads?: number;

  /**
   * Dropout rate.
   * Default: 0.2
   */
  dropout?: number;

  /**
   * Link prediction method: 'dot' (dot product), 'bilinear', or 'mlp'
   * Default: 'mlp'
   */
  linkPredictor?: 'dot' | 'bilinear' | 'mlp';
}

/**
 * Concept Prerequisite Mapper Model
 *
 * Architecture:
 * 1. Input projection
 * 2. GNN encoder (GCN or GAT) for node embeddings
 * 3. Link prediction head for prerequisite scoring
 *
 * The model learns embeddings such that prerequisite relationships
 * can be predicted from the embeddings of source and target concepts.
 */
export class ConceptPrerequisiteMapper extends Module {
  readonly config: Required<ConceptPrerequisiteMapperConfig>;

  private inputProj: Sequential;
  private gnnLayers: Module[];
  private dropout: Dropout;
  private linkPredictor: Module | null;
  private bilinearWeight: Tensor | null;
  private sigmoid: Sigmoid;

  constructor(config: ConceptPrerequisiteMapperConfig) {
    super();

    // Apply defaults
    this.config = {
      inputFeatures: config.inputFeatures,
      hiddenDim: config.hiddenDim ?? 64,
      numLayers: config.numLayers ?? 2,
      useAttention: config.useAttention ?? true,
      numHeads: config.numHeads ?? 4,
      dropout: config.dropout ?? 0.2,
      linkPredictor: config.linkPredictor ?? 'mlp',
    };

    // Input projection
    this.inputProj = new Sequential([
      new Linear({
        inFeatures: this.config.inputFeatures,
        outFeatures: this.config.hiddenDim,
      }),
      new ELU(),
      new Dropout(this.config.dropout),
    ]);
    this.registerModule('inputProj', this.inputProj);

    // GNN encoder layers
    this.gnnLayers = [];
    for (let i = 0; i < this.config.numLayers; i++) {
      let layer: Module;

      if (this.config.useAttention) {
        const outChannels = Math.floor(this.config.hiddenDim / this.config.numHeads);
        layer = new GATConv({
          inChannels: this.config.hiddenDim,
          outChannels,
          heads: this.config.numHeads,
          concat: true,
          dropout: this.config.dropout,
        });
      } else {
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

    // Link prediction head
    this.bilinearWeight = null;
    this.linkPredictor = null;

    if (this.config.linkPredictor === 'bilinear') {
      // Bilinear: score = h_src^T * W * h_tgt
      const weightData = new Float32Array(this.config.hiddenDim * this.config.hiddenDim);
      const stdv = Math.sqrt(2.0 / this.config.hiddenDim);
      for (let i = 0; i < weightData.length; i++) {
        const u1 = Math.random();
        const u2 = Math.random();
        weightData[i] = stdv * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }
      this.bilinearWeight = new Tensor(weightData, [this.config.hiddenDim, this.config.hiddenDim]);
      this.registerParameter('bilinearWeight', this.bilinearWeight);
    } else if (this.config.linkPredictor === 'mlp') {
      // MLP: score = MLP([h_src, h_tgt, h_src * h_tgt])
      const mlpInputDim = this.config.hiddenDim * 3;
      this.linkPredictor = new Sequential([
        new Linear({
          inFeatures: mlpInputDim,
          outFeatures: this.config.hiddenDim,
        }),
        new ReLU(),
        new Dropout(this.config.dropout / 2),
        new Linear({
          inFeatures: this.config.hiddenDim,
          outFeatures: 1,
        }),
      ]);
      this.registerModule('linkPredictor', this.linkPredictor);
    }
    // 'dot' doesn't need additional parameters

    this.sigmoid = new Sigmoid();
  }

  /**
   * Encode nodes to get embeddings.
   *
   * @param graph Knowledge graph
   * @returns Node embeddings [numNodes, hiddenDim]
   */
  async encode(graph: GraphData): Promise<Tensor> {
    // Input projection
    let h = (await this.inputProj.forward(graph.x)) as Tensor;

    // Create working graph
    let workingGraph = graph.withFeatures(h);

    // GNN layers
    for (let i = 0; i < this.gnnLayers.length; i++) {
      const layer = this.gnnLayers[i]!;

      // GNN forward
      if (layer instanceof GCNConv || layer instanceof GATConv) {
        workingGraph = layer.forward(workingGraph);
      }

      // Apply activation (except last layer)
      if (i < this.gnnLayers.length - 1) {
        let newH = new ELU().forward(workingGraph.x) as Tensor;
        newH = this.dropout.forward(newH) as Tensor;
        workingGraph = workingGraph.withFeatures(newH);
      }
    }

    return workingGraph.x;
  }

  /**
   * Predict prerequisite probability for given node pairs.
   *
   * @param embeddings Node embeddings from encode()
   * @param srcNodes Source node indices
   * @param tgtNodes Target node indices
   * @returns Prerequisite probabilities for each pair
   */
  predictLinks(
    embeddings: Tensor,
    srcNodes: number[],
    tgtNodes: number[]
  ): number[] {
    const numPairs = srcNodes.length;
    const scores: number[] = [];
    const hiddenDim = this.config.hiddenDim;

    for (let p = 0; p < numPairs; p++) {
      const srcIdx = srcNodes[p]!;
      const tgtIdx = tgtNodes[p]!;

      // Get embeddings
      const srcEmb = new Float32Array(hiddenDim);
      const tgtEmb = new Float32Array(hiddenDim);
      for (let i = 0; i < hiddenDim; i++) {
        srcEmb[i] = embeddings.data[srcIdx * hiddenDim + i] ?? 0;
        tgtEmb[i] = embeddings.data[tgtIdx * hiddenDim + i] ?? 0;
      }

      let score: number;

      if (this.config.linkPredictor === 'dot') {
        // Dot product
        score = 0;
        for (let i = 0; i < hiddenDim; i++) {
          score += srcEmb[i]! * tgtEmb[i]!;
        }
        score = 1 / (1 + Math.exp(-score)); // Sigmoid
      } else if (this.config.linkPredictor === 'bilinear' && this.bilinearWeight) {
        // Bilinear: src^T * W * tgt
        const Wtgt = new Float32Array(hiddenDim);
        for (let i = 0; i < hiddenDim; i++) {
          let sum = 0;
          for (let j = 0; j < hiddenDim; j++) {
            sum += (this.bilinearWeight.data[i * hiddenDim + j] ?? 0) * (tgtEmb[j] ?? 0);
          }
          Wtgt[i] = sum;
        }
        score = 0;
        for (let i = 0; i < hiddenDim; i++) {
          score += (srcEmb[i] ?? 0) * (Wtgt[i] ?? 0);
        }
        score = 1 / (1 + Math.exp(-score)); // Sigmoid
      } else if (this.linkPredictor) {
        // MLP: [src, tgt, src * tgt]
        const mlpInput = new Float32Array(hiddenDim * 3);
        for (let i = 0; i < hiddenDim; i++) {
          mlpInput[i] = srcEmb[i]!;
          mlpInput[hiddenDim + i] = tgtEmb[i]!;
          mlpInput[2 * hiddenDim + i] = srcEmb[i]! * tgtEmb[i]!;
        }
        const inputTensor = new Tensor(mlpInput, [1, hiddenDim * 3]);
        const output = this.linkPredictor.forward(inputTensor) as Tensor;
        score = 1 / (1 + Math.exp(-(output.data[0] ?? 0))); // Sigmoid
      } else {
        score = 0;
      }

      scores.push(score);
    }

    return scores;
  }

  /**
   * Forward pass to get node embeddings.
   * Use predictAllLinks for link prediction.
   *
   * @param graph Knowledge graph
   * @returns Node embeddings
   */
  async forward(graph: GraphData): Promise<Tensor> {
    return this.encode(graph);
  }

  /**
   * Full forward pass: encode graph and predict all potential links.
   *
   * @param graph Knowledge graph
   * @param candidatePairs Optional specific pairs to evaluate
   * @returns Link predictions
   */
  async predictAllLinks(
    graph: GraphData,
    candidatePairs?: { src: number[]; tgt: number[] }
  ): Promise<{
    embeddings: Tensor;
    scores: number[];
    pairs: { src: number; tgt: number; score: number }[];
  }> {
    const embeddings = await this.encode(graph);

    let srcNodes: number[];
    let tgtNodes: number[];

    if (candidatePairs) {
      srcNodes = candidatePairs.src;
      tgtNodes = candidatePairs.tgt;
    } else {
      // Evaluate all possible pairs (excluding self-loops)
      srcNodes = [];
      tgtNodes = [];
      for (let i = 0; i < graph.numNodes; i++) {
        for (let j = 0; j < graph.numNodes; j++) {
          if (i !== j) {
            srcNodes.push(i);
            tgtNodes.push(j);
          }
        }
      }
    }

    const scores = this.predictLinks(embeddings, srcNodes, tgtNodes);

    const pairs = srcNodes.map((src, i) => ({
      src,
      tgt: tgtNodes[i]!,
      score: scores[i]!,
    }));

    return { embeddings, scores, pairs };
  }

  /**
   * Discover new prerequisite relationships.
   *
   * @param graph Knowledge graph with known prerequisites
   * @param threshold Score threshold for prediction
   * @param topK Maximum number of new edges to return
   * @returns Predicted new prerequisite edges
   */
  async discoverPrerequisites(
    graph: GraphData,
    threshold: number = 0.7,
    topK: number = 20
  ): Promise<{
    newEdges: { src: number; tgt: number; score: number }[];
    existingEdges: Set<string>;
  }> {
    this.eval();

    // Get existing edges
    const existingEdges = new Set<string>();
    for (let i = 0; i < graph.numEdges; i++) {
      const src = graph.edgeIndex[i]!;
      const tgt = graph.edgeIndex[graph.numEdges + i]!;
      existingEdges.add(`${src}->${tgt}`);
    }

    // Get all predictions
    const { pairs } = await this.predictAllLinks(graph);

    // Filter to new, high-confidence edges
    const newEdges = pairs
      .filter(p => !existingEdges.has(`${p.src}->${p.tgt}`))
      .filter(p => p.score >= threshold)
      .sort((a, b) => b.score - a.score)
      .slice(0, topK);

    return { newEdges, existingEdges };
  }

  /**
   * Find critical prerequisite chains (longest paths).
   *
   * @param graph Knowledge graph
   * @param startNode Starting concept
   * @returns Critical path information
   */
  findCriticalPath(
    graph: GraphData,
    startNode: number
  ): {
    path: number[];
    length: number;
  } {
    // Build adjacency list from edges
    const adj = new Map<number, number[]>();
    for (let i = 0; i < graph.numEdges; i++) {
      const src = graph.edgeIndex[i]!;
      const tgt = graph.edgeIndex[graph.numEdges + i]!;
      if (!adj.has(src)) {
        adj.set(src, []);
      }
      adj.get(src)!.push(tgt);
    }

    // DFS to find longest path
    const visited = new Set<number>();
    const memo = new Map<number, { path: number[]; length: number }>();

    const dfs = (node: number): { path: number[]; length: number } => {
      if (memo.has(node)) {
        return memo.get(node)!;
      }
      if (visited.has(node)) {
        return { path: [node], length: 0 };
      }

      visited.add(node);
      const neighbors = adj.get(node) || [];

      let bestPath = [node];
      let bestLength = 0;

      for (const next of neighbors) {
        const result = dfs(next);
        if (result.length + 1 > bestLength) {
          bestLength = result.length + 1;
          bestPath = [node, ...result.path];
        }
      }

      visited.delete(node);
      const result = { path: bestPath, length: bestLength };
      memo.set(node, result);
      return result;
    };

    return dfs(startNode);
  }

  /**
   * Create a sample concept graph for testing.
   */
  static createSampleConceptGraph(numConcepts: number = 12): GraphData {
    // Feature dimensions: [embedding(16), difficulty(1), domain(3)] = 20
    const featureDim = 20;
    const features = new Float32Array(numConcepts * featureDim);

    // Create features
    for (let i = 0; i < numConcepts; i++) {
      // Random embedding
      for (let j = 0; j < 16; j++) {
        features[i * featureDim + j] = Math.random() * 2 - 1;
      }
      // Difficulty
      features[i * featureDim + 16] = i / numConcepts;
      // Domain (one-hot, 3 domains)
      const domain = i % 3;
      features[i * featureDim + 17 + domain] = 1;
    }

    // Create prerequisite edges (some known, some to be discovered)
    const srcEdges: number[] = [];
    const tgtEdges: number[] = [];

    // Create known prerequisite chains
    for (let i = 0; i < numConcepts - 3; i += 3) {
      srcEdges.push(i);
      tgtEdges.push(i + 1);
      srcEdges.push(i + 1);
      tgtEdges.push(i + 2);
      // Reverse edges for message passing
      srcEdges.push(i + 1);
      tgtEdges.push(i);
      srcEdges.push(i + 2);
      tgtEdges.push(i + 1);
    }

    // Cross-domain connections
    for (let d = 0; d < 3; d++) {
      srcEdges.push(d);
      tgtEdges.push(d + 3);
      srcEdges.push(d + 3);
      tgtEdges.push(d);
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
    return `ConceptPrerequisiteMapper(${this.config.inputFeatures} -> ${gnnType}x${this.config.numLayers} -> ${this.config.linkPredictor})`;
  }
}
