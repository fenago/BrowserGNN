/**
 * BrowserGNN by Dr. Lee
 * Graph Data Structure
 *
 * Core data structure for representing graphs in GNN computations.
 * Follows PyTorch Geometric conventions for compatibility.
 */

import { Tensor } from './tensor';

/**
 * Configuration for creating a GraphData instance
 */
export interface GraphDataConfig {
  /** Node features as flat array [numNodes * numFeatures] */
  x: Float32Array | number[];
  /** Number of nodes in the graph */
  numNodes: number;
  /** Number of features per node */
  numFeatures: number;
  /** Edge index in COO format [2 * numEdges]: [src0, src1, ..., tgt0, tgt1, ...] */
  edgeIndex: Uint32Array | number[];
  /** Number of edges */
  numEdges: number;
  /** Optional edge attributes [numEdges * edgeFeatures] */
  edgeAttr?: Float32Array | number[];
  /** Number of edge features (required if edgeAttr provided) */
  numEdgeFeatures?: number;
  /** Optional node labels [numNodes] or [numNodes * numClasses] */
  y?: Float32Array | number[];
  /** Optional graph-level label */
  graphLabel?: number;
  /** Optional batch assignment for batched graphs [numNodes] */
  batch?: Uint32Array | number[];
}

/**
 * GraphData class - Core data structure for graph neural networks
 *
 * Stores node features, edge connectivity, and optional attributes
 * in a format optimized for message passing operations.
 */
export class GraphData {
  /** Node feature tensor [numNodes, numFeatures] */
  readonly x: Tensor;
  /** Number of nodes */
  readonly numNodes: number;
  /** Number of features per node */
  readonly numFeatures: number;
  /** Edge index tensor [2, numEdges] */
  readonly edgeIndex: Uint32Array;
  /** Number of edges */
  readonly numEdges: number;
  /** Optional edge attributes tensor [numEdges, numEdgeFeatures] */
  readonly edgeAttr?: Tensor;
  /** Number of edge features */
  readonly numEdgeFeatures: number;
  /** Optional node labels tensor */
  readonly y?: Tensor;
  /** Optional graph-level label */
  readonly graphLabel?: number;
  /** Batch assignment for batched graphs */
  readonly batch?: Uint32Array;

  constructor(config: GraphDataConfig) {
    // Validate and store node features
    const xData = config.x instanceof Float32Array ? config.x : new Float32Array(config.x);
    if (xData.length !== config.numNodes * config.numFeatures) {
      throw new Error(
        `Node features length ${xData.length} doesn't match numNodes(${config.numNodes}) * numFeatures(${config.numFeatures})`
      );
    }
    this.x = new Tensor(xData, [config.numNodes, config.numFeatures]);
    this.numNodes = config.numNodes;
    this.numFeatures = config.numFeatures;

    // Validate and store edge index
    const edgeData =
      config.edgeIndex instanceof Uint32Array
        ? config.edgeIndex
        : new Uint32Array(config.edgeIndex);
    if (edgeData.length !== config.numEdges * 2) {
      throw new Error(
        `Edge index length ${edgeData.length} doesn't match numEdges(${config.numEdges}) * 2`
      );
    }
    // Validate edge indices are within bounds
    for (let i = 0; i < edgeData.length; i++) {
      if (edgeData[i]! >= config.numNodes) {
        throw new Error(`Edge index ${edgeData[i]} out of bounds for ${config.numNodes} nodes`);
      }
    }
    this.edgeIndex = edgeData;
    this.numEdges = config.numEdges;

    // Store optional edge attributes
    if (config.edgeAttr) {
      if (!config.numEdgeFeatures) {
        throw new Error('numEdgeFeatures required when edgeAttr provided');
      }
      const edgeAttrData =
        config.edgeAttr instanceof Float32Array
          ? config.edgeAttr
          : new Float32Array(config.edgeAttr);
      this.edgeAttr = new Tensor(edgeAttrData, [config.numEdges, config.numEdgeFeatures]);
      this.numEdgeFeatures = config.numEdgeFeatures;
    } else {
      this.numEdgeFeatures = 0;
    }

    // Store optional labels
    if (config.y) {
      const yData = config.y instanceof Float32Array ? config.y : new Float32Array(config.y);
      // Determine if it's node labels or one-hot encoded
      if (yData.length === config.numNodes) {
        this.y = new Tensor(yData, [config.numNodes]);
      } else {
        const numClasses = yData.length / config.numNodes;
        this.y = new Tensor(yData, [config.numNodes, numClasses]);
      }
    }

    this.graphLabel = config.graphLabel;

    // Store batch assignment
    if (config.batch) {
      this.batch =
        config.batch instanceof Uint32Array ? config.batch : new Uint32Array(config.batch);
    }
  }

  /**
   * Get source nodes from edge index
   */
  get sourceNodes(): Uint32Array {
    return this.edgeIndex.slice(0, this.numEdges);
  }

  /**
   * Get target nodes from edge index
   */
  get targetNodes(): Uint32Array {
    return this.edgeIndex.slice(this.numEdges);
  }

  /**
   * Get edge as [source, target] pair
   */
  getEdge(idx: number): [number, number] {
    if (idx < 0 || idx >= this.numEdges) {
      throw new Error(`Edge index ${idx} out of bounds`);
    }
    return [this.edgeIndex[idx]!, this.edgeIndex[this.numEdges + idx]!];
  }

  /**
   * Get neighbors of a node
   */
  getNeighbors(nodeIdx: number): number[] {
    const neighbors: number[] = [];
    for (let i = 0; i < this.numEdges; i++) {
      if (this.edgeIndex[i] === nodeIdx) {
        neighbors.push(this.edgeIndex[this.numEdges + i]!);
      }
    }
    return neighbors;
  }

  /**
   * Get in-degree of each node
   */
  getInDegrees(): Uint32Array {
    const degrees = new Uint32Array(this.numNodes);
    for (let i = 0; i < this.numEdges; i++) {
      const target = this.edgeIndex[this.numEdges + i];
      if (target !== undefined) {
        degrees[target] = (degrees[target] ?? 0) + 1;
      }
    }
    return degrees;
  }

  /**
   * Get out-degree of each node
   */
  getOutDegrees(): Uint32Array {
    const degrees = new Uint32Array(this.numNodes);
    for (let i = 0; i < this.numEdges; i++) {
      const source = this.edgeIndex[i];
      if (source !== undefined) {
        degrees[source] = (degrees[source] ?? 0) + 1;
      }
    }
    return degrees;
  }

  /**
   * Check if graph is directed (has asymmetric edges)
   */
  isDirected(): boolean {
    const edgeSet = new Set<string>();
    for (let i = 0; i < this.numEdges; i++) {
      const src = this.edgeIndex[i]!;
      const tgt = this.edgeIndex[this.numEdges + i]!;
      edgeSet.add(`${src}-${tgt}`);
    }
    for (let i = 0; i < this.numEdges; i++) {
      const src = this.edgeIndex[i]!;
      const tgt = this.edgeIndex[this.numEdges + i]!;
      if (!edgeSet.has(`${tgt}-${src}`)) {
        return true;
      }
    }
    return false;
  }

  /**
   * Check if graph has self-loops
   */
  hasSelfLoops(): boolean {
    for (let i = 0; i < this.numEdges; i++) {
      if (this.edgeIndex[i] === this.edgeIndex[this.numEdges + i]) {
        return true;
      }
    }
    return false;
  }

  /**
   * Add self-loops to the graph
   */
  addSelfLoops(): GraphData {
    // Check which nodes already have self-loops
    const hasSelfLoop = new Set<number>();
    for (let i = 0; i < this.numEdges; i++) {
      if (this.edgeIndex[i] === this.edgeIndex[this.numEdges + i]) {
        hasSelfLoop.add(this.edgeIndex[i]!);
      }
    }

    // Count new self-loops needed
    const newLoops = this.numNodes - hasSelfLoop.size;
    const newNumEdges = this.numEdges + newLoops;

    // Create new edge index
    const newEdgeIndex = new Uint32Array(newNumEdges * 2);
    newEdgeIndex.set(this.edgeIndex.slice(0, this.numEdges), 0);
    newEdgeIndex.set(this.edgeIndex.slice(this.numEdges), newNumEdges);

    // Add missing self-loops
    let idx = this.numEdges;
    for (let i = 0; i < this.numNodes; i++) {
      if (!hasSelfLoop.has(i)) {
        newEdgeIndex[idx] = i;
        newEdgeIndex[newNumEdges + idx] = i;
        idx++;
      }
    }

    return new GraphData({
      x: this.x.data,
      numNodes: this.numNodes,
      numFeatures: this.numFeatures,
      edgeIndex: newEdgeIndex,
      numEdges: newNumEdges,
      edgeAttr: this.edgeAttr?.data,
      numEdgeFeatures: this.numEdgeFeatures || undefined,
      y: this.y?.data,
      graphLabel: this.graphLabel,
      batch: this.batch,
    });
  }

  /**
   * Remove self-loops from the graph
   */
  removeSelfLoops(): GraphData {
    const nonSelfLoopIndices: number[] = [];
    for (let i = 0; i < this.numEdges; i++) {
      if (this.edgeIndex[i] !== this.edgeIndex[this.numEdges + i]) {
        nonSelfLoopIndices.push(i);
      }
    }

    const newNumEdges = nonSelfLoopIndices.length;
    const newEdgeIndex = new Uint32Array(newNumEdges * 2);

    for (let i = 0; i < newNumEdges; i++) {
      const oldIdx = nonSelfLoopIndices[i]!;
      newEdgeIndex[i] = this.edgeIndex[oldIdx]!;
      newEdgeIndex[newNumEdges + i] = this.edgeIndex[this.numEdges + oldIdx]!;
    }

    return new GraphData({
      x: this.x.data,
      numNodes: this.numNodes,
      numFeatures: this.numFeatures,
      edgeIndex: newEdgeIndex,
      numEdges: newNumEdges,
      y: this.y?.data,
      graphLabel: this.graphLabel,
      batch: this.batch,
    });
  }

  /**
   * Create a copy with new node features
   */
  withFeatures(newX: Tensor): GraphData {
    if (newX.shape[0] !== this.numNodes) {
      throw new Error(`New features must have ${this.numNodes} nodes`);
    }
    return new GraphData({
      x: newX.data,
      numNodes: this.numNodes,
      numFeatures: newX.shape[1]!,
      edgeIndex: this.edgeIndex,
      numEdges: this.numEdges,
      edgeAttr: this.edgeAttr?.data,
      numEdgeFeatures: this.numEdgeFeatures || undefined,
      y: this.y?.data,
      graphLabel: this.graphLabel,
      batch: this.batch,
    });
  }

  /**
   * Convert to adjacency list representation
   */
  toAdjacencyList(): Map<number, number[]> {
    const adj = new Map<number, number[]>();
    for (let i = 0; i < this.numNodes; i++) {
      adj.set(i, []);
    }
    for (let i = 0; i < this.numEdges; i++) {
      const src = this.edgeIndex[i]!;
      const tgt = this.edgeIndex[this.numEdges + i]!;
      adj.get(src)!.push(tgt);
    }
    return adj;
  }

  /**
   * Convert edge index to dense adjacency matrix
   */
  toAdjacencyMatrix(): Tensor {
    const adj = Tensor.zeros([this.numNodes, this.numNodes]);
    for (let i = 0; i < this.numEdges; i++) {
      const src = this.edgeIndex[i]!;
      const tgt = this.edgeIndex[this.numEdges + i]!;
      adj.set(1, src, tgt);
    }
    return adj;
  }

  /**
   * Get normalized adjacency matrix with self-loops (for GCN)
   * D^{-1/2} * (A + I) * D^{-1/2}
   */
  getNormalizedAdjacency(): Tensor {
    const graphWithLoops = this.hasSelfLoops() ? this : this.addSelfLoops();
    const adj = graphWithLoops.toAdjacencyMatrix();

    // Calculate degree
    const degrees = new Float32Array(this.numNodes);
    for (let i = 0; i < graphWithLoops.numEdges; i++) {
      const tgt = graphWithLoops.edgeIndex[graphWithLoops.numEdges + i];
      if (tgt !== undefined) {
        degrees[tgt] = (degrees[tgt] ?? 0) + 1;
      }
    }

    // D^{-1/2}
    const dInvSqrt = new Float32Array(this.numNodes);
    for (let i = 0; i < this.numNodes; i++) {
      dInvSqrt[i] = degrees[i]! > 0 ? 1 / Math.sqrt(degrees[i]!) : 0;
    }

    // Normalize: D^{-1/2} * A * D^{-1/2}
    const result = new Float32Array(this.numNodes * this.numNodes);
    for (let i = 0; i < this.numNodes; i++) {
      for (let j = 0; j < this.numNodes; j++) {
        const aij = adj.get(i, j);
        result[i * this.numNodes + j] = dInvSqrt[i]! * aij * dInvSqrt[j]!;
      }
    }

    return new Tensor(result, [this.numNodes, this.numNodes]);
  }

  /**
   * Clone the graph
   */
  clone(): GraphData {
    return new GraphData({
      x: new Float32Array(this.x.data),
      numNodes: this.numNodes,
      numFeatures: this.numFeatures,
      edgeIndex: new Uint32Array(this.edgeIndex),
      numEdges: this.numEdges,
      edgeAttr: this.edgeAttr ? new Float32Array(this.edgeAttr.data) : undefined,
      numEdgeFeatures: this.numEdgeFeatures || undefined,
      y: this.y ? new Float32Array(this.y.data) : undefined,
      graphLabel: this.graphLabel,
      batch: this.batch ? new Uint32Array(this.batch) : undefined,
    });
  }

  /**
   * String representation
   */
  toString(): string {
    return `GraphData(numNodes=${this.numNodes}, numEdges=${this.numEdges}, numFeatures=${this.numFeatures})`;
  }

  /**
   * Print graph summary
   */
  print(): void {
    console.log('GraphData:');
    console.log(`  Nodes: ${this.numNodes}`);
    console.log(`  Edges: ${this.numEdges}`);
    console.log(`  Node features: ${this.numFeatures}`);
    if (this.edgeAttr) {
      console.log(`  Edge features: ${this.numEdgeFeatures}`);
    }
    console.log(`  Directed: ${this.isDirected()}`);
    console.log(`  Has self-loops: ${this.hasSelfLoops()}`);
    if (this.y) {
      console.log(`  Labels shape: [${this.y.shape.join(', ')}]`);
    }
  }
}

/**
 * Batch multiple graphs into a single GraphData object
 * Used for mini-batch training/inference
 */
export function batchGraphs(graphs: GraphData[]): GraphData {
  if (graphs.length === 0) {
    throw new Error('Cannot batch empty array of graphs');
  }

  const numFeatures = graphs[0]!.numFeatures;
  for (const g of graphs) {
    if (g.numFeatures !== numFeatures) {
      throw new Error('All graphs must have same number of features');
    }
  }

  // Calculate total sizes
  const totalNodes = graphs.reduce((sum, g) => sum + g.numNodes, 0);
  const totalEdges = graphs.reduce((sum, g) => sum + g.numEdges, 0);

  // Concatenate node features
  const x = new Float32Array(totalNodes * numFeatures);
  const batch = new Uint32Array(totalNodes);
  const edgeIndex = new Uint32Array(totalEdges * 2);

  let nodeOffset = 0;
  let edgeOffset = 0;

  for (let graphIdx = 0; graphIdx < graphs.length; graphIdx++) {
    const g = graphs[graphIdx]!;

    // Copy node features
    x.set(g.x.data, nodeOffset * numFeatures);

    // Set batch assignment
    for (let i = 0; i < g.numNodes; i++) {
      batch[nodeOffset + i] = graphIdx;
    }

    // Copy edges with offset
    for (let i = 0; i < g.numEdges; i++) {
      edgeIndex[edgeOffset + i] = g.edgeIndex[i]! + nodeOffset;
      edgeIndex[totalEdges + edgeOffset + i] = g.edgeIndex[g.numEdges + i]! + nodeOffset;
    }

    nodeOffset += g.numNodes;
    edgeOffset += g.numEdges;
  }

  return new GraphData({
    x,
    numNodes: totalNodes,
    numFeatures,
    edgeIndex,
    numEdges: totalEdges,
    batch,
  });
}

/**
 * Create a simple graph from edge list
 */
export function fromEdgeList(
  edges: [number, number][],
  numNodes?: number,
  nodeFeatures?: Float32Array | number[]
): GraphData {
  const actualNumNodes = numNodes ?? Math.max(...edges.flat()) + 1;
  const numEdges = edges.length;

  const edgeIndex = new Uint32Array(numEdges * 2);
  for (let i = 0; i < numEdges; i++) {
    edgeIndex[i] = edges[i]![0];
    edgeIndex[numEdges + i] = edges[i]![1];
  }

  const x = nodeFeatures
    ? nodeFeatures instanceof Float32Array
      ? nodeFeatures
      : new Float32Array(nodeFeatures)
    : new Float32Array(actualNumNodes); // Single feature per node (degree or 1)

  const numFeatures = nodeFeatures ? x.length / actualNumNodes : 1;

  // If no features provided, use 1 as default feature
  if (!nodeFeatures) {
    x.fill(1);
  }

  return new GraphData({
    x,
    numNodes: actualNumNodes,
    numFeatures,
    edgeIndex,
    numEdges,
  });
}

/**
 * Create a random graph using Erdos-Renyi model
 */
export function randomGraph(
  numNodes: number,
  edgeProbability: number,
  numFeatures: number = 1
): GraphData {
  const edges: [number, number][] = [];

  for (let i = 0; i < numNodes; i++) {
    for (let j = 0; j < numNodes; j++) {
      if (i !== j && Math.random() < edgeProbability) {
        edges.push([i, j]);
      }
    }
  }

  // Random node features
  const x = new Float32Array(numNodes * numFeatures);
  for (let i = 0; i < x.length; i++) {
    x[i] = Math.random();
  }

  return fromEdgeList(edges, numNodes, x);
}
