/**
 * BrowserGNN by Dr. Lee
 * Graph Tests
 */

import { describe, it, expect } from 'vitest';
import { GraphData, batchGraphs, fromEdgeList, randomGraph } from '../src/core/graph';

describe('GraphData', () => {
  describe('creation', () => {
    it('should create a simple graph', () => {
      const graph = new GraphData({
        x: new Float32Array([1, 2, 3, 4, 5, 6]),
        numNodes: 2,
        numFeatures: 3,
        edgeIndex: new Uint32Array([0, 1, 1, 0]),
        numEdges: 2,
      });

      expect(graph.numNodes).toBe(2);
      expect(graph.numEdges).toBe(2);
      expect(graph.numFeatures).toBe(3);
    });

    it('should validate node features length', () => {
      expect(() => {
        new GraphData({
          x: new Float32Array([1, 2, 3]),
          numNodes: 2,
          numFeatures: 3,
          edgeIndex: new Uint32Array([0, 1]),
          numEdges: 1,
        });
      }).toThrow();
    });

    it('should validate edge index bounds', () => {
      expect(() => {
        new GraphData({
          x: new Float32Array([1, 2]),
          numNodes: 2,
          numFeatures: 1,
          edgeIndex: new Uint32Array([0, 5, 1, 0]), // 5 is out of bounds
          numEdges: 2,
        });
      }).toThrow();
    });
  });

  describe('edge operations', () => {
    const graph = new GraphData({
      x: new Float32Array([1, 2, 3, 4]),
      numNodes: 4,
      numFeatures: 1,
      edgeIndex: new Uint32Array([0, 1, 2, 1, 2, 3]),
      numEdges: 3,
    });

    it('should get source and target nodes', () => {
      expect(Array.from(graph.sourceNodes)).toEqual([0, 1, 2]);
      expect(Array.from(graph.targetNodes)).toEqual([1, 2, 3]);
    });

    it('should get edge by index', () => {
      expect(graph.getEdge(0)).toEqual([0, 1]);
      expect(graph.getEdge(1)).toEqual([1, 2]);
      expect(graph.getEdge(2)).toEqual([2, 3]);
    });

    it('should get neighbors', () => {
      const neighbors = graph.getNeighbors(1);
      expect(neighbors).toContain(2);
    });
  });

  describe('degree computation', () => {
    const graph = new GraphData({
      x: new Float32Array([1, 2, 3]),
      numNodes: 3,
      numFeatures: 1,
      edgeIndex: new Uint32Array([0, 0, 1, 1, 2, 2]),
      numEdges: 3,
    });

    it('should compute in-degrees', () => {
      const degrees = graph.getInDegrees();
      expect(degrees[0]).toBe(0); // No incoming edges to node 0
      expect(degrees[1]).toBe(1); // One edge from 0 to 1
      expect(degrees[2]).toBe(2); // Edges from 0 and 1 to 2
    });

    it('should compute out-degrees', () => {
      const degrees = graph.getOutDegrees();
      expect(degrees[0]).toBe(2); // Node 0 has 2 outgoing
      expect(degrees[1]).toBe(1); // Node 1 has 1 outgoing
      expect(degrees[2]).toBe(0); // Node 2 has 0 outgoing
    });
  });

  describe('self-loops', () => {
    it('should detect graphs without self-loops', () => {
      const graph = new GraphData({
        x: new Float32Array([1, 2]),
        numNodes: 2,
        numFeatures: 1,
        edgeIndex: new Uint32Array([0, 1, 1, 0]),
        numEdges: 2,
      });

      expect(graph.hasSelfLoops()).toBe(false);
    });

    it('should detect graphs with self-loops', () => {
      const graph = new GraphData({
        x: new Float32Array([1, 2]),
        numNodes: 2,
        numFeatures: 1,
        edgeIndex: new Uint32Array([0, 0, 0, 1]),
        numEdges: 2,
      });

      expect(graph.hasSelfLoops()).toBe(true);
    });

    it('should add self-loops', () => {
      const graph = new GraphData({
        x: new Float32Array([1, 2, 3]),
        numNodes: 3,
        numFeatures: 1,
        edgeIndex: new Uint32Array([0, 1, 1, 2]),
        numEdges: 2,
      });

      const withLoops = graph.addSelfLoops();

      expect(withLoops.numEdges).toBe(5); // 2 original + 3 self-loops
      expect(withLoops.hasSelfLoops()).toBe(true);
    });

    it('should remove self-loops', () => {
      const graph = new GraphData({
        x: new Float32Array([1, 2]),
        numNodes: 2,
        numFeatures: 1,
        edgeIndex: new Uint32Array([0, 0, 1, 0, 1, 1]),
        numEdges: 3,
      });

      const noLoops = graph.removeSelfLoops();

      expect(noLoops.numEdges).toBe(1);
      expect(noLoops.hasSelfLoops()).toBe(false);
    });
  });

  describe('adjacency matrix', () => {
    it('should convert to adjacency matrix', () => {
      const graph = new GraphData({
        x: new Float32Array([1, 2, 3]),
        numNodes: 3,
        numFeatures: 1,
        edgeIndex: new Uint32Array([0, 1, 1, 2]),
        numEdges: 2,
      });

      const adj = graph.toAdjacencyMatrix();

      expect(adj.shape).toEqual([3, 3]);
      expect(adj.get(0, 1)).toBe(1);
      expect(adj.get(1, 2)).toBe(1);
      expect(adj.get(0, 2)).toBe(0);
    });
  });

  describe('directed detection', () => {
    it('should detect undirected graph', () => {
      const graph = new GraphData({
        x: new Float32Array([1, 2]),
        numNodes: 2,
        numFeatures: 1,
        edgeIndex: new Uint32Array([0, 1, 1, 0]),
        numEdges: 2,
      });

      expect(graph.isDirected()).toBe(false);
    });

    it('should detect directed graph', () => {
      const graph = new GraphData({
        x: new Float32Array([1, 2]),
        numNodes: 2,
        numFeatures: 1,
        edgeIndex: new Uint32Array([0, 1]),
        numEdges: 1,
      });

      expect(graph.isDirected()).toBe(true);
    });
  });
});

describe('fromEdgeList', () => {
  it('should create graph from edge list', () => {
    const edges: [number, number][] = [
      [0, 1],
      [1, 2],
      [2, 0],
    ];

    const graph = fromEdgeList(edges, 3);

    expect(graph.numNodes).toBe(3);
    expect(graph.numEdges).toBe(3);
  });

  it('should infer number of nodes', () => {
    const edges: [number, number][] = [
      [0, 1],
      [1, 5],
    ];

    const graph = fromEdgeList(edges);

    expect(graph.numNodes).toBe(6); // 0 to 5
  });
});

describe('randomGraph', () => {
  it('should create random graph', () => {
    const graph = randomGraph(10, 0.5, 4);

    expect(graph.numNodes).toBe(10);
    expect(graph.numFeatures).toBe(4);
    expect(graph.numEdges).toBeGreaterThan(0);
  });
});

describe('batchGraphs', () => {
  it('should batch multiple graphs', () => {
    const g1 = new GraphData({
      x: new Float32Array([1, 2, 3, 4]),
      numNodes: 2,
      numFeatures: 2,
      edgeIndex: new Uint32Array([0, 1, 1, 0]),
      numEdges: 2,
    });

    const g2 = new GraphData({
      x: new Float32Array([5, 6, 7, 8, 9, 10]),
      numNodes: 3,
      numFeatures: 2,
      edgeIndex: new Uint32Array([0, 1, 1, 2]),
      numEdges: 2,
    });

    const batched = batchGraphs([g1, g2]);

    expect(batched.numNodes).toBe(5); // 2 + 3
    expect(batched.numEdges).toBe(4); // 2 + 2
    expect(batched.batch).toBeDefined();
    expect(batched.batch![0]).toBe(0);
    expect(batched.batch![2]).toBe(1);
  });
});
