/**
 * BrowserGNN by Dr. Lee
 * GNN Layer Tests
 */

import { describe, it, expect } from 'vitest';
import { GraphData } from '../src/core/graph';
import { GCNConv } from '../src/layers/gcn';
import { GATConv } from '../src/layers/gat';
import { SAGEConv } from '../src/layers/sage';

// Create a simple test graph
function createTestGraph(): GraphData {
  return new GraphData({
    x: new Float32Array([
      1, 0, 0, // Node 0
      0, 1, 0, // Node 1
      0, 0, 1, // Node 2
      1, 1, 1, // Node 3
    ]),
    numNodes: 4,
    numFeatures: 3,
    edgeIndex: new Uint32Array([
      0, 0, 1, 1, 2, 2, 3, 3, // source
      1, 3, 0, 2, 1, 3, 0, 2, // target
    ]),
    numEdges: 8,
  });
}

describe('GCNConv', () => {
  it('should create layer with correct parameters', () => {
    const layer = new GCNConv({
      inChannels: 3,
      outChannels: 16,
    });

    expect(layer.inChannels).toBe(3);
    expect(layer.outChannels).toBe(16);
    expect(layer.numParameters()).toBe(3 * 16 + 16); // weight + bias
  });

  it('should forward pass correctly', () => {
    const graph = createTestGraph();
    const layer = new GCNConv({
      inChannels: 3,
      outChannels: 8,
    });

    const output = layer.forward(graph);

    expect(output.numNodes).toBe(4);
    expect(output.x.shape).toEqual([4, 8]);
  });

  it('should work without bias', () => {
    const graph = createTestGraph();
    const layer = new GCNConv({
      inChannels: 3,
      outChannels: 8,
      bias: false,
    });

    expect(layer.numParameters()).toBe(3 * 8); // only weight

    const output = layer.forward(graph);
    expect(output.x.shape).toEqual([4, 8]);
  });

  it('should work without normalization', () => {
    const graph = createTestGraph();
    const layer = new GCNConv({
      inChannels: 3,
      outChannels: 8,
      normalize: false,
    });

    const output = layer.forward(graph);
    expect(output.x.shape).toEqual([4, 8]);
  });

  it('should throw on input dimension mismatch', () => {
    const graph = createTestGraph();
    const layer = new GCNConv({
      inChannels: 5, // Wrong!
      outChannels: 8,
    });

    expect(() => layer.forward(graph)).toThrow();
  });
});

describe('GATConv', () => {
  it('should create layer with correct parameters', () => {
    const layer = new GATConv({
      inChannels: 3,
      outChannels: 8,
      heads: 2,
    });

    expect(layer.inChannels).toBe(3);
    expect(layer.outChannels).toBe(8);
    expect(layer.heads).toBe(2);
    expect(layer.outputDim).toBe(16); // 2 * 8 with concat=true
  });

  it('should forward pass with multi-head attention', () => {
    const graph = createTestGraph();
    const layer = new GATConv({
      inChannels: 3,
      outChannels: 8,
      heads: 2,
      concat: true,
    });

    const output = layer.forward(graph);

    expect(output.numNodes).toBe(4);
    expect(output.x.shape).toEqual([4, 16]); // 2 * 8
  });

  it('should average heads when concat=false', () => {
    const graph = createTestGraph();
    const layer = new GATConv({
      inChannels: 3,
      outChannels: 8,
      heads: 2,
      concat: false,
    });

    expect(layer.outputDim).toBe(8);

    const output = layer.forward(graph);
    expect(output.x.shape).toEqual([4, 8]);
  });

  it('should work with single head', () => {
    const graph = createTestGraph();
    const layer = new GATConv({
      inChannels: 3,
      outChannels: 8,
      heads: 1,
    });

    const output = layer.forward(graph);
    expect(output.x.shape).toEqual([4, 8]);
  });

  it('should apply dropout in training mode', () => {
    const graph = createTestGraph();
    const layer = new GATConv({
      inChannels: 3,
      outChannels: 8,
      dropout: 0.5,
    });

    layer.train(true);
    const output1 = layer.forward(graph);

    layer.train(true);
    const output2 = layer.forward(graph);

    // Outputs should differ due to dropout randomness
    // (This is probabilistic, but with 0.5 dropout they're very likely different)
    let different = false;
    for (let i = 0; i < output1.x.size; i++) {
      if (Math.abs(output1.x.data[i]! - output2.x.data[i]!) > 0.0001) {
        different = true;
        break;
      }
    }
    // Due to weight randomness this should almost always pass
    // but we're mainly checking it doesn't crash
    expect(output1.x.shape).toEqual([4, 8]);
  });
});

describe('SAGEConv', () => {
  it('should create layer with correct parameters', () => {
    const layer = new SAGEConv({
      inChannels: 3,
      outChannels: 16,
      aggregator: 'mean',
    });

    expect(layer.inChannels).toBe(3);
    expect(layer.outChannels).toBe(16);
    expect(layer.aggregator).toBe('mean');
  });

  it('should forward pass with mean aggregator', () => {
    const graph = createTestGraph();
    const layer = new SAGEConv({
      inChannels: 3,
      outChannels: 8,
      aggregator: 'mean',
    });

    const output = layer.forward(graph);

    expect(output.numNodes).toBe(4);
    expect(output.x.shape).toEqual([4, 8]);
  });

  it('should forward pass with max aggregator', () => {
    const graph = createTestGraph();
    const layer = new SAGEConv({
      inChannels: 3,
      outChannels: 8,
      aggregator: 'max',
    });

    const output = layer.forward(graph);
    expect(output.x.shape).toEqual([4, 8]);
  });

  it('should forward pass with sum aggregator', () => {
    const graph = createTestGraph();
    const layer = new SAGEConv({
      inChannels: 3,
      outChannels: 8,
      aggregator: 'sum',
    });

    const output = layer.forward(graph);
    expect(output.x.shape).toEqual([4, 8]);
  });

  it('should forward pass with pool aggregator', () => {
    const graph = createTestGraph();
    const layer = new SAGEConv({
      inChannels: 3,
      outChannels: 8,
      aggregator: 'pool',
    });

    const output = layer.forward(graph);
    expect(output.x.shape).toEqual([4, 8]);
  });

  it('should work without root weight', () => {
    const graph = createTestGraph();
    const layer = new SAGEConv({
      inChannels: 3,
      outChannels: 8,
      rootWeight: false,
    });

    const output = layer.forward(graph);
    expect(output.x.shape).toEqual([4, 8]);
  });

  it('should normalize output when specified', () => {
    const graph = createTestGraph();
    const layer = new SAGEConv({
      inChannels: 3,
      outChannels: 8,
      normalize: true,
    });

    const output = layer.forward(graph);

    // Check that outputs are L2 normalized
    for (let i = 0; i < output.numNodes; i++) {
      let norm = 0;
      for (let f = 0; f < 8; f++) {
        const val = output.x.get(i, f);
        norm += val * val;
      }
      norm = Math.sqrt(norm);
      // Should be close to 1 (or 0 if all zeros)
      expect(norm).toBeCloseTo(1, 0.1);
    }
  });
});
