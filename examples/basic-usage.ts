/**
 * BrowserGNN by Dr. Lee
 * Basic Usage Example
 *
 * Demonstrates how to create graphs and run GNN inference.
 */

import {
  GraphData,
  GCNConv,
  GATConv,
  SAGEConv,
  Sequential,
  ReLU,
  Softmax,
  Dropout,
  createBrowserGNN,
  fromEdgeList,
  randomGraph,
} from '../src';

async function main() {
  console.log('=== BrowserGNN by Dr. Lee ===\n');

  // Initialize BrowserGNN
  const { backend, info } = await createBrowserGNN();
  console.log(`Initialized with ${backend} backend`);
  console.log(info);
  console.log();

  // Example 1: Create a simple graph manually
  console.log('--- Example 1: Simple Graph ---');

  const graph = new GraphData({
    // Node features: 4 nodes, 3 features each
    x: new Float32Array([
      0.1, 0.2, 0.3, // Node 0
      0.4, 0.5, 0.6, // Node 1
      0.7, 0.8, 0.9, // Node 2
      1.0, 1.1, 1.2, // Node 3
    ]),
    numNodes: 4,
    numFeatures: 3,
    // Edge index: [source nodes..., target nodes...]
    edgeIndex: new Uint32Array([
      0, 0, 1, 1, 2, 2, 3, 3, // source nodes
      1, 3, 0, 2, 1, 3, 0, 2, // target nodes
    ]),
    numEdges: 8,
  });

  console.log(graph.toString());
  graph.print();
  console.log();

  // Example 2: Create graph from edge list
  console.log('--- Example 2: Graph from Edge List ---');

  const edges: [number, number][] = [
    [0, 1], [1, 0], // bidirectional edge
    [1, 2], [2, 1],
    [2, 3], [3, 2],
    [0, 3], [3, 0],
  ];

  const graphFromEdges = fromEdgeList(edges, 4);
  console.log(graphFromEdges.toString());
  console.log();

  // Example 3: GCN Layer
  console.log('--- Example 3: GCN Layer ---');

  const gcn = new GCNConv({
    inChannels: 3,
    outChannels: 16,
  });

  console.log(gcn.toString());
  console.log(`Parameters: ${gcn.numParameters()}`);

  const gcnOutput = gcn.forward(graph);
  console.log(`Input shape: [${graph.x.shape}]`);
  console.log(`Output shape: [${gcnOutput.x.shape}]`);
  console.log();

  // Example 4: GAT Layer with Multi-Head Attention
  console.log('--- Example 4: GAT Layer ---');

  const gat = new GATConv({
    inChannels: 3,
    outChannels: 8,
    heads: 2,
    concat: true, // Output will be 2 * 8 = 16
  });

  console.log(gat.toString());
  console.log(`Parameters: ${gat.numParameters()}`);

  const gatOutput = gat.forward(graph);
  console.log(`Input shape: [${graph.x.shape}]`);
  console.log(`Output shape: [${gatOutput.x.shape}]`);
  console.log();

  // Example 5: GraphSAGE Layer
  console.log('--- Example 5: GraphSAGE Layer ---');

  const sage = new SAGEConv({
    inChannels: 3,
    outChannels: 16,
    aggregator: 'mean',
  });

  console.log(sage.toString());
  console.log(`Parameters: ${sage.numParameters()}`);

  const sageOutput = sage.forward(graph);
  console.log(`Input shape: [${graph.x.shape}]`);
  console.log(`Output shape: [${sageOutput.x.shape}]`);
  console.log();

  // Example 6: Build a Complete GNN Model
  console.log('--- Example 6: Complete GNN Model ---');

  const model = new Sequential([
    new GCNConv({ inChannels: 3, outChannels: 16 }),
    new ReLU(),
    new Dropout(0.5),
    new GCNConv({ inChannels: 16, outChannels: 8 }),
    new ReLU(),
    new GCNConv({ inChannels: 8, outChannels: 2 }), // 2 classes
    new Softmax(),
  ]);

  // Set to evaluation mode (disables dropout)
  model.eval();

  console.log('Model architecture:');
  console.log(model.summary());
  console.log(`Total parameters: ${model.numParameters()}`);

  const predictions = await model.forward(graph);
  if (predictions instanceof GraphData) {
    console.log(`\nPredictions shape: [${predictions.x.shape}]`);
    console.log('Node class probabilities:');
    for (let i = 0; i < predictions.numNodes; i++) {
      const p0 = predictions.x.get(i, 0).toFixed(4);
      const p1 = predictions.x.get(i, 1).toFixed(4);
      console.log(`  Node ${i}: [${p0}, ${p1}]`);
    }
  }
  console.log();

  // Example 7: Random Graph
  console.log('--- Example 7: Random Graph ---');

  const largeGraph = randomGraph(100, 0.1, 32);
  console.log(largeGraph.toString());
  console.log(`Directed: ${largeGraph.isDirected()}`);
  console.log(`Has self-loops: ${largeGraph.hasSelfLoops()}`);

  // Run inference on large graph
  const largeModel = new Sequential([
    new GCNConv({ inChannels: 32, outChannels: 64 }),
    new ReLU(),
    new GCNConv({ inChannels: 64, outChannels: 10 }),
  ]);

  const start = performance.now();
  const largeOutput = await largeModel.forward(largeGraph);
  const elapsed = performance.now() - start;

  if (largeOutput instanceof GraphData) {
    console.log(`Output shape: [${largeOutput.x.shape}]`);
    console.log(`Inference time: ${elapsed.toFixed(2)}ms`);
  }

  console.log('\n=== Done ===');
}

// Run if executed directly
main().catch(console.error);
