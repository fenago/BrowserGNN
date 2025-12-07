# Using BrowserGNN with Your Own Data

This guide shows how to convert your data into BrowserGNN format.

## Understanding the Data Format

BrowserGNN uses two main data structures:

1. **Node features** (`x`): A flat `Float32Array` of shape `[numNodes * numFeatures]`
2. **Edge index** (`edgeIndex`): A `Uint32Array` of shape `[2 * numEdges]` containing `[source nodes..., target nodes...]`

## Example 1: Social Network

Say you have a social network with 4 users and their connections:

```
Users: Alice(0), Bob(1), Carol(2), Dave(3)
Connections: Alice-Bob, Bob-Carol, Carol-Dave, Alice-Carol
Features: [age, follower_count]
```

```javascript
import { GraphData, GCNConv } from 'browser-gnn';

// User features: [age, follower_count] (normalized)
const features = new Float32Array([
  0.25, 0.1,   // Alice: age=25, 100 followers
  0.30, 0.5,   // Bob: age=30, 500 followers
  0.22, 0.3,   // Carol: age=22, 300 followers
  0.35, 0.8    // Dave: age=35, 800 followers
]);

// Edges (bidirectional)
// Alice-Bob, Bob-Alice, Bob-Carol, Carol-Bob, Carol-Dave, Dave-Carol, Alice-Carol, Carol-Alice
const edgeIndex = new Uint32Array([
  0, 1, 1, 2, 2, 3, 0, 2,  // Source nodes
  1, 0, 2, 1, 3, 2, 2, 0   // Target nodes
]);

const graph = new GraphData({
  x: features,
  numNodes: 4,
  numFeatures: 2,
  edgeIndex: edgeIndex,
  numEdges: 8
});

// Run GNN
const gcn = new GCNConv({ inChannels: 2, outChannels: 4 });
const output = gcn.forward(graph);
console.log('Node embeddings:', output.x.data);
```

## Example 2: Converting from Adjacency List

```javascript
// Your data as adjacency list
const adjacencyList = {
  0: [1, 2],      // Node 0 connects to 1 and 2
  1: [0, 2, 3],   // Node 1 connects to 0, 2, and 3
  2: [0, 1],      // etc.
  3: [1]
};

const nodeFeatures = {
  0: [1.0, 0.5],
  1: [0.8, 0.3],
  2: [0.2, 0.9],
  3: [0.6, 0.4]
};

// Convert to BrowserGNN format
function convertToBrowserGNN(adjList, features) {
  const numNodes = Object.keys(adjList).length;
  const numFeatures = Object.values(features)[0].length;

  // Build edge index
  const sources = [];
  const targets = [];
  for (const [src, neighbors] of Object.entries(adjList)) {
    for (const dst of neighbors) {
      sources.push(parseInt(src));
      targets.push(dst);
    }
  }

  // Flatten features
  const x = new Float32Array(numNodes * numFeatures);
  for (let i = 0; i < numNodes; i++) {
    for (let f = 0; f < numFeatures; f++) {
      x[i * numFeatures + f] = features[i][f];
    }
  }

  return new GraphData({
    x: x,
    numNodes: numNodes,
    numFeatures: numFeatures,
    edgeIndex: new Uint32Array([...sources, ...targets]),
    numEdges: sources.length
  });
}

const graph = convertToBrowserGNN(adjacencyList, nodeFeatures);
```

## Example 3: Converting from Edge List (CSV)

```javascript
// Suppose you have edges.csv:
// source,target
// 0,1
// 1,2
// 2,0

async function loadFromCSV(edgesCSV, featuresCSV) {
  // Parse edges
  const edgeLines = edgesCSV.trim().split('\n').slice(1); // Skip header
  const sources = [];
  const targets = [];

  for (const line of edgeLines) {
    const [src, dst] = line.split(',').map(Number);
    // Add both directions for undirected graph
    sources.push(src, dst);
    targets.push(dst, src);
  }

  // Parse features (nodes.csv: node_id,feat1,feat2,...)
  const featureLines = featuresCSV.trim().split('\n').slice(1);
  const numNodes = featureLines.length;
  const numFeatures = featureLines[0].split(',').length - 1;

  const x = new Float32Array(numNodes * numFeatures);
  for (const line of featureLines) {
    const parts = line.split(',').map(Number);
    const nodeId = parts[0];
    for (let f = 0; f < numFeatures; f++) {
      x[nodeId * numFeatures + f] = parts[f + 1];
    }
  }

  return new GraphData({
    x: x,
    numNodes: numNodes,
    numFeatures: numFeatures,
    edgeIndex: new Uint32Array([...sources, ...targets]),
    numEdges: sources.length
  });
}
```

## Example 4: Molecular Graph (Atoms & Bonds)

```javascript
// Caffeine molecule: C8H10N4O2
// Atoms: C, N, O with features [atomic_number, is_aromatic, num_hydrogens]

const atoms = [
  { element: 'C', aromatic: 1, hydrogens: 0 },  // 0
  { element: 'N', aromatic: 1, hydrogens: 0 },  // 1
  { element: 'C', aromatic: 1, hydrogens: 1 },  // 2
  { element: 'N', aromatic: 1, hydrogens: 0 },  // 3
  { element: 'C', aromatic: 1, hydrogens: 0 },  // 4
  { element: 'C', aromatic: 1, hydrogens: 0 },  // 5
  { element: 'N', aromatic: 0, hydrogens: 0 },  // 6 (methyl N)
  { element: 'C', aromatic: 0, hydrogens: 3 },  // 7 (methyl C)
  { element: 'O', aromatic: 0, hydrogens: 0 },  // 8
  { element: 'N', aromatic: 0, hydrogens: 0 },  // 9
  { element: 'C', aromatic: 0, hydrogens: 3 },  // 10 (methyl C)
  { element: 'O', aromatic: 0, hydrogens: 0 },  // 11
];

const atomicNumbers = { 'C': 6, 'N': 7, 'O': 8 };

// Create features
const numFeatures = 3;
const x = new Float32Array(atoms.length * numFeatures);
atoms.forEach((atom, i) => {
  x[i * numFeatures + 0] = atomicNumbers[atom.element] / 10; // Normalized
  x[i * numFeatures + 1] = atom.aromatic;
  x[i * numFeatures + 2] = atom.hydrogens / 3; // Normalized
});

// Bonds (simplified)
const bonds = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,0], [4,6], [6,7], [5,8], [0,9], [9,10], [5,11]];

const sources = [];
const targets = [];
bonds.forEach(([a, b]) => {
  sources.push(a, b);
  targets.push(b, a);
});

const molecule = new GraphData({
  x: x,
  numNodes: atoms.length,
  numFeatures: numFeatures,
  edgeIndex: new Uint32Array([...sources, ...targets]),
  numEdges: sources.length
});

// Predict atom properties
const gcn = new GCNConv({ inChannels: 3, outChannels: 8 });
const embeddings = gcn.forward(molecule);
```

## Example 5: Knowledge Graph

```javascript
// Entities with text embedding features (e.g., from a pre-trained model)
const entities = {
  0: { name: 'Albert Einstein', embedding: [0.1, 0.9, 0.3, 0.7] },
  1: { name: 'Physics', embedding: [0.8, 0.2, 0.6, 0.4] },
  2: { name: 'Nobel Prize', embedding: [0.5, 0.5, 0.8, 0.2] },
  3: { name: 'Germany', embedding: [0.3, 0.7, 0.1, 0.9] }
};

// Relations: (Einstein)-[studied]->(Physics), (Einstein)-[won]->(Nobel), (Einstein)-[born_in]->(Germany)
const relations = [
  [0, 1], [0, 2], [0, 3]
];

const numNodes = Object.keys(entities).length;
const numFeatures = 4;

const x = new Float32Array(numNodes * numFeatures);
for (const [id, entity] of Object.entries(entities)) {
  for (let f = 0; f < numFeatures; f++) {
    x[parseInt(id) * numFeatures + f] = entity.embedding[f];
  }
}

const sources = [];
const targets = [];
relations.forEach(([s, t]) => {
  sources.push(s, t);
  targets.push(t, s);
});

const knowledgeGraph = new GraphData({
  x: x,
  numNodes: numNodes,
  numFeatures: numFeatures,
  edgeIndex: new Uint32Array([...sources, ...targets]),
  numEdges: sources.length
});
```

## Feature Engineering Tips

For best results, use **structural features** computed from graph topology:

```javascript
// Compute BFS distances to landmark nodes
function bfsDistances(adjList, source, numNodes) {
  const dist = new Array(numNodes).fill(Infinity);
  dist[source] = 0;
  const queue = [source];

  while (queue.length > 0) {
    const curr = queue.shift();
    for (const neighbor of adjList[curr] || []) {
      if (dist[neighbor] === Infinity) {
        dist[neighbor] = dist[curr] + 1;
        queue.push(neighbor);
      }
    }
  }
  return dist;
}

// Compute degree for each node
function computeDegrees(adjList, numNodes) {
  const degrees = new Array(numNodes).fill(0);
  for (const [node, neighbors] of Object.entries(adjList)) {
    degrees[parseInt(node)] = neighbors.length;
  }
  return degrees;
}

// Create structural features
function createStructuralFeatures(adjList, landmarks) {
  const numNodes = Object.keys(adjList).length;
  const degrees = computeDegrees(adjList, numNodes);
  const maxDegree = Math.max(...degrees);

  // Features: [closeness_to_landmark1, closeness_to_landmark2, ..., normalized_degree]
  const numFeatures = landmarks.length + 1;
  const x = new Float32Array(numNodes * numFeatures);

  const landmarkDistances = landmarks.map(l => bfsDistances(adjList, l, numNodes));

  for (let i = 0; i < numNodes; i++) {
    // Closeness to each landmark (1 / (1 + distance))
    landmarks.forEach((_, j) => {
      x[i * numFeatures + j] = 1 / (1 + landmarkDistances[j][i]);
    });
    // Normalized degree
    x[i * numFeatures + landmarks.length] = degrees[i] / maxDegree;
  }

  return { x, numFeatures };
}

// Usage
const landmarks = [0, 10]; // Important nodes in your graph
const { x, numFeatures } = createStructuralFeatures(adjacencyList, landmarks);
```

## Common Patterns

### Stacking Multiple Layers

```javascript
const gcn1 = new GCNConv({ inChannels: inputDim, outChannels: 64 });
const gcn2 = new GCNConv({ inChannels: 64, outChannels: 32 });
const gcn3 = new GCNConv({ inChannels: 32, outChannels: outputDim });

let h = graph;
h = gcn1.forward(h);  // Apply ReLU manually if needed
h = gcn2.forward(h);
h = gcn3.forward(h);
```

### Getting Node Embeddings

```javascript
const output = gcn.forward(graph);

// Get embedding for node 0
const nodeEmbedding = output.x.data.slice(0, outputDim);

// Get all embeddings as 2D array
const embeddings = [];
for (let i = 0; i < output.numNodes; i++) {
  embeddings.push(Array.from(output.x.data.slice(i * outputDim, (i + 1) * outputDim)));
}
```

## Links

- [Getting Started](getting_started.md)
- [Live Demo](https://browsergnn.com)
- [GitHub](https://github.com/fenago/BrowserGNN)
- [npm Package](https://www.npmjs.com/package/browser-gnn)
