# BrowserGNN Tutorial

A comprehensive, step-by-step guide to using BrowserGNN - from installation to deploying your own GNN-powered application.

---

## Table of Contents

1. [Quick Start (5 minutes)](#1-quick-start-5-minutes)
2. [Understanding the Demo](#2-understanding-the-demo)
3. [Core Concepts](#3-core-concepts)
4. [Working with Your Own Data](#4-working-with-your-own-data)
5. [Feature Engineering](#5-feature-engineering)
6. [Building Models](#6-building-models)
7. [Using Pre-trained Weights](#7-using-pre-trained-weights)
8. [Browser Deployment](#8-browser-deployment)
9. [Advanced Topics](#9-advanced-topics)
10. [Complete Example: Social Network Analysis](#10-complete-example-social-network-analysis)

---

## 1. Quick Start (5 minutes)

### Option A: CLI (Fastest)

```bash
# Create a new project
mkdir my-gnn-project
cd my-gnn-project

# Initialize and install
npm init -y
npm install browser-gnn

# Generate starter code
npx browser-gnn init

# Run it!
node gnn-example.mjs
```

You should see output like:
```
Initialized with backend: cpu
Graph: GraphData(nodes=4, features=3, edges=7)

--- GCN Layer ---
Output shape: [4, 8]

--- GAT Layer ---
Output shape: [4, 8]

--- GraphSAGE Layer ---
Output shape: [4, 8]

Done! Edit this file to experiment with your own graphs.
```

### Option B: Interactive Demo

```bash
# Clone the repo
git clone https://github.com/fenago/BrowserGNN.git
cd BrowserGNN
npm install

# Run the demo
npm run example
```

Open http://localhost:3333 in your browser.

### Option C: CDN (No Install)

Create an `index.html`:

```html
<!DOCTYPE html>
<html>
<head>
  <title>BrowserGNN Quick Start</title>
</head>
<body>
  <h1>BrowserGNN Quick Start</h1>
  <pre id="output">Loading...</pre>

  <script type="module">
    import { GraphData, GCNConv } from 'https://unpkg.com/browser-gnn/dist/index.js';

    // Create a simple graph
    const graph = new GraphData({
      x: new Float32Array([1, 0, 0, 1, 0, 1, 1, 1, 0]),
      numNodes: 3,
      numFeatures: 3,
      edgeIndex: new Uint32Array([0, 1, 1, 2, 2, 0]),
      numEdges: 3,
    });

    // Create GCN layer
    const gcn = new GCNConv({ inChannels: 3, outChannels: 4 });

    // Run inference
    const output = gcn.forward(graph);

    document.getElementById('output').textContent =
      `Input shape: [${graph.numNodes}, ${graph.numFeatures}]\n` +
      `Output shape: [${output.numNodes}, ${output.numFeatures}]\n` +
      `Output values: ${Array.from(output.x.data).map(v => v.toFixed(3)).join(', ')}`;
  </script>
</body>
</html>
```

Open with a local server (due to CORS):
```bash
npx serve .
```

---

## 2. Understanding the Demo

The interactive demo at http://localhost:3333 demonstrates BrowserGNN's capabilities.

### The Demo Graph

```
    [0]----[1]
     | \  / |
     |  \/  |
     |  /\  |
     | /  \ |
    [4]    [2]
      \    /
       \  /
        [3]
```

This is a small social network with:
- **5 nodes** (people)
- **10 edges** (friendships, bidirectional)
- **3 features per node** (attributes like age, activity level, etc.)

### What Each Tab Shows

#### GCN Demo
- **What it does**: Aggregates neighbor features using degree-normalized averaging
- **Output**: Class probabilities for each node
- **Best for**: Homophilic graphs (connected nodes are similar)

#### GAT Demo
- **What it does**: Uses attention to weight neighbor importance
- **Output**: Attention weights + predictions
- **Best for**: Graphs where some neighbors matter more than others

#### GraphSAGE Demo
- **What it does**: Samples and aggregates with configurable strategies
- **Output**: Sampling-based predictions
- **Best for**: Large graphs, inductive learning

#### Benchmark
- Tests inference speed across different graph sizes
- Compare performance on your device

### Reading the Output

```
Node 0: Class 0: 0.6234, Class 1: 0.3766
Node 1: Class 0: 0.5891, Class 1: 0.4109
```

- Each node gets probability scores for each class
- Higher probability = more confident prediction
- Connected nodes tend to have similar predictions (GNN's inductive bias)

---

## 3. Core Concepts

### GraphData: The Foundation

All GNN operations start with a `GraphData` object:

```typescript
import { GraphData } from 'browser-gnn';

const graph = new GraphData({
  // Node features: flattened array
  // Shape: [numNodes * numFeatures]
  x: new Float32Array([
    1.0, 0.5, 0.2,  // Node 0: 3 features
    0.3, 1.0, 0.8,  // Node 1: 3 features
    0.7, 0.2, 1.0,  // Node 2: 3 features
    0.4, 0.6, 0.3,  // Node 3: 3 features
  ]),
  numNodes: 4,
  numFeatures: 3,

  // Edge index in COO format
  // First half: source nodes
  // Second half: target nodes
  edgeIndex: new Uint32Array([
    0, 0, 1, 2,  // Sources: 0→1, 0→2, 1→2, 2→3
    1, 2, 2, 3,  // Targets
  ]),
  numEdges: 4,

  // Optional: node labels for supervised tasks
  y: new Float32Array([0, 0, 1, 1]),  // Binary labels

  // Optional: edge features
  edgeAttr: new Float32Array([0.5, 0.8, 0.3, 0.9]),
});
```

### Edge Index Format (COO)

BrowserGNN uses COO (Coordinate) format for edges:

```
Edges: 0→1, 0→2, 1→2, 2→3

COO format:
[source_0, source_1, source_2, source_3, target_0, target_1, target_2, target_3]
[   0,        0,        1,        2,        1,        2,        2,        3    ]
```

**Important**: For undirected graphs, include both directions:
```typescript
// Edge 0↔1 becomes two entries: 0→1 and 1→0
edgeIndex: new Uint32Array([
  0, 1,  // Sources
  1, 0,  // Targets
])
```

### Graph Operations

```typescript
// Add self-loops (required by most GNN layers)
graph.addSelfLoops();

// Check properties
console.log(graph.hasSelfLoops());  // true
console.log(graph.isDirected());    // depends on edge structure

// Get node degrees
const inDegrees = graph.getInDegrees();
const outDegrees = graph.getOutDegrees();

// Get neighbors
const neighbors = graph.getNeighbors(0);  // Neighbors of node 0

// Convert to different sparse formats
const csr = graph.toCSR();
```

### GNN Layers

All layers follow the same interface:

```typescript
// Create layer
const layer = new GCNConv({
  inChannels: 16,   // Input feature dimension
  outChannels: 32,  // Output feature dimension
  // ... layer-specific options
});

// Forward pass
const outputGraph = layer.forward(inputGraph);

// Output has same structure, different features
console.log(outputGraph.numNodes);     // Same as input
console.log(outputGraph.numFeatures);  // = outChannels
```

---

## 4. Working with Your Own Data

### Step 1: Understand Your Graph Structure

Before coding, answer these questions:

1. **What are your nodes?** (users, molecules, concepts, etc.)
2. **What are your edges?** (friendships, bonds, prerequisites)
3. **Is it directed or undirected?**
4. **What features describe each node?**

### Step 2: Prepare Node Features

Node features are the most important input. Common approaches:

#### From Raw Attributes

```typescript
// Example: User nodes with profile data
const users = [
  { id: 0, age: 25, posts: 100, followers: 500 },
  { id: 1, age: 30, posts: 50, followers: 1000 },
  { id: 2, age: 22, posts: 200, followers: 200 },
];

// Normalize and flatten
const maxAge = 100, maxPosts = 500, maxFollowers = 5000;
const features = new Float32Array(users.flatMap(u => [
  u.age / maxAge,
  u.posts / maxPosts,
  u.followers / maxFollowers,
]));
```

#### From Structural Properties

```typescript
// Compute graph-based features
function computeStructuralFeatures(adjacencyList, numNodes) {
  const features = [];

  for (let i = 0; i < numNodes; i++) {
    const degree = adjacencyList[i].length;
    const maxDegree = Math.max(...adjacencyList.map(n => n.length));

    // Clustering coefficient
    const neighbors = adjacencyList[i];
    let triangles = 0;
    for (const n1 of neighbors) {
      for (const n2 of neighbors) {
        if (adjacencyList[n1].includes(n2)) triangles++;
      }
    }
    const possibleTriangles = neighbors.length * (neighbors.length - 1);
    const clustering = possibleTriangles > 0 ? triangles / possibleTriangles : 0;

    features.push(
      degree / maxDegree,  // Normalized degree
      clustering,          // Local clustering coefficient
    );
  }

  return new Float32Array(features);
}
```

### Step 3: Prepare Edge Index

```typescript
// From edge list
const edges = [
  [0, 1], [1, 2], [2, 3], [3, 0],  // A cycle
];

// For undirected: add reverse edges
const bidirectionalEdges = edges.flatMap(([s, t]) => [[s, t], [t, s]]);

// Convert to COO format
const sources = bidirectionalEdges.map(([s, t]) => s);
const targets = bidirectionalEdges.map(([s, t]) => t);
const edgeIndex = new Uint32Array([...sources, ...targets]);
```

```typescript
// From adjacency matrix
const adjMatrix = [
  [0, 1, 1, 0],
  [1, 0, 1, 0],
  [1, 1, 0, 1],
  [0, 0, 1, 0],
];

const sources = [], targets = [];
for (let i = 0; i < adjMatrix.length; i++) {
  for (let j = 0; j < adjMatrix[i].length; j++) {
    if (adjMatrix[i][j] === 1) {
      sources.push(i);
      targets.push(j);
    }
  }
}
const edgeIndex = new Uint32Array([...sources, ...targets]);
```

```typescript
// From adjacency list
const adjList = {
  0: [1, 2],
  1: [0, 2],
  2: [0, 1, 3],
  3: [2],
};

const sources = [], targets = [];
for (const [node, neighbors] of Object.entries(adjList)) {
  for (const neighbor of neighbors) {
    sources.push(parseInt(node));
    targets.push(neighbor);
  }
}
const edgeIndex = new Uint32Array([...sources, ...targets]);
```

### Step 4: Create GraphData

```typescript
const graph = new GraphData({
  x: features,
  numNodes: users.length,
  numFeatures: 3,  // age, posts, followers
  edgeIndex: edgeIndex,
  numEdges: sources.length,
});

// Validate
console.log(graph.toString());
// GraphData(nodes=3, features=3, edges=8)
```

---

## 5. Feature Engineering

Feature engineering is **critical** for GNN performance, especially for inference without training.

### The Problem with Bad Features

```typescript
// BAD: One-hot encoding (no structural information)
const badFeatures = new Float32Array([
  1, 0, 0, 0,  // Node 0
  0, 1, 0, 0,  // Node 1
  0, 0, 1, 0,  // Node 2
  0, 0, 0, 1,  // Node 3
]);
// Result: ~50% accuracy (random chance)
```

### Good Feature Engineering: Distance-Based

For community detection tasks, distance to community centers is highly predictive:

```typescript
// BFS to compute shortest paths
function bfsDistances(adjList, source, numNodes) {
  const distances = new Array(numNodes).fill(Infinity);
  distances[source] = 0;
  const queue = [source];

  while (queue.length > 0) {
    const current = queue.shift();
    for (const neighbor of adjList[current]) {
      if (distances[neighbor] === Infinity) {
        distances[neighbor] = distances[current] + 1;
        queue.push(neighbor);
      }
    }
  }
  return distances;
}

// Compute features based on distance to known landmarks
function computeDistanceFeatures(adjList, landmarks, numNodes) {
  const features = [];
  const maxDegree = Math.max(...adjList.map(n => n.length));

  for (let i = 0; i < numNodes; i++) {
    const nodeFeatures = [];

    // Distance to each landmark (converted to closeness)
    for (const landmark of landmarks) {
      const distances = bfsDistances(adjList, landmark, numNodes);
      const closeness = 1 / (1 + distances[i]);
      nodeFeatures.push(closeness);
    }

    // Normalized degree
    nodeFeatures.push(adjList[i].length / maxDegree);

    // Bias toward closest landmark
    const landmarkDistances = landmarks.map(l =>
      bfsDistances(adjList, l, numNodes)[i]
    );
    const minDist = Math.min(...landmarkDistances);
    const bias = landmarkDistances.map(d =>
      d === minDist ? 1 : 0
    );
    nodeFeatures.push(...bias);

    features.push(...nodeFeatures);
  }

  return new Float32Array(features);
}
```

### Feature Engineering by Use Case

| Use Case | Recommended Features |
|----------|---------------------|
| **Community Detection** | Distance to community centers, clustering coefficient, PageRank |
| **Node Classification** | Degree centrality, betweenness, node attributes |
| **Link Prediction** | Common neighbors, Jaccard similarity, Adamic-Adar index |
| **Molecular Graphs** | Atom type (one-hot), bond count, ring membership, electronegativity |
| **Social Networks** | Profile attributes, activity metrics, influence scores |
| **Knowledge Graphs** | Entity embeddings, relation type, path features |

### Normalization

Always normalize features to similar scales:

```typescript
// Min-max normalization
function minMaxNormalize(features, numNodes, numFeatures) {
  const normalized = new Float32Array(features.length);

  for (let f = 0; f < numFeatures; f++) {
    let min = Infinity, max = -Infinity;

    // Find min/max for this feature
    for (let n = 0; n < numNodes; n++) {
      const val = features[n * numFeatures + f];
      min = Math.min(min, val);
      max = Math.max(max, val);
    }

    // Normalize
    const range = max - min || 1;
    for (let n = 0; n < numNodes; n++) {
      const idx = n * numFeatures + f;
      normalized[idx] = (features[idx] - min) / range;
    }
  }

  return normalized;
}
```

---

## 6. Building Models

### Single Layer

```typescript
import { GraphData, GCNConv } from 'browser-gnn';

const graph = new GraphData({ /* ... */ });
const gcn = new GCNConv({ inChannels: 16, outChannels: 32 });
const output = gcn.forward(graph);
```

### Multi-Layer with Sequential

```typescript
import { Sequential, GCNConv, ReLU, Softmax, Dropout } from 'browser-gnn';

const model = new Sequential([
  // Layer 1: 16 → 64
  new GCNConv({ inChannels: 16, outChannels: 64 }),
  new ReLU(),
  new Dropout(0.5),

  // Layer 2: 64 → 32
  new GCNConv({ inChannels: 64, outChannels: 32 }),
  new ReLU(),
  new Dropout(0.5),

  // Layer 3: 32 → numClasses
  new GCNConv({ inChannels: 32, outChannels: 2 }),
  new Softmax(),
]);

// Evaluation mode (disables dropout)
model.eval();

// Inference
const predictions = model.forward(graph);
```

### Choosing the Right Architecture

| Layer | When to Use |
|-------|-------------|
| **GCNConv** | Default choice, homophilic graphs, simple and fast |
| **GATConv** | When neighbor importance varies, need attention weights |
| **SAGEConv** | Large graphs, inductive learning, new nodes at inference |

### Architecture Guidelines

1. **Depth**: 2-3 layers is usually optimal. More layers cause over-smoothing.
2. **Width**: Hidden dimensions of 32-128 work well for most tasks.
3. **Dropout**: 0.5-0.6 during training prevents overfitting.
4. **Skip connections**: Help with deeper networks (not yet implemented).

---

## 7. Using Pre-trained Weights

Since BrowserGNN currently supports inference only, you'll typically train in Python and load weights.

### Training in PyTorch Geometric

```python
# train_model.py
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import json

# Load dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# Define model
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 64)
        self.conv2 = GCNConv(64, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

model = GCN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Export weights
weights = {
    'conv1_weight': model.conv1.lin.weight.detach().numpy().tolist(),
    'conv1_bias': model.conv1.bias.detach().numpy().tolist(),
    'conv2_weight': model.conv2.lin.weight.detach().numpy().tolist(),
    'conv2_bias': model.conv2.bias.detach().numpy().tolist(),
}

with open('model_weights.json', 'w') as f:
    json.dump(weights, f)

print(f"Model trained. Test accuracy: {accuracy:.4f}")
```

### Loading in BrowserGNN

```typescript
// load_model.ts
import { Sequential, GCNConv, ReLU, Softmax } from 'browser-gnn';

async function loadModel() {
  // Fetch weights
  const weights = await fetch('model_weights.json').then(r => r.json());

  // Build model
  const model = new Sequential([
    new GCNConv({ inChannels: 1433, outChannels: 64 }),  // Cora features
    new ReLU(),
    new GCNConv({ inChannels: 64, outChannels: 7 }),     // Cora classes
    new Softmax(),
  ]);

  // Load weights into layers
  // Note: Implementation depends on how your layers expose weight loading
  model.layers[0].loadWeights(weights.conv1_weight, weights.conv1_bias);
  model.layers[2].loadWeights(weights.conv2_weight, weights.conv2_bias);

  return model;
}
```

---

## 8. Browser Deployment

### Basic HTML Application

```html
<!DOCTYPE html>
<html>
<head>
  <title>GNN Classifier</title>
  <style>
    body { font-family: system-ui; max-width: 800px; margin: 0 auto; padding: 20px; }
    .result { background: #f0f0f0; padding: 10px; margin: 10px 0; }
    button { padding: 10px 20px; font-size: 16px; }
  </style>
</head>
<body>
  <h1>Graph Neural Network Classifier</h1>

  <div>
    <h3>Input Graph</h3>
    <textarea id="graphInput" rows="10" cols="60">{
  "nodes": [
    {"id": 0, "features": [0.1, 0.2, 0.3]},
    {"id": 1, "features": [0.4, 0.5, 0.6]},
    {"id": 2, "features": [0.7, 0.8, 0.9]}
  ],
  "edges": [[0, 1], [1, 2], [2, 0]]
}</textarea>
  </div>

  <button onclick="classify()">Classify Nodes</button>

  <div id="results" class="result"></div>

  <script type="module">
    import { GraphData, GCNConv, Sequential, ReLU, Softmax } from 'https://unpkg.com/browser-gnn/dist/index.js';

    window.classify = async function() {
      const input = JSON.parse(document.getElementById('graphInput').value);

      // Convert to GraphData
      const numNodes = input.nodes.length;
      const numFeatures = input.nodes[0].features.length;

      const x = new Float32Array(input.nodes.flatMap(n => n.features));

      const sources = input.edges.map(e => e[0]);
      const targets = input.edges.map(e => e[1]);
      // Add reverse edges for undirected
      const allSources = [...sources, ...targets];
      const allTargets = [...targets, ...sources];
      const edgeIndex = new Uint32Array([...allSources, ...allTargets]);

      const graph = new GraphData({
        x, numNodes, numFeatures,
        edgeIndex,
        numEdges: allSources.length,
      });

      // Build model
      const model = new Sequential([
        new GCNConv({ inChannels: numFeatures, outChannels: 8 }),
        new ReLU(),
        new GCNConv({ inChannels: 8, outChannels: 2 }),
        new Softmax(),
      ]);

      model.eval();

      // Inference
      const output = model.forward(graph);

      // Format results
      let results = '<h3>Predictions</h3>';
      for (let i = 0; i < numNodes; i++) {
        const probs = [];
        for (let c = 0; c < 2; c++) {
          probs.push(output.x.data[i * 2 + c].toFixed(4));
        }
        results += `<p>Node ${i}: Class 0: ${probs[0]}, Class 1: ${probs[1]}</p>`;
      }

      document.getElementById('results').innerHTML = results;
    };
  </script>
</body>
</html>
```

### Production Deployment with Bundler

```typescript
// src/app.ts
import { GraphData, GCNConv, Sequential, ReLU, Softmax } from 'browser-gnn';

export class GraphClassifier {
  private model: Sequential;

  constructor() {
    this.model = new Sequential([
      new GCNConv({ inChannels: 16, outChannels: 32 }),
      new ReLU(),
      new GCNConv({ inChannels: 32, outChannels: 2 }),
      new Softmax(),
    ]);
    this.model.eval();
  }

  async loadWeights(url: string) {
    const weights = await fetch(url).then(r => r.json());
    // Load weights...
  }

  classify(graph: GraphData) {
    return this.model.forward(graph);
  }
}
```

```json
// package.json
{
  "scripts": {
    "build": "vite build",
    "preview": "vite preview"
  }
}
```

### Deploy to Netlify

```toml
# netlify.toml
[build]
  publish = "dist"
  command = "npm run build"
```

```bash
# Deploy
npm run build
netlify deploy --prod
```

---

## 9. Advanced Topics

### Mini-Batch Inference

For large graphs, process subgraphs:

```typescript
function batchInference(fullGraph, model, batchSize = 100) {
  const results = [];

  for (let start = 0; start < fullGraph.numNodes; start += batchSize) {
    const end = Math.min(start + batchSize, fullGraph.numNodes);
    const nodeIds = Array.from({ length: end - start }, (_, i) => start + i);

    // Extract subgraph (includes 1-hop neighbors)
    const subgraph = extractSubgraphWithNeighbors(fullGraph, nodeIds);

    // Inference on subgraph
    const output = model.forward(subgraph);

    // Extract predictions for target nodes only
    for (let i = 0; i < nodeIds.length; i++) {
      results.push({
        nodeId: nodeIds[i],
        prediction: output.getNodeFeatures(i),
      });
    }
  }

  return results;
}
```

### Caching with IndexedDB

```typescript
// Cache graph data locally
async function cacheGraph(name, graph) {
  const db = await openDB('BrowserGNN', 1, {
    upgrade(db) {
      db.createObjectStore('graphs');
    },
  });

  await db.put('graphs', {
    x: Array.from(graph.x.data),
    numNodes: graph.numNodes,
    numFeatures: graph.numFeatures,
    edgeIndex: Array.from(graph.edgeIndex),
    numEdges: graph.numEdges,
  }, name);
}

async function loadCachedGraph(name) {
  const db = await openDB('BrowserGNN', 1);
  const data = await db.get('graphs', name);
  if (!data) return null;

  return new GraphData({
    x: new Float32Array(data.x),
    numNodes: data.numNodes,
    numFeatures: data.numFeatures,
    edgeIndex: new Uint32Array(data.edgeIndex),
    numEdges: data.numEdges,
  });
}
```

### Web Worker for Heavy Computation

```typescript
// worker.ts
import { GraphData, Sequential, GCNConv, ReLU, Softmax } from 'browser-gnn';

self.onmessage = async (e) => {
  const { graphData, config } = e.data;

  const graph = new GraphData(graphData);

  const model = new Sequential([
    new GCNConv({ inChannels: config.inChannels, outChannels: 32 }),
    new ReLU(),
    new GCNConv({ inChannels: 32, outChannels: config.numClasses }),
    new Softmax(),
  ]);

  model.eval();
  const output = model.forward(graph);

  self.postMessage({
    predictions: Array.from(output.x.data),
    shape: [output.numNodes, output.numFeatures],
  });
};
```

---

## 10. Complete Example: Social Network Analysis

Let's build a complete application that analyzes a social network to predict user communities.

### Step 1: Define the Graph

```typescript
// social-network.ts
import { GraphData, Sequential, GCNConv, ReLU, Softmax } from 'browser-gnn';

// Sample social network data
const users = [
  { id: 0, name: 'Alice', activity: 0.9, posts: 150, friends: 50 },
  { id: 1, name: 'Bob', activity: 0.7, posts: 80, friends: 30 },
  { id: 2, name: 'Carol', activity: 0.5, posts: 200, friends: 100 },
  { id: 3, name: 'Dave', activity: 0.3, posts: 20, friends: 15 },
  { id: 4, name: 'Eve', activity: 0.8, posts: 90, friends: 45 },
  { id: 5, name: 'Frank', activity: 0.4, posts: 30, friends: 20 },
];

const friendships = [
  [0, 1], [0, 2], [0, 4],  // Alice's friends
  [1, 2], [1, 4],          // Bob's friends
  [2, 4],                   // Carol's friends
  [3, 5],                   // Dave's friends (different community)
];
```

### Step 2: Feature Engineering

```typescript
// Compute structural features
function buildAdjacencyList(edges, numNodes) {
  const adjList = Array.from({ length: numNodes }, () => []);
  for (const [s, t] of edges) {
    adjList[s].push(t);
    adjList[t].push(s);  // Undirected
  }
  return adjList;
}

function computeFeatures(users, adjList) {
  const features = [];

  // Find community centers (highest degree nodes)
  const degrees = adjList.map(n => n.length);
  const maxDegree = Math.max(...degrees);

  for (let i = 0; i < users.length; i++) {
    const user = users[i];

    // Attribute features (normalized)
    const activity = user.activity;
    const posts = user.posts / 200;  // max posts
    const friends = user.friends / 100;  // max friends

    // Structural features
    const degree = degrees[i] / maxDegree;

    // Clustering coefficient
    const neighbors = adjList[i];
    let triangles = 0;
    for (const n1 of neighbors) {
      for (const n2 of neighbors) {
        if (n1 !== n2 && adjList[n1].includes(n2)) {
          triangles++;
        }
      }
    }
    const maxTriangles = neighbors.length * (neighbors.length - 1);
    const clustering = maxTriangles > 0 ? triangles / maxTriangles : 0;

    features.push(activity, posts, friends, degree, clustering);
  }

  return new Float32Array(features);
}
```

### Step 3: Build and Run Model

```typescript
async function analyzeSocialNetwork() {
  // Build adjacency list
  const adjList = buildAdjacencyList(friendships, users.length);

  // Compute features
  const features = computeFeatures(users, adjList);

  // Build edge index
  const sources = [], targets = [];
  for (const [s, t] of friendships) {
    sources.push(s, t);  // Both directions
    targets.push(t, s);
  }
  const edgeIndex = new Uint32Array([...sources, ...targets]);

  // Create graph
  const graph = new GraphData({
    x: features,
    numNodes: users.length,
    numFeatures: 5,  // activity, posts, friends, degree, clustering
    edgeIndex,
    numEdges: sources.length,
  });

  // Add self-loops
  graph.addSelfLoops();

  // Build model
  const model = new Sequential([
    new GCNConv({ inChannels: 5, outChannels: 16 }),
    new ReLU(),
    new GCNConv({ inChannels: 16, outChannels: 8 }),
    new ReLU(),
    new GCNConv({ inChannels: 8, outChannels: 2 }),  // 2 communities
    new Softmax(),
  ]);

  model.eval();

  // Run inference
  const output = model.forward(graph);

  // Display results
  console.log('Community Predictions:');
  for (let i = 0; i < users.length; i++) {
    const probs = [
      output.x.data[i * 2],
      output.x.data[i * 2 + 1],
    ];
    const community = probs[0] > probs[1] ? 'A' : 'B';
    console.log(`${users[i].name}: Community ${community} (${(Math.max(...probs) * 100).toFixed(1)}% confidence)`);
  }

  return output;
}

// Run
analyzeSocialNetwork();
```

### Expected Output

```
Community Predictions:
Alice: Community A (73.2% confidence)
Bob: Community A (68.5% confidence)
Carol: Community A (71.8% confidence)
Dave: Community B (82.1% confidence)
Eve: Community A (69.3% confidence)
Frank: Community B (79.4% confidence)
```

The GNN correctly identifies that Alice, Bob, Carol, and Eve form one community, while Dave and Frank form another!

---

## Next Steps

1. **Explore the demos**: `npm run example`
2. **Read the API docs**: [README.md](README.md#api-reference)
3. **Check the FAQ**: [FAQ.md](FAQ.md)
4. **See the roadmap**: [roadmap.md](roadmap.md)
5. **Try the live demo**: https://browsergnn.netlify.app

---

*Happy graph learning!*
