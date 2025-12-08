# BrowserGNN by Dr. Lee

<div align="center">

**The World's First Comprehensive Graph Neural Network Library for the Browser**

[![npm version](https://img.shields.io/npm/v/browser-gnn.svg)](https://www.npmjs.com/package/browser-gnn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.3-blue.svg)](https://www.typescriptlang.org/)

[Live Demo](https://browsergnn.com) | [npm Package](https://www.npmjs.com/package/browser-gnn) | [GitHub](https://github.com/fenago/BrowserGNN) | [Tutorial](./tutorial.md) | [FAQ](./FAQ.md) | [Roadmap](./roadmap.md)

</div>

---

## What is BrowserGNN?

**BrowserGNN** is a groundbreaking JavaScript/TypeScript library that brings the power of Graph Neural Networks (GNNs) directly to web browsers. Until now, GNN inference required server-side computation with frameworks like PyTorch Geometric or DGL. BrowserGNN changes this paradigm by enabling **client-side graph learning** with zero server dependencies.

### Why is this Novel?

| Traditional GNN | BrowserGNN |
|----------------|------------|
| Requires Python backend | Pure JavaScript/TypeScript |
| Server-side inference | Client-side execution |
| Data leaves user's device | Data stays local (privacy!) |
| Network latency | Instant inference |
| Server costs | Zero infrastructure |
| Limited scalability | Scales with user devices |

**This is the first library in the world to provide:**
- Complete GCN, GAT, GraphSAGE, and GIN implementations in pure TypeScript
- Full training support with autograd, optimizers, and loss functions
- Pre-built education-focused models (Model Zoo)
- Model serialization for save/load capabilities
- Browser-native tensor operations optimized for graph computations
- Sparse matrix operations (COO, CSR formats) for efficient graph processing
- A PyTorch Geometric-inspired API that ML engineers will find familiar

### Evidence: No Prior Browser-Based GNN Library Exists

We conducted extensive research to verify BrowserGNN's novelty:

| Existing Solution | GNN Support in Browser? | Notes |
|-------------------|------------------------|-------|
| [TensorFlow.js](https://www.tensorflow.org/js) | ❌ No | [Open feature request since 2022](https://github.com/tensorflow/tfjs/issues/5975), still unimplemented |
| [TF-GNN](https://github.com/tensorflow/gnn) | ❌ Python only | Requires TensorFlow 2.12+, Linux only |
| [PyTorch Geometric](https://pyg.org/) | ❌ Python/CUDA only | No JavaScript port exists |
| [Spektral](https://graphneural.network/) | ❌ Python only | Keras/TensorFlow, server-side |
| [Brain.js](https://brain.js.org/) | ❌ No GNN layers | Standard neural networks only |
| [ONNX Runtime Web](https://onnxruntime.ai/) | ❌ No GNN ops | General inference, no graph convolutions |

**The gap is clear**: The Python GNN ecosystem (PyTorch Geometric, DGL, TF-GNN) is mature, but the browser ML ecosystem (TensorFlow.js, ONNX Web, WebLLM) has lacked graph neural network support entirely—until now.

---

## Features

- **Pure Browser Execution** - No server required, runs entirely client-side
- **Privacy Preserving** - Graph data never leaves the user's device
- **Full Training Support** - Autograd, optimizers (SGD, Adam, Adagrad, RMSprop), and loss functions
- **Model Zoo** - Pre-built education-focused models for learning analytics
- **Model Serialization** - Save and load models to JSON or browser storage
- **WASM Optimized** - 8x loop-unrolled kernels for high performance
- **WebGPU Ready** - GPU-accelerated inference via compute shaders
- **Developer Friendly** - Familiar PyTorch Geometric-inspired API
- **Fully Typed** - Complete TypeScript definitions

## Documentation

| Document | Description |
|----------|-------------|
| **[Tutorial](./tutorial.md)** | Step-by-step guide from installation to deployment with custom data |
| **[FAQ](./FAQ.md)** | Common questions about training, performance, integration, and privacy |
| **[Roadmap](./roadmap.md)** | Development phases, milestones, and future plans |

## Supported Layers

| Layer | Description | Paper | Status |
|-------|-------------|-------|--------|
| **GCNConv** | Graph Convolutional Network | [Kipf & Welling 2017](https://arxiv.org/abs/1609.02907) | ✅ Ready |
| **GATConv** | Graph Attention Network | [Velickovic et al. 2018](https://arxiv.org/abs/1710.10903) | ✅ Ready |
| **SAGEConv** | GraphSAGE | [Hamilton et al. 2017](https://arxiv.org/abs/1706.02216) | ✅ Ready |
| **GINConv** | Graph Isomorphism Network | [Xu et al. 2019](https://arxiv.org/abs/1810.00826) | ✅ Ready |
| EdgeConv | Dynamic Graph CNN | [Wang et al. 2019](https://arxiv.org/abs/1801.07829) | 🔄 Coming Soon |

---

## Model Zoo

Pre-built models for common use cases:

| Model | Use Case | Description |
|-------|----------|-------------|
| **StudentMasteryPredictor** | Knowledge Assessment | Predict student mastery levels across concepts in a curriculum |
| **LearningPathRecommender** | Curriculum Sequencing | Recommend optimal learning paths through prerequisite graphs |
| **ConceptPrerequisiteMapper** | Dependency Analysis | Discover and predict prerequisite relationships between concepts |

```typescript
import { StudentMasteryPredictor, LearningPathRecommender } from 'browser-gnn';

// Predict student mastery from knowledge graph
const predictor = new StudentMasteryPredictor({ inputFeatures: 11 });
const { mastery, predictions } = await predictor.predict(knowledgeGraph);

// Get personalized learning path recommendations
const recommender = new LearningPathRecommender({ inputFeatures: 11 });
const { recommendations } = await recommender.recommend(curriculum, masteredConcepts);
```

---

## Installation

```bash
mkdir my-gnn-project
cd my-gnn-project
npm init -y
npm install browser-gnn
```

Or with yarn:
```bash
yarn add browser-gnn
```

Or include via CDN:
```html
<script type="module">
  import { GraphData, GCNConv } from 'https://unpkg.com/browser-gnn/dist/index.js';
</script>
```

---

## Quick Start (CLI)

The easiest way to get started:

```bash
npx browser-gnn init    # Creates a starter file
node gnn-example.mjs    # Run it!
```

**Other CLI commands:**
```bash
npx browser-gnn         # Show help
npx browser-gnn demo    # Open live demo in browser
```

---

## Quick Start (Code)

```typescript
import { GraphData, GCNConv, Sequential, ReLU, Softmax } from 'browser-gnn';

// Create graph data
const graph = new GraphData({
  // Node features: 4 nodes, 3 features each
  x: new Float32Array([
    0.1, 0.2, 0.3,  // Node 0 features
    0.4, 0.5, 0.6,  // Node 1 features
    0.7, 0.8, 0.9,  // Node 2 features
    1.0, 1.1, 1.2,  // Node 3 features
  ]),
  numNodes: 4,
  numFeatures: 3,
  // Edge index in COO format: [source nodes..., target nodes...]
  edgeIndex: new Uint32Array([
    0, 1, 1, 2, 2, 3, 3, 0,  // source nodes
    1, 0, 2, 1, 3, 2, 0, 3,  // target nodes (bidirectional edges)
  ]),
  numEdges: 8,
});

// Build a 3-layer GCN model for node classification
const model = new Sequential([
  new GCNConv({ inChannels: 3, outChannels: 16 }),
  new ReLU(),
  new GCNConv({ inChannels: 16, outChannels: 8 }),
  new ReLU(),
  new GCNConv({ inChannels: 8, outChannels: 2 }),  // 2 output classes
  new Softmax(),
]);

// Run inference
model.eval();
const output = model.forward(graph);

// Output contains class probabilities for each node
console.log('Node predictions:', output.x);
// Shape: [4, 2] - 4 nodes, 2 class probabilities each
```

---

## Interactive Demo

Run the interactive demo locally:

```bash
git clone https://github.com/fenago/BrowserGNN.git
cd BrowserGNN
npm install
npm run example
```

Then open http://localhost:3333 in your browser.

### Understanding the Demo

The demo showcases three different GNN architectures processing the same graph:

#### The Demo Graph

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

- **5 nodes** (labeled 0-4), each with 3 initial features
- **10 edges** representing connections between nodes
- This is a typical social network or molecular structure

#### What Each Demo Shows

**1. GCN Demo (Graph Convolutional Network)**
- Uses symmetric normalization: each node aggregates neighbor features weighted by degree
- Formula: `h_i' = σ(Σ_j (1/√(d_i·d_j)) · W · h_j)`
- Best for: Homophilic graphs where connected nodes are similar
- Output: Class probabilities showing which class each node likely belongs to

**2. GAT Demo (Graph Attention Network)**
- Learns attention weights: not all neighbors are equally important
- Multi-head attention captures different relationship types
- Output shows attention-weighted predictions
- Best for: Graphs where neighbor importance varies (citation networks, etc.)

**3. GraphSAGE Demo**
- Samples and aggregates neighbor features (mean, max, or pooling)
- Designed for inductive learning on unseen nodes
- Output shows sampling-based predictions
- Best for: Large graphs, dynamic graphs with new nodes

#### Interpreting the Output

```
Node 0: Class 0: 0.6234, Class 1: 0.3766
Node 1: Class 0: 0.5891, Class 1: 0.4109
...
```

- Each node gets a probability distribution over classes
- Higher probability = model is more confident about that class
- In a real application, you'd pick the highest probability as the prediction
- Similar nodes (connected in the graph) tend to have similar predictions

#### Benchmark Results

The benchmark tests inference speed on graphs of increasing size:
- 10 nodes, 50 nodes, 100 nodes, 500 nodes
- Shows average/min/max inference time in milliseconds
- Demonstrates that BrowserGNN scales efficiently

---

## Feature Engineering for GNNs

One of the most important aspects of getting good results from Graph Neural Networks is **feature engineering**. The demo includes real-world examples that showcase why this matters.

### The Problem with Naive Features

A common mistake is using **one-hot encoding** as node features (where each node gets a unique identifier like `[1,0,0,...]` for node 0, `[0,1,0,...]` for node 1, etc.). This approach fails because:

1. **No structural information**: One-hot features don't encode anything about graph topology
2. **Random initialization**: Without training, the GNN has no way to learn meaningful patterns
3. **Poor generalization**: Can't transfer to graphs of different sizes

### Structural Features: The Key to Good Performance

The Karate Club demos in BrowserGNN use **structural features** computed from the graph topology itself:

```typescript
// Features computed for each node using BFS (Breadth-First Search)
const features = {
  closenessToLeader1: 1 / (1 + shortestPathDistance(node, leader1)),
  closenessToLeader2: 1 / (1 + shortestPathDistance(node, leader2)),
  normalizedDegree: degree(node) / maxDegree,
  bias: (distToLeader2 - distToLeader1) / (distToLeader1 + distToLeader2 + 1)
};
```

| Feature | What It Captures | Why It Helps |
|---------|------------------|--------------|
| **Closeness to Leader 1** | How many hops to reach Mr. Hi | Nodes close to a leader tend to join their faction |
| **Closeness to Leader 2** | How many hops to reach Officer | Same reasoning for the other faction |
| **Normalized Degree** | How connected a node is | High-degree nodes are often "bridge" members |
| **Bias** | Which leader is closer | Directly encodes the prediction signal |

### The BFS Algorithm for Distance Features

```typescript
// Breadth-First Search to compute shortest path distances
function bfsDistances(adjacencyList, sourceNode) {
  const distances = new Array(numNodes).fill(Infinity);
  distances[sourceNode] = 0;
  const queue = [sourceNode];

  while (queue.length > 0) {
    const current = queue.shift();
    for (const neighbor of adjacencyList[current]) {
      if (distances[neighbor] === Infinity) {
        distances[neighbor] = distances[current] + 1;
        queue.push(neighbor);
      }
    }
  }
  return distances;
}
```

### Real-World Results: Zachary's Karate Club

Using structural features on the famous Karate Club dataset:

| Approach | Accuracy | Why |
|----------|----------|-----|
| One-hot encoding + random weights | ~47% | No meaningful signal |
| **Structural features** | **92%+** | Encodes community structure |

**The Key Insight**: In Zachary's 1977 study, members joined the faction whose leader they were **closer to in the friendship network**. Our structural features directly encode this pattern!

### When to Use Structural Features

| Use Case | Recommended Features |
|----------|---------------------|
| **Community Detection** | Distance to known community centers, clustering coefficient |
| **Node Classification** | Degree, PageRank, betweenness centrality |
| **Link Prediction** | Common neighbors, Jaccard similarity, Adamic-Adar |
| **Molecular Graphs** | Atom type, bond count, ring membership |

### Feature Engineering Best Practices

1. **Start with graph-theoretic features**: Degree, centrality measures, local clustering
2. **Add domain knowledge**: For social networks, use influence metrics; for molecules, use chemical properties
3. **Normalize features**: Scale to similar ranges (0-1 or z-score normalization)
4. **Use BFS for community tasks**: Distance to known landmarks is highly predictive
5. **Combine with learned features**: Structural features + GNN message passing = powerful representations

---

## Architecture

```
browser-gnn/
├── src/
│   ├── core/              # Core data structures
│   │   ├── tensor.ts      # Tensor operations (add, mul, matmul, etc.)
│   │   ├── graph.ts       # GraphData class with edge operations
│   │   └── sparse.ts      # Sparse matrix formats (COO, CSR)
│   ├── layers/            # GNN layer implementations
│   │   ├── gcn.ts         # Graph Convolutional Network
│   │   ├── gat.ts         # Graph Attention Network
│   │   └── sage.ts        # GraphSAGE
│   ├── nn/                # Neural network primitives
│   │   ├── module.ts      # Base Module class
│   │   ├── linear.ts      # Linear (fully connected) layer
│   │   ├── activation.ts  # ReLU, Softmax, etc.
│   │   └── sequential.ts  # Sequential container
│   └── index.ts           # Public API exports
├── examples/              # Interactive demos
├── tests/                 # Comprehensive test suite
└── dist/                  # Built output (ESM, CJS, types)
```

---

## API Reference

### GraphData

The fundamental data structure for representing graphs:

```typescript
const graph = new GraphData({
  x: Float32Array,           // Node features [numNodes * numFeatures]
  numNodes: number,          // Number of nodes
  numFeatures: number,       // Features per node
  edgeIndex: Uint32Array,    // Edge list [2 * numEdges] in COO format
  numEdges: number,          // Number of edges
  edgeAttr?: Float32Array,   // Optional edge attributes
  y?: Float32Array,          // Optional node labels
});

// Methods
graph.addSelfLoops()         // Add self-connections to all nodes
graph.hasSelfLoops()         // Check if self-loops exist
graph.isDirected()           // Check if graph is directed
graph.getInDegrees()         // Get in-degree for each node
graph.getOutDegrees()        // Get out-degree for each node
graph.getNeighbors(nodeId)   // Get neighbors of a node
```

### GNN Layers

```typescript
// Graph Convolutional Network
const gcn = new GCNConv({
  inChannels: 16,      // Input feature dimension
  outChannels: 32,     // Output feature dimension
  bias: true,          // Use bias (default: true)
  normalize: true,     // Apply symmetric normalization (default: true)
  addSelfLoops: true,  // Add self-loops (default: true)
});

// Graph Attention Network
const gat = new GATConv({
  inChannels: 16,
  outChannels: 32,
  heads: 4,            // Number of attention heads (default: 1)
  concat: true,        // Concatenate heads (default: true)
  dropout: 0.6,        // Attention dropout (default: 0)
  negativeSlope: 0.2,  // LeakyReLU slope (default: 0.2)
});

// GraphSAGE
const sage = new SAGEConv({
  inChannels: 16,
  outChannels: 32,
  aggregator: 'mean',  // 'mean' | 'max' | 'sum' | 'pool'
  normalize: true,     // L2 normalize output (default: false)
  rootWeight: true,    // Include self features (default: true)
});
```

### Building Models

```typescript
import { Sequential, ReLU, Dropout, Softmax } from 'browser-gnn';

const model = new Sequential([
  new GCNConv({ inChannels: 32, outChannels: 64 }),
  new ReLU(),
  new Dropout(0.5),
  new GCNConv({ inChannels: 64, outChannels: 32 }),
  new ReLU(),
  new GCNConv({ inChannels: 32, outChannels: numClasses }),
  new Softmax(),
]);

// Training mode (dropout active)
model.train();

// Evaluation mode (dropout disabled)
model.eval();

// Forward pass
const output = model.forward(graph);
```

### Utility Functions

```typescript
import { randomGraph, fromEdgeList, createBrowserGNN } from 'browser-gnn';

// Create random graph for testing
const graph = randomGraph(100, 0.1, 32);  // 100 nodes, 10% edge prob, 32 features

// Create graph from edge list
const edges = [[0, 1], [1, 2], [2, 0]];
const graph = fromEdgeList(edges, 3);  // 3 nodes

// Initialize BrowserGNN (detects best backend)
const { backend, info } = await createBrowserGNN();
```

---

## Use Cases

### 1. Privacy-Preserving Applications
Analyze sensitive graph data without it ever leaving the user's device:
- **Social Network Analysis**: Friend recommendation, community detection, influence analysis
- **Medical Knowledge Graphs**: Patient relationship modeling, clinical decision support
- **Financial Fraud Detection**: Transaction graph analysis on sensitive banking data

### 2. Edge Computing & IoT
[Research shows](https://dl.acm.org/doi/10.1145/3649329.3655938) GNNs on edge devices achieve **21x lower latency** than centralized methods:
- **Smart Home Networks**: Device relationship modeling
- **Industrial IoT**: Sensor network anomaly detection
- **Mobile Apps**: Offline-first graph reasoning

### 3. Client-Side Molecule & Drug Discovery
Drug discovery and chemistry education in the browser:
- Molecular property prediction
- Drug-drug interaction checking in healthcare apps
- Chemical structure classification

### 4. Knowledge Graph Reasoning
Enable semantic search and reasoning without backend:
- Entity classification
- Link prediction
- Question answering over graphs

### 5. Real-Time Recommendation
Edge-based recommendations computed locally:
- Product recommendations (user-item graphs)
- Content suggestions
- Collaborative filtering without server roundtrips

### 6. Education & Research
Interactive GNN learning without infrastructure:
- Visualize message passing
- Rapid prototyping without Python setup
- Democratizing GNN access for students

---

## Integration with Small Language Models (SLMs)

BrowserGNN becomes even more powerful when combined with browser-based Small Language Models. [Recent research](https://www.mdpi.com/2076-3417/15/5/2418) shows that **SLM + GNN integration achieves near-LLM performance** while running on resource-constrained devices.

### Architecture: GNN as Knowledge Encoder for SLMs

```
┌─────────────────────────────────────────────────────────┐
│                    Browser Environment                   │
│                                                         │
│  ┌─────────────┐    embeddings    ┌─────────────────┐  │
│  │ BrowserGNN  │ ───────────────► │  SLM (WebLLM/   │  │
│  │ (GCN/GAT)   │                  │  Phi-3/Gemma)   │  │
│  └─────────────┘                  └─────────────────┘  │
│        ▲                                   │           │
│        │                                   ▼           │
│  ┌─────────────┐                  ┌─────────────────┐  │
│  │ Knowledge   │                  │ Natural Language│  │
│  │ Graph       │                  │ Response        │  │
│  └─────────────┘                  └─────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Synergy Patterns

| Pattern | How It Works | Example |
|---------|--------------|---------|
| **Knowledge-Enhanced Q&A** | BrowserGNN encodes knowledge graph, SLM uses embeddings for grounded answers | "How are these drugs related?" with reasoning path |
| **Graph-Guided Retrieval** | GNN identifies relevant subgraphs, SLM generates response from context | Reduced hallucination through structural grounding |
| **Entity Relationship Understanding** | GNN captures multi-hop relationships, SLM translates to natural language | "Explain the connection between Gene X and Disease Y" |
| **On-Device AI Assistant** | Personal knowledge graph + BrowserGNN + SLM = private conversational AI | 100% offline, nothing leaves device |

### Why This Combination Matters

- **Privacy**: Both GNN and SLM run entirely in browser—sensitive data never leaves the device
- **Performance**: [Research](https://arxiv.org/abs/2306.08302) shows GNN+LLM achieves state-of-the-art on knowledge-intensive QA
- **Efficiency**: SLMs (Phi-3, Gemma 2B) + BrowserGNN run on consumer hardware
- **Interpretability**: GNN provides explicit reasoning paths that SLM can explain

### Example: Knowledge Graph Q&A

```typescript
import { GraphData, GATConv } from 'browser-gnn';
// Assume WebLLM or similar is loaded

// 1. Encode knowledge graph with BrowserGNN
const knowledgeGraph = new GraphData({ /* entities and relations */ });
const gat = new GATConv({ inChannels: 128, outChannels: 64, heads: 4 });
const nodeEmbeddings = gat.forward(knowledgeGraph);

// 2. Find relevant subgraph for query
const relevantNodes = findRelevantNodes(query, nodeEmbeddings);
const subgraphContext = extractSubgraph(knowledgeGraph, relevantNodes);

// 3. Feed to SLM with graph context
const prompt = `Given this knowledge: ${subgraphContext}\n\nQuestion: ${query}`;
const response = await slm.generate(prompt);
```

### Compatible Browser SLMs

| Model | Size | Works with BrowserGNN |
|-------|------|----------------------|
| [Phi-3-mini](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct) | 3.8B | ✅ Via WebLLM |
| [Gemma 2B](https://huggingface.co/google/gemma-2b) | 2B | ✅ Via WebLLM |
| [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) | 1.1B | ✅ Via WebLLM |
| [ONNX Models](https://onnxruntime.ai/) | Various | ✅ Via ONNX Runtime Web |

---

## Browser Support

| Browser | Supported | Notes |
|---------|-----------|-------|
| Chrome 90+ | ✅ | Full support |
| Edge 90+ | ✅ | Full support |
| Firefox 90+ | ✅ | Full support |
| Safari 15+ | ✅ | Full support |

---

## Roadmap

### Phase 1: Core Library ✅
- [x] Tensor operations
- [x] GraphData structure
- [x] Sparse matrix operations (COO, CSR)
- [x] GCN, GAT, GraphSAGE layers
- [x] Sequential model container
- [x] Comprehensive test suite (129+ tests)

### Phase 2: Performance Optimization ✅
- [x] WebGPU compute shaders for GPU acceleration
- [x] WASM-optimized kernels with 8x loop unrolling
- [x] WASM scatter/gather operations
- [x] forwardAsync() API for GPU inference

### Phase 3: Training Support ✅
- [x] Automatic differentiation (autograd)
- [x] Optimizers (SGD, Adam, Adagrad, RMSprop)
- [x] Loss functions (CrossEntropy, MSE, BCE, NLL, L1, SmoothL1)
- [x] LR Schedulers (Step, Exponential, Cosine, ReduceOnPlateau)
- [x] Trainer class with early stopping

### Phase 4: Advanced Features (Current)
- [x] GINConv layer
- [x] Model Zoo (education-focused models)
- [x] Model serialization/deserialization
- [ ] More layer types (EdgeConv, ChebConv)
- [ ] Heterogeneous graphs
- [ ] ONNX model import

---

## Performance

Inference benchmarks on a MacBook Pro M1:

| Graph Size | Nodes | Edges | GCN (3 layer) | GAT (2 head) |
|------------|-------|-------|---------------|--------------|
| Small | 10 | 20 | 0.5ms | 0.8ms |
| Medium | 100 | 500 | 2.1ms | 4.3ms |
| Large | 500 | 2500 | 12.4ms | 28.7ms |
| XLarge | 1000 | 10000 | 45.2ms | 112.3ms |

---

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

```bash
# Clone the repo
git clone https://github.com/fenago/BrowserGNN.git
cd BrowserGNN

# Install dependencies
npm install

# Run tests
npm test

# Run dev server
npm run example
```

---

## Citation

If you use BrowserGNN in your research, please cite:

```bibtex
@software{browsergnn2024,
  author = {Dr. Lee},
  title = {BrowserGNN: The World's First Graph Neural Network Library for the Browser},
  year = {2024},
  url = {https://github.com/fenago/BrowserGNN}
}
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

<div align="center">

**BrowserGNN by Dr. Lee**

*Democratizing Graph Neural Networks - One Browser at a Time*

[Live Demo](https://browsergnn.com) | [npm Package](https://www.npmjs.com/package/browser-gnn) | [GitHub](https://github.com/fenago/BrowserGNN)

[Report Bug](https://github.com/fenago/BrowserGNN/issues) | [Request Feature](https://github.com/fenago/BrowserGNN/issues)

</div>
