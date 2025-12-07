# BrowserGNN FAQ

Frequently asked questions about BrowserGNN - The World's First Browser-Based Graph Neural Network Library.

---

## Table of Contents

- [General Questions](#general-questions)
- [Training & Data](#training--data)
- [Performance](#performance)
- [Integration & Deployment](#integration--deployment)
- [Privacy & Security](#privacy--security)
- [Comparison with Other Libraries](#comparison-with-other-libraries)
- [Technical Details](#technical-details)
- [Troubleshooting](#troubleshooting)

---

## General Questions

### What is BrowserGNN?

BrowserGNN is a JavaScript/TypeScript library that enables Graph Neural Network (GNN) inference directly in web browsers. It provides implementations of popular GNN architectures (GCN, GAT, GraphSAGE) that run entirely client-side without requiring a Python backend or server.

### Why would I use GNNs in the browser?

Several compelling reasons:

1. **Privacy**: Sensitive graph data (social connections, medical records, financial transactions) never leaves the user's device
2. **Latency**: No network round-trips means instant inference
3. **Cost**: No server infrastructure to maintain
4. **Offline**: Works without internet connection after initial load
5. **Scale**: Computation scales with user devices, not your servers

### What types of graphs can BrowserGNN handle?

BrowserGNN supports:
- **Homogeneous graphs**: Single node type, single edge type (social networks, molecules)
- **Directed and undirected graphs**: Both are supported
- **Graphs with node features**: Required for GNN inference
- **Graphs with edge features**: Supported in certain layers (GAT)
- **Self-loops**: Can be added automatically or manually

Current limitations:
- Heterogeneous graphs (multiple node/edge types) are planned but not yet supported
- Dynamic graphs require manual updates

### What's the maximum graph size BrowserGNN can handle?

This depends on the user's device, but general guidelines:

| Graph Size | Nodes | Edges | Performance |
|------------|-------|-------|-------------|
| Small | < 100 | < 500 | < 5ms |
| Medium | < 1,000 | < 10,000 | < 50ms |
| Large | < 10,000 | < 100,000 | < 500ms |
| Very Large | < 50,000 | < 500,000 | 1-5 seconds |

For graphs larger than 50,000 nodes, consider:
- Subgraph sampling
- Mini-batch inference
- Server-side processing for initial embedding

---

## Training & Data

### Can I train models with BrowserGNN?

**Currently: Inference only.** BrowserGNN v0.2.0 supports forward passes (inference) but not backpropagation (training).

**Recommended workflow:**
1. Train your model in Python using PyTorch Geometric or TensorFlow GNN
2. Export weights or convert to a format BrowserGNN can load
3. Run inference in the browser with BrowserGNN

**Coming soon:** Training support is on our [roadmap](roadmap.md) for Phase 3.

### How do I prepare my own data?

Graph data in BrowserGNN requires:

```typescript
const graph = new GraphData({
  // Node features: flattened array [numNodes * numFeatures]
  x: new Float32Array([...]),
  numNodes: 100,
  numFeatures: 16,

  // Edge index in COO format: [sources..., targets...]
  edgeIndex: new Uint32Array([...]),
  numEdges: 500,

  // Optional: labels for supervised tasks
  y: new Float32Array([...]),
});
```

See the [Tutorial](tutorial.md) for detailed data preparation guidance.

### What features should I use for my nodes?

This is the most important decision for GNN performance. Options include:

| Feature Type | Examples | When to Use |
|--------------|----------|-------------|
| **Structural** | Degree, centrality, clustering coefficient | When topology is the signal |
| **Attribute-based** | User age, molecule properties | When you have node metadata |
| **Learned embeddings** | node2vec, DeepWalk | When you have large unlabeled graphs |
| **One-hot encoding** | Node ID as sparse vector | Small graphs with training (not recommended for inference-only) |
| **Distance-based** | Shortest path to landmarks | Community detection, navigation |

**Best practice:** Combine structural features with domain-specific attributes. See the Karate Club demo in the [README](README.md) for an example of effective feature engineering.

### How do I convert PyTorch Geometric models to BrowserGNN?

Currently, you need to manually transfer weights:

```python
# In Python (PyTorch Geometric)
import torch
import json

model = YourGCNModel()
model.load_state_dict(torch.load('model.pt'))

weights = {
    'layer0_weight': model.conv1.weight.detach().numpy().tolist(),
    'layer0_bias': model.conv1.bias.detach().numpy().tolist(),
    # ... more layers
}

with open('weights.json', 'w') as f:
    json.dump(weights, f)
```

```typescript
// In BrowserGNN
const weights = await fetch('weights.json').then(r => r.json());
const gcn = new GCNConv({ inChannels: 16, outChannels: 32 });
gcn.loadWeights(weights.layer0_weight, weights.layer0_bias);
```

**Coming soon:** Automated ONNX model import is on our roadmap.

### Can I fine-tune pre-trained models in the browser?

Not yet. This requires backpropagation support, which is planned for Phase 3. However, you can:

1. Use pre-trained embeddings as input features
2. Combine BrowserGNN with other browser ML libraries that support training
3. Use transfer learning on the server, deploy frozen model to browser

---

## Performance

### How do I measure GNN performance?

**Inference speed:**
```typescript
const start = performance.now();
const output = model.forward(graph);
const elapsed = performance.now() - start;
console.log(`Inference time: ${elapsed.toFixed(2)}ms`);
```

**Model accuracy** (requires labels):
```typescript
// For node classification
const predictions = output.x.argmax(1);  // Get predicted class per node
const correct = predictions.filter((pred, i) => pred === trueLabels[i]).length;
const accuracy = correct / graph.numNodes;
```

**Benchmarking suite:**
```typescript
import { benchmark } from 'browser-gnn';

const results = await benchmark({
  model: myModel,
  graph: myGraph,
  iterations: 100,
  warmup: 10,
});

console.log(results);
// { avg: 12.4, min: 10.1, max: 18.7, std: 2.3 }
```

### How does BrowserGNN compare to PyTorch Geometric performance?

BrowserGNN is optimized for inference, not training. Rough comparisons:

| Operation | PyTorch Geometric (GPU) | BrowserGNN (WebGPU) | BrowserGNN (CPU/WASM) |
|-----------|------------------------|---------------------|----------------------|
| GCN forward (1K nodes) | ~1ms | ~5ms | ~15ms |
| GAT forward (1K nodes) | ~3ms | ~15ms | ~50ms |
| Training step | ~10ms | Not supported | Not supported |

**Note:** Browser performance varies significantly by device. WebGPU acceleration (when available) closes the gap considerably.

### Why is my inference slow?

Common causes and solutions:

1. **Large graph**: Consider subgraph sampling
2. **Too many layers**: GNNs rarely need > 3 layers (over-smoothing)
3. **High feature dimensions**: Reduce with linear projection
4. **No WebGPU**: Falls back to slower CPU path
5. **First inference**: Initial run includes JIT compilation; subsequent runs are faster

**Optimization tips:**
```typescript
// Warm up the model
model.forward(smallGraph);

// Reuse graph objects instead of recreating
graph.updateFeatures(newFeatures);  // Instead of new GraphData(...)

// Use eval mode (disables dropout)
model.eval();
```

### Does BrowserGNN use WebGPU?

**Architecture is WebGPU-ready**, but full WebGPU compute shader implementation is in progress (Phase 2 of our roadmap). Currently:

- ✅ Detection of WebGPU availability
- ✅ Fallback to optimized JavaScript/WASM
- 🔄 WebGPU compute shaders (coming soon)

Check your backend:
```typescript
const { backend, info } = await createBrowserGNN();
console.log(backend);  // 'webgpu' | 'wasm' | 'cpu'
```

---

## Integration & Deployment

### How do I integrate BrowserGNN into my web app?

**npm/bundler (recommended):**
```bash
npm install browser-gnn
```

```typescript
import { GraphData, GCNConv } from 'browser-gnn';
```

**CDN:**
```html
<script type="module">
  import { GraphData, GCNConv } from 'https://unpkg.com/browser-gnn/dist/index.js';
</script>
```

**Node.js (for testing/SSR):**
```javascript
import { GraphData, GCNConv } from 'browser-gnn';
// Works in Node.js 18+ with ES modules
```

### How do I deploy a BrowserGNN application?

BrowserGNN runs entirely client-side, so deployment is simple:

1. **Static hosting**: Works with any static host (Netlify, Vercel, GitHub Pages, S3)
2. **Bundle size**: ~30KB gzipped (core library)
3. **No server requirements**: No Python, no GPU servers needed

```bash
# Build your app
npm run build

# Deploy to Netlify
netlify deploy --prod --dir=dist

# Or GitHub Pages
gh-pages -d dist
```

### Can I use BrowserGNN with React/Vue/Svelte?

Yes! BrowserGNN is framework-agnostic:

**React:**
```tsx
import { useEffect, useState } from 'react';
import { GraphData, GCNConv } from 'browser-gnn';

function GraphClassifier({ graphData }) {
  const [prediction, setPrediction] = useState(null);

  useEffect(() => {
    const model = new GCNConv({ inChannels: 16, outChannels: 2 });
    const graph = new GraphData(graphData);
    const output = model.forward(graph);
    setPrediction(output.x);
  }, [graphData]);

  return <div>Prediction: {prediction}</div>;
}
```

**Vue:**
```vue
<script setup>
import { ref, onMounted } from 'vue';
import { GraphData, GCNConv } from 'browser-gnn';

const prediction = ref(null);

onMounted(async () => {
  const model = new GCNConv({ inChannels: 16, outChannels: 2 });
  const graph = new GraphData(props.graphData);
  prediction.value = model.forward(graph).x;
});
</script>
```

### How do I integrate BrowserGNN with an SLM (Small Language Model)?

See the [README section on SLM integration](README.md#integration-with-small-language-models-slms) for architecture and code examples. The basic pattern:

1. BrowserGNN encodes your knowledge graph into embeddings
2. Relevant subgraph is extracted based on query
3. SLM (via WebLLM or ONNX) generates response using graph context

### Can I use BrowserGNN in a Web Worker?

Yes, and it's recommended for heavy computations:

```typescript
// worker.js
import { GraphData, GCNConv } from 'browser-gnn';

self.onmessage = async (e) => {
  const { graphData, modelConfig } = e.data;
  const graph = new GraphData(graphData);
  const model = new GCNConv(modelConfig);
  const result = model.forward(graph);
  self.postMessage({ prediction: Array.from(result.x.data) });
};

// main.js
const worker = new Worker('worker.js', { type: 'module' });
worker.postMessage({ graphData, modelConfig });
worker.onmessage = (e) => console.log(e.data.prediction);
```

---

## Privacy & Security

### Does my graph data leave the browser?

**No.** BrowserGNN performs all computation client-side. Your graph data:
- Never sent to any server
- Never logged or tracked
- Stays entirely in the browser's memory
- Can be stored locally in IndexedDB/localStorage if you choose

### Is BrowserGNN suitable for sensitive data?

Yes, this is one of its primary use cases:
- **Medical knowledge graphs**: Patient data stays on device
- **Financial graphs**: Transaction analysis without exposing data
- **Social graphs**: Relationship analysis with full privacy

### Can I use BrowserGNN offline?

Yes, once loaded:
1. Cache the library (via service worker or bundling)
2. Cache your model weights
3. Graph data is already client-side
4. Full functionality without internet

```typescript
// Service worker caching example
const CACHE_NAME = 'browsergnn-v1';
const ASSETS = [
  '/browser-gnn.js',
  '/model-weights.json',
];

self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE_NAME).then(c => c.addAll(ASSETS)));
});
```

---

## Comparison with Other Libraries

### BrowserGNN vs PyTorch Geometric

| Aspect | PyTorch Geometric | BrowserGNN |
|--------|-------------------|------------|
| **Platform** | Python/CUDA | Browser/JavaScript |
| **Training** | ✅ Full support | ❌ Inference only |
| **Inference** | ✅ Fast (GPU) | ✅ Good (WebGPU/WASM) |
| **Privacy** | ❌ Requires server | ✅ Client-side |
| **Layers** | 50+ layer types | 3 (GCN, GAT, SAGE) |
| **Ecosystem** | Mature, extensive | New, growing |

**Use PyG for:** Training, research, complex architectures
**Use BrowserGNN for:** Privacy-preserving inference, edge deployment, web apps

### BrowserGNN vs TensorFlow.js

| Aspect | TensorFlow.js | BrowserGNN |
|--------|---------------|------------|
| **GNN Support** | ❌ None ([requested](https://github.com/tensorflow/tfjs/issues/5975)) | ✅ GCN, GAT, SAGE |
| **General ML** | ✅ Comprehensive | ❌ GNN-focused |
| **Training** | ✅ Supported | ❌ Inference only |
| **Bundle size** | ~500KB+ | ~30KB |

**Use TF.js for:** General ML in browser (CNNs, RNNs, etc.)
**Use BrowserGNN for:** Graph-specific tasks

### BrowserGNN vs DGL

| Aspect | DGL | BrowserGNN |
|--------|-----|------------|
| **Platform** | Python/CUDA | Browser/JavaScript |
| **Focus** | Large-scale GNNs | Privacy-preserving inference |
| **Heterogeneous graphs** | ✅ Excellent | ❌ Not yet |
| **Sampling** | ✅ Advanced | 🔄 Basic (coming) |

---

## Technical Details

### What sparse matrix formats does BrowserGNN support?

- **COO (Coordinate)**: Default format, good for construction
- **CSR (Compressed Sparse Row)**: Efficient for row-wise operations
- **CSC (Compressed Sparse Column)**: Efficient for column-wise operations

```typescript
// COO (default)
const edges = new Uint32Array([0, 1, 2, 1, 2, 0]);  // [sources, targets]

// Convert to CSR for efficiency
const csr = graph.toCSR();
```

### What activation functions are available?

- `ReLU` - Rectified Linear Unit
- `LeakyReLU` - Leaky ReLU with configurable slope
- `Softmax` - For classification outputs
- `Sigmoid` - For binary classification
- `Tanh` - Hyperbolic tangent
- `ELU` - Exponential Linear Unit

### How does message passing work in BrowserGNN?

BrowserGNN implements the standard message-passing framework:

```
h_i^{(l+1)} = UPDATE(h_i^{(l)}, AGGREGATE({h_j^{(l)} : j ∈ N(i)}))
```

Each layer:
1. **Message**: Transform neighbor features
2. **Aggregate**: Combine neighbor messages (mean, sum, max, attention)
3. **Update**: Combine with self features, apply non-linearity

```typescript
// GCN: mean aggregation with symmetric normalization
// GAT: attention-weighted aggregation
// SAGE: configurable aggregation (mean, max, sum, pool)
```

---

## Troubleshooting

### "Cannot find module 'browser-gnn'"

Ensure you're using ES modules:
```json
// package.json
{ "type": "module" }
```

Or use the correct import syntax:
```typescript
// ESM
import { GraphData } from 'browser-gnn';

// CommonJS
const { GraphData } = require('browser-gnn');
```

### "Graph has no self-loops" warning

Many GNN layers require self-loops. Add them:
```typescript
graph.addSelfLoops();
// or
const gcn = new GCNConv({ ..., addSelfLoops: true });  // default
```

### "Out of memory" error

Your graph is too large for browser memory. Solutions:
1. Reduce graph size (sample nodes/edges)
2. Reduce feature dimensions
3. Use mini-batch processing
4. Process subgraphs individually

### Inference returns NaN

Common causes:
1. **Uninitialized weights**: Ensure model is properly initialized
2. **Feature scaling**: Normalize features to reasonable range
3. **Numerical instability**: Check for very large/small values
4. **Empty neighborhoods**: Ensure graph is connected

```typescript
// Check for issues
console.log('Features range:', Math.min(...graph.x), Math.max(...graph.x));
console.log('Any NaN?:', graph.x.some(v => isNaN(v)));
```

---

## Still have questions?

- [Open an issue](https://github.com/fenago/BrowserGNN/issues)
- [Check the documentation](README.md)
- [Try the live demo](https://browsergnn.netlify.app)
