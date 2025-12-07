# Getting Started with BrowserGNN

Get up and running with BrowserGNN in under 5 minutes.

## Installation

```bash
mkdir my-gnn-project
cd my-gnn-project
npm init -y
npm install browser-gnn
```

## Quick Test

Create `test.mjs`:

```javascript
import { GraphData, GCNConv, createBrowserGNN } from 'browser-gnn';

async function main() {
  // Initialize BrowserGNN
  const { backend } = await createBrowserGNN();
  console.log('Initialized with backend:', backend);

  // Create a simple graph: 3 nodes, 2 features each
  const graph = new GraphData({
    x: new Float32Array([
      1, 0,   // Node 0 features
      0, 1,   // Node 1 features
      1, 1    // Node 2 features
    ]),
    numNodes: 3,
    numFeatures: 2,
    edgeIndex: new Uint32Array([
      0, 1, 1, 2,  // Source nodes
      1, 0, 2, 1   // Target nodes (edges: 0-1, 1-0, 1-2, 2-1)
    ]),
    numEdges: 4
  });

  console.log('Graph:', graph.toString());

  // Apply a GCN layer
  const gcn = new GCNConv({ inChannels: 2, outChannels: 4 });
  const output = gcn.forward(graph);

  console.log('Output shape:', output.x.shape);
  console.log('Output features:', output.x.data);
}

main();
```

Run it:

```bash
node test.mjs
```

Expected output:
```
Initialized with backend: wasm
Graph: GraphData(numNodes=3, numEdges=4, numFeatures=2)
Output shape: [ 3, 4 ]
Output features: Float32Array(12) [...]
```

## Available GNN Layers

### GCN (Graph Convolutional Network)

```javascript
import { GCNConv } from 'browser-gnn';

const gcn = new GCNConv({
  inChannels: 16,    // Input feature dimension
  outChannels: 32,   // Output feature dimension
  bias: true         // Include bias (default: true)
});

const output = gcn.forward(graph);
```

### GAT (Graph Attention Network)

```javascript
import { GATConv } from 'browser-gnn';

const gat = new GATConv({
  inChannels: 16,
  outChannels: 32,
  heads: 4,          // Number of attention heads
  concat: true,      // Concatenate heads (true) or average (false)
  dropout: 0.6       // Attention dropout
});

const output = gat.forward(graph);
// Output dim = outChannels * heads if concat=true
```

### GraphSAGE

```javascript
import { SAGEConv } from 'browser-gnn';

const sage = new SAGEConv({
  inChannels: 16,
  outChannels: 32,
  aggregator: 'mean'  // 'mean', 'max', or 'sum'
});

const output = sage.forward(graph);
```

## Browser Usage

```html
<!DOCTYPE html>
<html>
<head>
  <title>BrowserGNN Demo</title>
</head>
<body>
  <script type="module">
    import { GraphData, GCNConv, createBrowserGNN } from 'https://unpkg.com/browser-gnn';

    async function run() {
      const { backend } = await createBrowserGNN();
      console.log('Running in browser with:', backend);

      const graph = new GraphData({
        x: new Float32Array([1, 2, 3, 4]),
        numNodes: 2,
        numFeatures: 2,
        edgeIndex: new Uint32Array([0, 1, 1, 0]),
        numEdges: 2
      });

      const gcn = new GCNConv({ inChannels: 2, outChannels: 4 });
      const result = gcn.forward(graph);

      console.log('Result:', result.x.data);
    }

    run();
  </script>
</body>
</html>
```

## Next Steps

- See [using_your_data.md](using_your_data.md) for working with your own graphs
- Check the [Live Demo](https://browsergnn.com) for interactive examples
- Read the full [API Reference](README.md#api-reference)

## Links

- [npm Package](https://www.npmjs.com/package/browser-gnn)
- [GitHub](https://github.com/fenago/BrowserGNN)
- [Live Demo](https://browsergnn.com)
