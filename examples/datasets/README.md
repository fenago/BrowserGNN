# Sample Graph Datasets for BrowserGNN

This folder contains sample CSV datasets for training GNN models in the BrowserGNN Training Dashboard.

## How to Use

1. Open the Training Dashboard (`training-dashboard.html`)
2. Click on "Custom CSV" in the Dataset panel
3. Upload the edge and label files together (select multiple files)
4. The dashboard will automatically parse and load your graph

## Dataset Format

Each dataset consists of two CSV files:

### Edges File (`*-edges.csv`)
Contains the graph structure with source and target node IDs:
```csv
source,target
0,1
0,2
1,2
```

### Labels File (`*-labels.csv`)
Contains the class label for each node (one per row):
```csv
label
0
1
0
```

## Available Datasets

### 1. Social Network (`social-network-*.csv`)
- **Nodes**: 20
- **Edges**: 48
- **Classes**: 2 (community membership)
- A small social network with friendship connections

### 2. Citation Network (`citation-network-*.csv`)
- **Nodes**: 25
- **Edges**: 54
- **Classes**: 3 (paper topics)
- A citation network between academic papers

### 3. Molecule Graph (`molecule-*.csv`)
- **Nodes**: 16
- **Edges**: 32
- **Classes**: 3 (atom types)
- A molecular structure with atom connections

### 4. Protein Interaction (`protein-*.csv`)
- **Nodes**: 30
- **Edges**: 54
- **Classes**: 3 (protein families)
- A protein-protein interaction network

### 5. Web Graph (`web-graph-*.csv`)
- **Nodes**: 25
- **Edges**: 56
- **Classes**: 3 (page categories)
- A web page link structure

## Creating Your Own Dataset

1. **Edges file**: Create a CSV with `source,target` columns containing node indices (0-based)
2. **Labels file**: Create a CSV with `label` column containing integer class labels
3. Upload both files together to the dashboard

### Optional: Features file (`*-features.csv`)
You can also provide custom node features:
```csv
feature1,feature2,feature3
0.5,0.2,0.8
0.1,0.9,0.3
```
If not provided, the dashboard will use one-hot encoding of node IDs as features.
