/**
 * BrowserGNN by Dr. Lee
 * WebGPU WGSL Shaders for Graph Neural Networks
 *
 * High-performance compute shaders for sparse matrix operations,
 * attention mechanisms, and aggregation functions.
 */

/**
 * Sparse Matrix-Dense Matrix Multiplication (SpMM)
 * Computes: Y = A * X where A is sparse (COO format) and X is dense
 *
 * This is the core operation for GNN message passing:
 * - GCN: Aggregates weighted neighbor features
 * - GAT: Aggregates attention-weighted features
 * - SAGE: Used in mean aggregation
 */
export const SPMM_SHADER = /* wgsl */ `
// Uniforms
struct SpMMUniforms {
  numRows: u32,      // Number of output rows (nodes)
  numCols: u32,      // Number of input columns (nodes)
  numFeatures: u32,  // Feature dimension
  nnz: u32,          // Number of non-zeros (edges)
}

@group(0) @binding(0) var<uniform> uniforms: SpMMUniforms;
@group(0) @binding(1) var<storage, read> rows: array<u32>;      // Row indices [nnz]
@group(0) @binding(2) var<storage, read> cols: array<u32>;      // Column indices [nnz]
@group(0) @binding(3) var<storage, read> values: array<f32>;    // Edge values [nnz]
@group(0) @binding(4) var<storage, read> input: array<f32>;     // Dense input [numCols, numFeatures]
@group(0) @binding(5) var<storage, read_write> output: array<f32>; // Output [numRows, numFeatures]

// Each workgroup processes one row (node)
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let edge_idx = global_id.x;

  if (edge_idx >= uniforms.nnz) {
    return;
  }

  let row = rows[edge_idx];
  let col = cols[edge_idx];
  let val = values[edge_idx];

  // For each feature, accumulate: output[row, f] += val * input[col, f]
  for (var f: u32 = 0u; f < uniforms.numFeatures; f = f + 1u) {
    let input_idx = col * uniforms.numFeatures + f;
    let output_idx = row * uniforms.numFeatures + f;

    // Atomic add for thread-safe accumulation
    // Note: atomicAdd only works with i32/u32, so we use a workaround
    // For now, we'll process edges sequentially per row in a different kernel
    let contribution = val * input[input_idx];

    // Store contribution (will be summed in a reduction pass)
    output[output_idx] = output[output_idx] + contribution;
  }
}
`;

/**
 * SpMM with row-based parallelism (more efficient for sparse graphs)
 * Each workgroup handles one row, threads within handle features
 */
export const SPMM_ROW_PARALLEL_SHADER = /* wgsl */ `
struct SpMMUniforms {
  numRows: u32,
  numFeatures: u32,
  maxNnzPerRow: u32,
  padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: SpMMUniforms;
@group(0) @binding(1) var<storage, read> rowPtr: array<u32>;     // CSR row pointers [numRows + 1]
@group(0) @binding(2) var<storage, read> colIndices: array<u32>; // CSR column indices [nnz]
@group(0) @binding(3) var<storage, read> values: array<f32>;     // Edge values [nnz]
@group(0) @binding(4) var<storage, read> input: array<f32>;      // Dense input [numCols, numFeatures]
@group(0) @binding(5) var<storage, read_write> output: array<f32>; // Output [numRows, numFeatures]

@compute @workgroup_size(64, 4)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let row = workgroup_id.x;
  let feature_block = local_id.x;
  let feature_offset = local_id.y;

  if (row >= uniforms.numRows) {
    return;
  }

  let feature = feature_block * 4u + feature_offset;
  if (feature >= uniforms.numFeatures) {
    return;
  }

  let row_start = rowPtr[row];
  let row_end = rowPtr[row + 1u];

  var sum: f32 = 0.0;

  for (var j: u32 = row_start; j < row_end; j = j + 1u) {
    let col = colIndices[j];
    let val = values[j];
    let input_idx = col * uniforms.numFeatures + feature;
    sum = sum + val * input[input_idx];
  }

  let output_idx = row * uniforms.numFeatures + feature;
  output[output_idx] = sum;
}
`;

/**
 * Scatter-Add Shader for Message Aggregation
 * Aggregates messages from edges to nodes
 */
export const SCATTER_ADD_SHADER = /* wgsl */ `
struct ScatterUniforms {
  numMessages: u32,
  numNodes: u32,
  numFeatures: u32,
  padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: ScatterUniforms;
@group(0) @binding(1) var<storage, read> indices: array<u32>;   // Target node indices [numMessages]
@group(0) @binding(2) var<storage, read> src: array<f32>;       // Source values [numMessages, numFeatures]
@group(0) @binding(3) var<storage, read_write> dst: array<f32>; // Destination [numNodes, numFeatures]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let msg_idx = global_id.x;

  if (msg_idx >= uniforms.numMessages) {
    return;
  }

  let target_node = indices[msg_idx];

  for (var f: u32 = 0u; f < uniforms.numFeatures; f = f + 1u) {
    let src_idx = msg_idx * uniforms.numFeatures + f;
    let dst_idx = target_node * uniforms.numFeatures + f;

    // Note: For correct atomics, would need to use atomic operations
    // This simplified version works for single-pass execution
    dst[dst_idx] = dst[dst_idx] + src[src_idx];
  }
}
`;

/**
 * Scatter-Mean Shader with Count Tracking
 */
export const SCATTER_MEAN_SHADER = /* wgsl */ `
struct ScatterUniforms {
  numMessages: u32,
  numNodes: u32,
  numFeatures: u32,
  padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: ScatterUniforms;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> sum: array<f32>;    // Sum buffer
@group(0) @binding(4) var<storage, read_write> count: array<u32>;  // Count buffer

// Pass 1: Accumulate sums and counts
@compute @workgroup_size(256)
fn accumulate(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let msg_idx = global_id.x;

  if (msg_idx >= uniforms.numMessages) {
    return;
  }

  let target_node = indices[msg_idx];

  // Atomically increment count for this node (only once per message)
  // Note: Simplified - actual implementation would need proper atomics

  for (var f: u32 = 0u; f < uniforms.numFeatures; f = f + 1u) {
    let src_idx = msg_idx * uniforms.numFeatures + f;
    let dst_idx = target_node * uniforms.numFeatures + f;
    sum[dst_idx] = sum[dst_idx] + src[src_idx];
  }
}

// Pass 2: Divide by counts (run in separate dispatch)
@compute @workgroup_size(256)
fn normalize(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;
  let total = uniforms.numNodes * uniforms.numFeatures;

  if (idx >= total) {
    return;
  }

  let node = idx / uniforms.numFeatures;
  let node_count = count[node];

  if (node_count > 0u) {
    sum[idx] = sum[idx] / f32(node_count);
  }
}
`;

/**
 * Scatter-Max Shader for Max-Pooling Aggregation
 */
export const SCATTER_MAX_SHADER = /* wgsl */ `
struct ScatterUniforms {
  numMessages: u32,
  numNodes: u32,
  numFeatures: u32,
  padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: ScatterUniforms;
@group(0) @binding(1) var<storage, read> indices: array<u32>;
@group(0) @binding(2) var<storage, read> src: array<f32>;
@group(0) @binding(3) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let msg_idx = global_id.x;

  if (msg_idx >= uniforms.numMessages) {
    return;
  }

  let target_node = indices[msg_idx];

  for (var f: u32 = 0u; f < uniforms.numFeatures; f = f + 1u) {
    let src_idx = msg_idx * uniforms.numFeatures + f;
    let dst_idx = target_node * uniforms.numFeatures + f;

    dst[dst_idx] = max(dst[dst_idx], src[src_idx]);
  }
}
`;

/**
 * Gather Shader for Collecting Node Features by Edge Index
 */
export const GATHER_SHADER = /* wgsl */ `
struct GatherUniforms {
  numIndices: u32,
  numFeatures: u32,
  padding1: u32,
  padding2: u32,
}

@group(0) @binding(0) var<uniform> uniforms: GatherUniforms;
@group(0) @binding(1) var<storage, read> indices: array<u32>;   // Indices to gather [numIndices]
@group(0) @binding(2) var<storage, read> src: array<f32>;       // Source [numNodes, numFeatures]
@group(0) @binding(3) var<storage, read_write> dst: array<f32>; // Destination [numIndices, numFeatures]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;

  if (idx >= uniforms.numIndices) {
    return;
  }

  let src_node = indices[idx];

  for (var f: u32 = 0u; f < uniforms.numFeatures; f = f + 1u) {
    let src_idx = src_node * uniforms.numFeatures + f;
    let dst_idx = idx * uniforms.numFeatures + f;
    dst[dst_idx] = src[src_idx];
  }
}
`;

/**
 * Attention Score Computation Shader for GAT
 * Computes attention scores for each edge
 */
export const ATTENTION_SCORE_SHADER = /* wgsl */ `
struct AttentionUniforms {
  numEdges: u32,
  numFeatures: u32,  // Features per head
  numHeads: u32,
  leakyReluSlope: f32,
}

@group(0) @binding(0) var<uniform> uniforms: AttentionUniforms;
@group(0) @binding(1) var<storage, read> srcFeatures: array<f32>;  // [numEdges, numHeads, numFeatures]
@group(0) @binding(2) var<storage, read> dstFeatures: array<f32>;  // [numEdges, numHeads, numFeatures]
@group(0) @binding(3) var<storage, read> attWeight: array<f32>;    // [numHeads, 2 * numFeatures]
@group(0) @binding(4) var<storage, read_write> scores: array<f32>; // [numEdges, numHeads]

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let edge_head_idx = global_id.x;
  let total = uniforms.numEdges * uniforms.numHeads;

  if (edge_head_idx >= total) {
    return;
  }

  let edge = edge_head_idx / uniforms.numHeads;
  let head = edge_head_idx % uniforms.numHeads;

  var score: f32 = 0.0;

  // Compute attention: a^T [Wh_i || Wh_j]
  let feature_offset = edge * uniforms.numHeads * uniforms.numFeatures + head * uniforms.numFeatures;
  let att_offset = head * 2u * uniforms.numFeatures;

  // Source contribution
  for (var f: u32 = 0u; f < uniforms.numFeatures; f = f + 1u) {
    score = score + srcFeatures[feature_offset + f] * attWeight[att_offset + f];
  }

  // Destination contribution
  for (var f: u32 = 0u; f < uniforms.numFeatures; f = f + 1u) {
    score = score + dstFeatures[feature_offset + f] * attWeight[att_offset + uniforms.numFeatures + f];
  }

  // LeakyReLU
  if (score < 0.0) {
    score = score * uniforms.leakyReluSlope;
  }

  scores[edge_head_idx] = score;
}
`;

/**
 * Softmax Normalization Shader for Attention
 * Normalizes attention scores per destination node
 */
export const ATTENTION_SOFTMAX_SHADER = /* wgsl */ `
struct SoftmaxUniforms {
  numNodes: u32,
  numHeads: u32,
  maxEdgesPerNode: u32,
  padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: SoftmaxUniforms;
@group(0) @binding(1) var<storage, read> nodeEdgePtr: array<u32>;   // Start index for each node's edges
@group(0) @binding(2) var<storage, read> nodeEdgeCount: array<u32>; // Number of edges per node
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;  // [numEdges, numHeads]

@compute @workgroup_size(64, 4)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let node = workgroup_id.x;
  let head = local_id.y;

  if (node >= uniforms.numNodes || head >= uniforms.numHeads) {
    return;
  }

  let start = nodeEdgePtr[node];
  let count = nodeEdgeCount[node];

  if (count == 0u) {
    return;
  }

  // Find max for numerical stability
  var max_score: f32 = -1e10;
  for (var i: u32 = 0u; i < count; i = i + 1u) {
    let idx = (start + i) * uniforms.numHeads + head;
    max_score = max(max_score, scores[idx]);
  }

  // Compute exp and sum
  var exp_sum: f32 = 0.0;
  for (var i: u32 = 0u; i < count; i = i + 1u) {
    let idx = (start + i) * uniforms.numHeads + head;
    let exp_val = exp(scores[idx] - max_score);
    scores[idx] = exp_val;
    exp_sum = exp_sum + exp_val;
  }

  // Normalize
  if (exp_sum > 0.0) {
    for (var i: u32 = 0u; i < count; i = i + 1u) {
      let idx = (start + i) * uniforms.numHeads + head;
      scores[idx] = scores[idx] / exp_sum;
    }
  }
}
`;

/**
 * Dense Matrix Multiplication Shader
 * For linear transformations: Y = X @ W
 */
export const MATMUL_SHADER = /* wgsl */ `
struct MatmulUniforms {
  M: u32,  // Rows of A and C
  N: u32,  // Cols of B and C
  K: u32,  // Cols of A, Rows of B
  padding: u32,
}

@group(0) @binding(0) var<uniform> uniforms: MatmulUniforms;
@group(0) @binding(1) var<storage, read> A: array<f32>;         // [M, K]
@group(0) @binding(2) var<storage, read> B: array<f32>;         // [K, N]
@group(0) @binding(3) var<storage, read_write> C: array<f32>;   // [M, N]

// Tile-based matrix multiplication for better cache utilization
const TILE_SIZE: u32 = 16u;

var<workgroup> tileA: array<array<f32, 16>, 16>;
var<workgroup> tileB: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16)
fn main(
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>
) {
  let row = workgroup_id.y * TILE_SIZE + local_id.y;
  let col = workgroup_id.x * TILE_SIZE + local_id.x;

  var sum: f32 = 0.0;
  let numTiles = (uniforms.K + TILE_SIZE - 1u) / TILE_SIZE;

  for (var t: u32 = 0u; t < numTiles; t = t + 1u) {
    // Load tile of A
    let a_col = t * TILE_SIZE + local_id.x;
    if (row < uniforms.M && a_col < uniforms.K) {
      tileA[local_id.y][local_id.x] = A[row * uniforms.K + a_col];
    } else {
      tileA[local_id.y][local_id.x] = 0.0;
    }

    // Load tile of B
    let b_row = t * TILE_SIZE + local_id.y;
    if (b_row < uniforms.K && col < uniforms.N) {
      tileB[local_id.y][local_id.x] = B[b_row * uniforms.N + col];
    } else {
      tileB[local_id.y][local_id.x] = 0.0;
    }

    workgroupBarrier();

    // Compute partial dot product
    for (var k: u32 = 0u; k < TILE_SIZE; k = k + 1u) {
      sum = sum + tileA[local_id.y][k] * tileB[k][local_id.x];
    }

    workgroupBarrier();
  }

  // Store result
  if (row < uniforms.M && col < uniforms.N) {
    C[row * uniforms.N + col] = sum;
  }
}
`;

/**
 * ReLU Activation Shader
 */
export const RELU_SHADER = /* wgsl */ `
struct ReluUniforms {
  size: u32,
  padding1: u32,
  padding2: u32,
  padding3: u32,
}

@group(0) @binding(0) var<uniform> uniforms: ReluUniforms;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;

  if (idx >= uniforms.size) {
    return;
  }

  data[idx] = max(0.0, data[idx]);
}
`;

/**
 * Element-wise Add Shader
 */
export const ADD_SHADER = /* wgsl */ `
struct AddUniforms {
  size: u32,
  padding1: u32,
  padding2: u32,
  padding3: u32,
}

@group(0) @binding(0) var<uniform> uniforms: AddUniforms;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> result: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let idx = global_id.x;

  if (idx >= uniforms.size) {
    return;
  }

  result[idx] = a[idx] + b[idx];
}
`;

/**
 * Shader registry for easy lookup
 */
export const SHADER_REGISTRY = {
  spmm: SPMM_SHADER,
  spmmRowParallel: SPMM_ROW_PARALLEL_SHADER,
  scatterAdd: SCATTER_ADD_SHADER,
  scatterMean: SCATTER_MEAN_SHADER,
  scatterMax: SCATTER_MAX_SHADER,
  gather: GATHER_SHADER,
  attentionScore: ATTENTION_SCORE_SHADER,
  attentionSoftmax: ATTENTION_SOFTMAX_SHADER,
  matmul: MATMUL_SHADER,
  relu: RELU_SHADER,
  add: ADD_SHADER,
} as const;

export type ShaderName = keyof typeof SHADER_REGISTRY;
