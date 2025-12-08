/**
 * BrowserGNN by Dr. Lee
 * Performance Benchmarks
 *
 * Compare CPU vs WebGPU performance for GNN operations.
 * Phase 3: Training benchmarks for autograd, loss, and optimizer performance.
 */

import { Tensor, randomGraph, isGPUAvailable, getGPUStats } from '../core';
import { GCNConv } from '../layers/gcn';
import { SAGEConv } from '../layers/sage';
import { GATConv } from '../layers/gat';
import { initBrowserGNN, getBackendInfo, getBackend } from '../backend';
import { Variable, crossEntropyLoss, mseLoss, Adam, SGD } from '../autograd';

export interface BenchmarkResult {
  name: string;
  numNodes: number;
  numEdges: number;
  cpuTimeMs: number;
  gpuTimeMs: number | null;
  speedup: number | null;
  iterations: number;
}

export interface TrainingBenchmarkResult {
  name: string;
  numNodes: number;
  numEdges: number;
  forwardMs: number;
  backwardMs: number;
  optimizerStepMs: number;
  totalEpochMs: number;
  iterations: number;
}

export interface BenchmarkConfig {
  warmupIterations?: number;
  benchmarkIterations?: number;
  nodeSizes?: number[];
  edgeDensity?: number;
}

export interface TrainingBenchmarkConfig {
  warmupIterations?: number;
  benchmarkIterations?: number;
  nodeSizes?: number[];
  edgeDensity?: number;
  numClasses?: number;
  learningRate?: number;
}

const DEFAULT_CONFIG: Required<BenchmarkConfig> = {
  warmupIterations: 3,
  benchmarkIterations: 10,
  nodeSizes: [100, 500, 1000, 5000, 10000],
  edgeDensity: 10, // average edges per node
};

/**
 * Benchmark a single GNN layer
 */
async function benchmarkLayer(
  layerName: string,
  createLayer: (inChannels: number, outChannels: number) => {
    forward: (input: any) => any;
    forwardAsync?: (input: any) => Promise<any>;
  },
  numNodes: number,
  numEdges: number,
  inChannels: number,
  outChannels: number,
  config: Required<BenchmarkConfig>
): Promise<BenchmarkResult> {
  // Create random graph with approximate edge probability
  // edgeProbability ≈ numEdges / (numNodes * (numNodes - 1))
  const edgeProbability = Math.min(numEdges / (numNodes * (numNodes - 1)), 1);
  const graph = randomGraph(numNodes, edgeProbability, inChannels);

  // Create layer
  const layer = createLayer(inChannels, outChannels);

  // Warmup CPU
  for (let i = 0; i < config.warmupIterations; i++) {
    layer.forward(graph);
  }

  // Benchmark CPU
  const cpuStart = performance.now();
  for (let i = 0; i < config.benchmarkIterations; i++) {
    layer.forward(graph);
  }
  const cpuEnd = performance.now();
  const cpuTimeMs = (cpuEnd - cpuStart) / config.benchmarkIterations;

  // Benchmark GPU if available
  let gpuTimeMs: number | null = null;

  if (isGPUAvailable() && layer.forwardAsync) {
    // Warmup GPU
    for (let i = 0; i < config.warmupIterations; i++) {
      await layer.forwardAsync(graph);
    }

    // Benchmark GPU
    const gpuStart = performance.now();
    for (let i = 0; i < config.benchmarkIterations; i++) {
      await layer.forwardAsync(graph);
    }
    const gpuEnd = performance.now();
    gpuTimeMs = (gpuEnd - gpuStart) / config.benchmarkIterations;
  }

  const speedup = gpuTimeMs !== null ? cpuTimeMs / gpuTimeMs : null;

  return {
    name: layerName,
    numNodes: graph.numNodes,
    numEdges: graph.numEdges,
    cpuTimeMs,
    gpuTimeMs,
    speedup,
    iterations: config.benchmarkIterations,
  };
}

/**
 * Run GCN benchmarks
 */
export async function benchmarkGCN(userConfig?: BenchmarkConfig): Promise<BenchmarkResult[]> {
  const config = { ...DEFAULT_CONFIG, ...userConfig };
  const results: BenchmarkResult[] = [];

  for (const numNodes of config.nodeSizes) {
    const numEdges = numNodes * config.edgeDensity;

    const result = await benchmarkLayer(
      'GCNConv',
      (inCh, outCh) =>
        new GCNConv({
          inChannels: inCh,
          outChannels: outCh,
        }),
      numNodes,
      numEdges,
      64,
      64,
      config
    );

    results.push(result);
  }

  return results;
}

/**
 * Run GraphSAGE benchmarks
 */
export async function benchmarkSAGE(userConfig?: BenchmarkConfig): Promise<BenchmarkResult[]> {
  const config = { ...DEFAULT_CONFIG, ...userConfig };
  const results: BenchmarkResult[] = [];

  for (const numNodes of config.nodeSizes) {
    const numEdges = numNodes * config.edgeDensity;

    const result = await benchmarkLayer(
      'SAGEConv',
      (inCh, outCh) =>
        new SAGEConv({
          inChannels: inCh,
          outChannels: outCh,
          aggregator: 'mean',
        }),
      numNodes,
      numEdges,
      64,
      64,
      config
    );

    results.push(result);
  }

  return results;
}

/**
 * Run GAT benchmarks
 */
export async function benchmarkGAT(userConfig?: BenchmarkConfig): Promise<BenchmarkResult[]> {
  const config = { ...DEFAULT_CONFIG, ...userConfig };
  const results: BenchmarkResult[] = [];

  for (const numNodes of config.nodeSizes) {
    const numEdges = numNodes * config.edgeDensity;

    const result = await benchmarkLayer(
      'GATConv',
      (inCh, outCh) =>
        new GATConv({
          inChannels: inCh,
          outChannels: outCh,
          heads: 4,
        }),
      numNodes,
      numEdges,
      64,
      16,
      config
    );

    results.push(result);
  }

  return results;
}

/**
 * Run all benchmarks
 */
export async function runAllBenchmarks(userConfig?: BenchmarkConfig): Promise<{
  gcn: BenchmarkResult[];
  sage: BenchmarkResult[];
  gat: BenchmarkResult[];
  backendInfo: ReturnType<typeof getBackend>;
  gpuStats: ReturnType<typeof getGPUStats> | null;
}> {
  // Initialize backend
  await initBrowserGNN({ preferredBackend: 'webgpu' });

  const backendInfo = getBackend();
  const gpuStats = isGPUAvailable() ? getGPUStats() : null;

  console.log('Backend Info:', backendInfo);
  console.log('GPU Available:', isGPUAvailable());

  const gcn = await benchmarkGCN(userConfig);
  const sage = await benchmarkSAGE(userConfig);
  const gat = await benchmarkGAT(userConfig);

  return { gcn, sage, gat, backendInfo, gpuStats };
}

/**
 * Format benchmark results as a table
 */
export function formatBenchmarkTable(results: BenchmarkResult[]): string {
  const header = '| Layer | Nodes | Edges | CPU (ms) | GPU (ms) | Speedup |';
  const separator = '|-------|-------|-------|----------|----------|---------|';

  const rows = results.map(r => {
    const gpu = r.gpuTimeMs !== null ? r.gpuTimeMs.toFixed(2) : 'N/A';
    const speedup = r.speedup !== null ? `${r.speedup.toFixed(2)}x` : 'N/A';
    return `| ${r.name} | ${r.numNodes} | ${r.numEdges} | ${r.cpuTimeMs.toFixed(2)} | ${gpu} | ${speedup} |`;
  });

  return [header, separator, ...rows].join('\n');
}

/**
 * Quick benchmark for demo purposes
 */
export async function quickBenchmark(): Promise<string> {
  const config: BenchmarkConfig = {
    warmupIterations: 2,
    benchmarkIterations: 5,
    nodeSizes: [100, 1000, 5000],
    edgeDensity: 10,
  };

  const { gcn, sage, gat, backendInfo } = await runAllBenchmarks(config);

  let output = '# BrowserGNN Benchmark Results\n\n';
  output += `Backend: ${backendInfo}\n`;
  output += `WebGPU Available: ${isGPUAvailable()}\n\n`;

  output += '## GCN Layer\n';
  output += formatBenchmarkTable(gcn) + '\n\n';

  output += '## GraphSAGE Layer\n';
  output += formatBenchmarkTable(sage) + '\n\n';

  output += '## GAT Layer\n';
  output += formatBenchmarkTable(gat) + '\n';

  return output;
}

// Phase 3 Training Benchmarks

const DEFAULT_TRAINING_CONFIG: Required<TrainingBenchmarkConfig> = {
  warmupIterations: 2,
  benchmarkIterations: 10,
  nodeSizes: [100, 500, 1000],
  edgeDensity: 10,
  numClasses: 7,
  learningRate: 0.01,
};

/**
 * Benchmark a single training iteration (forward + loss + backward + optimizer step)
 */
async function benchmarkTrainingIteration(
  layerName: string,
  createLayer: (inChannels: number, outChannels: number) => {
    forward: (input: any) => any;
    parameters: () => Array<{ tensor: Tensor; requiresGrad: boolean }>;
  },
  numNodes: number,
  numEdges: number,
  inChannels: number,
  numClasses: number,
  config: Required<TrainingBenchmarkConfig>
): Promise<TrainingBenchmarkResult> {
  // Create random graph
  const edgeProbability = Math.min(numEdges / (numNodes * (numNodes - 1)), 1);
  const graph = randomGraph(numNodes, edgeProbability, inChannels);

  // Create layer
  const layer = createLayer(inChannels, numClasses);

  // Convert parameters to Variables for optimizer
  const params = layer.parameters();
  const paramVars = params.map(p => new Variable(p.tensor, { requiresGrad: p.requiresGrad }));
  const optimizer = new Adam(paramVars, { lr: config.learningRate });

  // Create random target labels
  const targetData = new Uint32Array(numNodes);
  for (let i = 0; i < numNodes; i++) {
    targetData[i] = Math.floor(Math.random() * numClasses);
  }

  // Warmup
  for (let i = 0; i < config.warmupIterations; i++) {
    const outputGraph = layer.forward(graph);
    const outputTensor = 'x' in outputGraph ? outputGraph.x : outputGraph;
    const outputVar = new Variable(outputTensor, { requiresGrad: true });
    const loss = crossEntropyLoss(outputVar, targetData);
    loss.backward();
    optimizer.stepOptimizer();
    optimizer.zeroGrad();
  }

  // Benchmark forward pass
  let forwardTotal = 0;
  for (let i = 0; i < config.benchmarkIterations; i++) {
    const start = performance.now();
    layer.forward(graph);
    forwardTotal += performance.now() - start;
  }
  const forwardMs = forwardTotal / config.benchmarkIterations;

  // Benchmark backward pass (loss + backward)
  let backwardTotal = 0;
  for (let i = 0; i < config.benchmarkIterations; i++) {
    const outputGraph = layer.forward(graph);
    const outputTensor = 'x' in outputGraph ? outputGraph.x : outputGraph;
    const outputVar = new Variable(outputTensor, { requiresGrad: true });
    const loss = crossEntropyLoss(outputVar, targetData);

    const start = performance.now();
    loss.backward();
    backwardTotal += performance.now() - start;

    optimizer.zeroGrad();
  }
  const backwardMs = backwardTotal / config.benchmarkIterations;

  // Benchmark optimizer step
  let optimizerTotal = 0;
  for (let i = 0; i < config.benchmarkIterations; i++) {
    const outputGraph = layer.forward(graph);
    const outputTensor = 'x' in outputGraph ? outputGraph.x : outputGraph;
    const outputVar = new Variable(outputTensor, { requiresGrad: true });
    const loss = crossEntropyLoss(outputVar, targetData);
    loss.backward();

    const start = performance.now();
    optimizer.stepOptimizer();
    optimizerTotal += performance.now() - start;

    optimizer.zeroGrad();
  }
  const optimizerStepMs = optimizerTotal / config.benchmarkIterations;

  // Benchmark full epoch (forward + backward + optimizer)
  let epochTotal = 0;
  for (let i = 0; i < config.benchmarkIterations; i++) {
    const start = performance.now();

    const outputGraph = layer.forward(graph);
    const outputTensor = 'x' in outputGraph ? outputGraph.x : outputGraph;
    const outputVar = new Variable(outputTensor, { requiresGrad: true });
    const loss = crossEntropyLoss(outputVar, targetData);
    loss.backward();
    optimizer.stepOptimizer();
    optimizer.zeroGrad();

    epochTotal += performance.now() - start;
  }
  const totalEpochMs = epochTotal / config.benchmarkIterations;

  return {
    name: layerName,
    numNodes: graph.numNodes,
    numEdges: graph.numEdges,
    forwardMs,
    backwardMs,
    optimizerStepMs,
    totalEpochMs,
    iterations: config.benchmarkIterations,
  };
}

/**
 * Run training benchmarks for GCN
 */
export async function benchmarkGCNTraining(
  userConfig?: TrainingBenchmarkConfig
): Promise<TrainingBenchmarkResult[]> {
  const config = { ...DEFAULT_TRAINING_CONFIG, ...userConfig };
  const results: TrainingBenchmarkResult[] = [];

  for (const numNodes of config.nodeSizes) {
    const numEdges = numNodes * config.edgeDensity;

    const result = await benchmarkTrainingIteration(
      'GCNConv Training',
      (inCh, outCh) =>
        new GCNConv({
          inChannels: inCh,
          outChannels: outCh,
        }),
      numNodes,
      numEdges,
      64,
      config.numClasses,
      config
    );

    results.push(result);
  }

  return results;
}

/**
 * Run training benchmarks for GraphSAGE
 */
export async function benchmarkSAGETraining(
  userConfig?: TrainingBenchmarkConfig
): Promise<TrainingBenchmarkResult[]> {
  const config = { ...DEFAULT_TRAINING_CONFIG, ...userConfig };
  const results: TrainingBenchmarkResult[] = [];

  for (const numNodes of config.nodeSizes) {
    const numEdges = numNodes * config.edgeDensity;

    const result = await benchmarkTrainingIteration(
      'SAGEConv Training',
      (inCh, outCh) =>
        new SAGEConv({
          inChannels: inCh,
          outChannels: outCh,
          aggregator: 'mean',
        }),
      numNodes,
      numEdges,
      64,
      config.numClasses,
      config
    );

    results.push(result);
  }

  return results;
}

/**
 * Run training benchmarks for GAT
 */
export async function benchmarkGATTraining(
  userConfig?: TrainingBenchmarkConfig
): Promise<TrainingBenchmarkResult[]> {
  const config = { ...DEFAULT_TRAINING_CONFIG, ...userConfig };
  const results: TrainingBenchmarkResult[] = [];

  for (const numNodes of config.nodeSizes) {
    const numEdges = numNodes * config.edgeDensity;

    const result = await benchmarkTrainingIteration(
      'GATConv Training',
      (inCh, outCh) =>
        new GATConv({
          inChannels: inCh,
          outChannels: outCh,
          heads: 4,
        }),
      numNodes,
      numEdges,
      64,
      config.numClasses,
      config
    );

    results.push(result);
  }

  return results;
}

/**
 * Run all training benchmarks
 */
export async function runAllTrainingBenchmarks(userConfig?: TrainingBenchmarkConfig): Promise<{
  gcn: TrainingBenchmarkResult[];
  sage: TrainingBenchmarkResult[];
  gat: TrainingBenchmarkResult[];
  backendInfo: ReturnType<typeof getBackend>;
}> {
  // Initialize backend
  await initBrowserGNN({ preferredBackend: 'cpu' });

  const backendInfo = getBackend();

  console.log('Training Benchmark - Backend Info:', backendInfo);

  const gcn = await benchmarkGCNTraining(userConfig);
  const sage = await benchmarkSAGETraining(userConfig);
  const gat = await benchmarkGATTraining(userConfig);

  return { gcn, sage, gat, backendInfo };
}

/**
 * Format training benchmark results as a table
 */
export function formatTrainingBenchmarkTable(results: TrainingBenchmarkResult[]): string {
  const header = '| Layer | Nodes | Edges | Forward (ms) | Backward (ms) | Optimizer (ms) | Total (ms) |';
  const separator = '|-------|-------|-------|--------------|---------------|----------------|------------|';

  const rows = results.map(r => {
    return `| ${r.name} | ${r.numNodes} | ${r.numEdges} | ${r.forwardMs.toFixed(2)} | ${r.backwardMs.toFixed(2)} | ${r.optimizerStepMs.toFixed(2)} | ${r.totalEpochMs.toFixed(2)} |`;
  });

  return [header, separator, ...rows].join('\n');
}

/**
 * Quick training benchmark for demo purposes
 */
export async function quickTrainingBenchmark(): Promise<string> {
  const config: TrainingBenchmarkConfig = {
    warmupIterations: 1,
    benchmarkIterations: 5,
    nodeSizes: [100, 500, 1000],
    edgeDensity: 10,
    numClasses: 7,
    learningRate: 0.01,
  };

  const { gcn, sage, gat, backendInfo } = await runAllTrainingBenchmarks(config);

  let output = '# BrowserGNN Training Benchmark Results\n\n';
  output += `Backend: ${backendInfo}\n\n`;

  output += '## GCN Layer Training\n';
  output += formatTrainingBenchmarkTable(gcn) + '\n\n';

  output += '## GraphSAGE Layer Training\n';
  output += formatTrainingBenchmarkTable(sage) + '\n\n';

  output += '## GAT Layer Training\n';
  output += formatTrainingBenchmarkTable(gat) + '\n';

  return output;
}
