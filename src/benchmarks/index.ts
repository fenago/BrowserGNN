/**
 * BrowserGNN by Dr. Lee
 * Performance Benchmarks
 *
 * Compare CPU vs WebGPU performance for GNN operations.
 */

import { Tensor, randomGraph, isGPUAvailable, getGPUStats } from '../core';
import { GCNConv } from '../layers/gcn';
import { SAGEConv } from '../layers/sage';
import { GATConv } from '../layers/gat';
import { initBrowserGNN, getBackendInfo, getBackend } from '../backend';

export interface BenchmarkResult {
  name: string;
  numNodes: number;
  numEdges: number;
  cpuTimeMs: number;
  gpuTimeMs: number | null;
  speedup: number | null;
  iterations: number;
}

export interface BenchmarkConfig {
  warmupIterations?: number;
  benchmarkIterations?: number;
  nodeSizes?: number[];
  edgeDensity?: number;
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
