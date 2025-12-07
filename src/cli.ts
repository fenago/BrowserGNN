#!/usr/bin/env node

import { existsSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { exec } from 'node:child_process';

const CYAN = '\x1b[36m';
const GREEN = '\x1b[32m';
const YELLOW = '\x1b[33m';
const BOLD = '\x1b[1m';
const DIM = '\x1b[2m';
const RESET = '\x1b[0m';

const logo = `
${CYAN}${BOLD}  ____                                    ____ _   _ _   _
 | __ ) _ __ _____      _____  ___ _ __ / ___| \\ | | \\ | |
 |  _ \\| '__/ _ \\ \\ /\\ / / __|/ _ \\ '__| |  _|  \\| |  \\| |
 | |_) | | | (_) \\ V  V /\\__ \\  __/ |  | |_| | |\\  | |\\  |
 |____/|_|  \\___/ \\_/\\_/ |___/\\___|_|   \\____|_| \\_|_| \\_|${RESET}
                                              ${DIM}by Dr. Lee${RESET}
`;

const links = `
${GREEN}${BOLD}Resources${RESET}

  ${YELLOW}Live Demo${RESET}     https://browsergnn.netlify.app
  ${YELLOW}GitHub${RESET}        https://github.com/fenago/BrowserGNN
  ${YELLOW}npm${RESET}           https://npmjs.com/package/browser-gnn
  ${YELLOW}Docs${RESET}          https://github.com/fenago/BrowserGNN#readme
`;

function showHelp(): void {
  console.log(logo);
  console.log(`${BOLD}The World's First Browser GNN Library${RESET}`);
  console.log(`\nGraph Neural Networks running entirely in your browser.`);
  console.log(`No server required. Privacy-preserving. WebGPU accelerated.\n`);
  console.log(`${GREEN}${BOLD}Commands${RESET}\n`);
  console.log(`  ${YELLOW}npx browser-gnn${RESET}        Show this help`);
  console.log(`  ${YELLOW}npx browser-gnn init${RESET}   Create a starter file`);
  console.log(`  ${YELLOW}npx browser-gnn demo${RESET}   Open the live demo\n`);
  console.log(`${GREEN}${BOLD}Available Layers${RESET}\n`);
  console.log(`  ${CYAN}GCNConv${RESET}   - Graph Convolutional Network`);
  console.log(`  ${CYAN}GATConv${RESET}   - Graph Attention Network`);
  console.log(`  ${CYAN}SAGEConv${RESET}  - GraphSAGE (mean/max/sum aggregation)\n`);
  console.log(links);
}

function createStarterFile(): void {
  const filename = 'gnn-example.mjs';
  const filepath = join(process.cwd(), filename);

  if (existsSync(filepath)) {
    console.log(`${YELLOW}File ${filename} already exists. Skipping.${RESET}`);
    return;
  }

  const content = `// BrowserGNN Example
// Run with: node gnn-example.mjs

import { GraphData, GCNConv, GATConv, SAGEConv, createBrowserGNN } from 'browser-gnn';

async function main() {
  // Initialize BrowserGNN
  const { backend } = await createBrowserGNN();
  console.log('Initialized with backend:', backend);

  // Create a simple graph
  // 4 nodes with 3 features each
  const graph = new GraphData({
    x: new Float32Array([
      1.0, 0.5, 0.2,  // Node 0
      0.3, 1.0, 0.8,  // Node 1
      0.7, 0.2, 1.0,  // Node 2
      0.4, 0.6, 0.3   // Node 3
    ]),
    numNodes: 4,
    numFeatures: 3,
    edgeIndex: new Uint32Array([
      0, 0, 1, 1, 2, 2, 3,  // Sources
      1, 2, 0, 2, 0, 3, 2   // Targets
    ]),
    numEdges: 7
  });

  console.log('\\nGraph:', graph.toString());

  // Try different GNN layers
  console.log('\\n--- GCN Layer ---');
  const gcn = new GCNConv({ inChannels: 3, outChannels: 8 });
  const gcnOut = gcn.forward(graph);
  console.log('Output shape:', gcnOut.x.shape);

  console.log('\\n--- GAT Layer ---');
  const gat = new GATConv({ inChannels: 3, outChannels: 4, heads: 2 });
  const gatOut = gat.forward(graph);
  console.log('Output shape:', gatOut.x.shape);

  console.log('\\n--- GraphSAGE Layer ---');
  const sage = new SAGEConv({ inChannels: 3, outChannels: 8, aggregator: 'mean' });
  const sageOut = sage.forward(graph);
  console.log('Output shape:', sageOut.x.shape);

  console.log('\\nDone! Edit this file to experiment with your own graphs.');
}

main().catch(console.error);
`;

  writeFileSync(filepath, content);
  console.log(logo);
  console.log(`${GREEN}${BOLD}Created ${filename}${RESET}\n`);
  console.log(`Run it with: ${YELLOW}node ${filename}${RESET}\n`);
  console.log(links);
}

function showDemo(): void {
  console.log(logo);
  console.log(`${GREEN}${BOLD}Live Demo${RESET}\n`);
  console.log(`Open in your browser: ${CYAN}https://browsergnn.netlify.app${RESET}\n`);

  // Try to open the URL
  const url = 'https://browsergnn.netlify.app';
  const platform = process.platform;
  let cmd: string;

  if (platform === 'darwin') {
    cmd = `open "${url}"`;
  } else if (platform === 'win32') {
    cmd = `start "${url}"`;
  } else {
    cmd = `xdg-open "${url}"`;
  }

  exec(cmd, (err: Error | null) => {
    if (err) {
      console.log(`${DIM}(Could not open browser automatically)${RESET}`);
    }
  });
}

const commands: Record<string, () => void> = {
  help: showHelp,
  init: createStarterFile,
  demo: showDemo,
};

// Main
const args = process.argv.slice(2);
const command = args[0] || 'help';

const handler = commands[command];
if (handler) {
  handler();
} else {
  console.log(`${YELLOW}Unknown command: ${command}${RESET}\n`);
  showHelp();
}
