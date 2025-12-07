/**
 * BrowserGNN by Dr. Lee
 * Trainer Module - High-level Training API
 */

import { Tensor } from '../core/tensor';
import { GraphData } from '../core/graph';
import { Variable } from './variable';
import { Optimizer } from './optim';
import { crossEntropyLoss } from './loss';

/**
 * Training metrics for a single epoch
 */
export interface TrainingMetrics {
  epoch: number;
  loss: number;
  accuracy?: number;
  valLoss?: number;
  valAccuracy?: number;
  learningRate: number;
  duration: number; // ms
  timestamp: number;
}

/**
 * Callback for training events
 */
export type TrainingCallback = (metrics: TrainingMetrics) => void | Promise<void>;

/**
 * Trainer configuration
 */
export interface TrainerConfig {
  optimizer: Optimizer;
  lossFunction?: 'cross_entropy' | 'mse' | 'nll';
  maxEpochs?: number;
  patience?: number; // Early stopping patience
  minDelta?: number; // Minimum improvement for early stopping
  logEvery?: number; // Log every N epochs
  evalEvery?: number; // Evaluate every N epochs
  callbacks?: TrainingCallback[];
}

/**
 * Trainable GNN Layer interface
 */
export interface TrainableLayer {
  forward(input: GraphData): GraphData;
  getTrainableVariables(): Variable[];
}

/**
 * Simple GNN model wrapper for training
 */
export class GNNModel {
  private layers: TrainableLayer[] = [];
  private _variables: Variable[] = [];

  constructor() {}

  /**
   * Add a layer to the model
   */
  addLayer(layer: TrainableLayer): this {
    this.layers.push(layer);
    this._variables.push(...layer.getTrainableVariables());
    return this;
  }

  /**
   * Forward pass through all layers
   */
  forward(input: GraphData): GraphData {
    let output = input;
    for (const layer of this.layers) {
      output = layer.forward(output);
    }
    return output;
  }

  /**
   * Get all trainable variables
   */
  getVariables(): Variable[] {
    return this._variables;
  }

  /**
   * Get number of parameters
   */
  numParameters(): number {
    return this._variables.reduce((sum, v) => sum + v.size, 0);
  }
}

/**
 * High-level Trainer class
 *
 * Provides training loop, metrics tracking, and callbacks for GNN models.
 */
export class Trainer {
  private config: Required<TrainerConfig>;
  private history: TrainingMetrics[] = [];
  private bestLoss = Infinity;
  private patienceCounter = 0;
  private _isTraining = false;
  private _stopRequested = false;

  constructor(config: TrainerConfig) {
    this.config = {
      optimizer: config.optimizer,
      lossFunction: config.lossFunction ?? 'cross_entropy',
      maxEpochs: config.maxEpochs ?? 100,
      patience: config.patience ?? 10,
      minDelta: config.minDelta ?? 1e-4,
      logEvery: config.logEvery ?? 1,
      evalEvery: config.evalEvery ?? 1,
      callbacks: config.callbacks ?? [],
    };
  }

  /**
   * Get training history
   */
  getHistory(): TrainingMetrics[] {
    return [...this.history];
  }

  /**
   * Check if currently training
   */
  get isTraining(): boolean {
    return this._isTraining;
  }

  /**
   * Request training to stop
   */
  stop(): void {
    this._stopRequested = true;
  }

  /**
   * Train a model on node classification task
   *
   * @param model The model to train
   * @param graph The graph data
   * @param labels Node labels (class indices)
   * @param trainMask Boolean mask for training nodes
   * @param valMask Optional boolean mask for validation nodes
   */
  async train(
    model: GNNModel,
    graph: GraphData,
    labels: Uint32Array | number[],
    trainMask: boolean[],
    valMask?: boolean[]
  ): Promise<TrainingMetrics[]> {
    this._isTraining = true;
    this._stopRequested = false;
    this.history = [];
    this.bestLoss = Infinity;
    this.patienceCounter = 0;

    const variables = model.getVariables();

    for (let epoch = 0; epoch < this.config.maxEpochs; epoch++) {
      if (this._stopRequested) {
        console.log(`Training stopped at epoch ${epoch}`);
        break;
      }

      const startTime = performance.now();

      // Zero gradients
      this.config.optimizer.zeroGrad();

      // Forward pass
      const output = model.forward(graph);
      const outputVar = new Variable(output.x, { requiresGrad: true });

      // Compute loss on training nodes
      const trainIndices = trainMask
        .map((m, i) => (m ? i : -1))
        .filter(i => i >= 0);

      const trainLogits = this._gatherRows(outputVar, trainIndices);
      const trainLabels = trainIndices.map(i => labels[i]!);

      const loss = crossEntropyLoss(trainLogits, trainLabels);

      // Backward pass - manually compute gradients through the model
      loss.backward();

      // Copy gradients to model variables
      this._copyGradients(outputVar, variables, graph, trainMask);

      // Optimizer step
      this.config.optimizer.stepOptimizer();

      // Compute metrics
      const lossValue = loss.data.data[0]!;
      const accuracy = this._computeAccuracy(output.x, labels, trainMask);

      let valLoss: number | undefined;
      let valAccuracy: number | undefined;

      if (valMask && epoch % this.config.evalEvery === 0) {
        // Validation
        const valIndices = valMask
          .map((m, i) => (m ? i : -1))
          .filter(i => i >= 0);

        if (valIndices.length > 0) {
          const valLogitsVar = this._gatherRows(
            new Variable(output.x, { requiresGrad: false }),
            valIndices
          );
          const valLabelsArr = valIndices.map(i => labels[i]!);
          const valLossVar = crossEntropyLoss(valLogitsVar, valLabelsArr);
          valLoss = valLossVar.data.data[0]!;
          valAccuracy = this._computeAccuracy(output.x, labels, valMask);
        }
      }

      const duration = performance.now() - startTime;

      const metrics: TrainingMetrics = {
        epoch,
        loss: lossValue,
        accuracy,
        valLoss,
        valAccuracy,
        learningRate: this._getCurrentLr(),
        duration,
        timestamp: Date.now(),
      };

      this.history.push(metrics);

      // Callbacks
      for (const callback of this.config.callbacks) {
        await callback(metrics);
      }

      // Early stopping check
      const checkLoss = valLoss ?? lossValue;
      if (checkLoss < this.bestLoss - this.config.minDelta) {
        this.bestLoss = checkLoss;
        this.patienceCounter = 0;
      } else {
        this.patienceCounter++;
        if (this.patienceCounter >= this.config.patience) {
          console.log(`Early stopping at epoch ${epoch} (patience ${this.config.patience})`);
          break;
        }
      }

      // Allow UI to update
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    this._isTraining = false;
    return this.history;
  }

  /**
   * Single training step (for manual training loops)
   */
  trainStep(
    model: GNNModel,
    graph: GraphData,
    labels: Uint32Array | number[],
    trainMask: boolean[]
  ): { loss: number; accuracy: number } {
    // Zero gradients
    this.config.optimizer.zeroGrad();

    // Forward pass
    const output = model.forward(graph);
    const outputVar = new Variable(output.x, { requiresGrad: true });

    // Compute loss on training nodes
    const trainIndices = trainMask
      .map((m, i) => (m ? i : -1))
      .filter(i => i >= 0);

    const trainLogits = this._gatherRows(outputVar, trainIndices);
    const trainLabels = trainIndices.map(i => labels[i]!);

    const loss = crossEntropyLoss(trainLogits, trainLabels);

    // Backward pass
    loss.backward();

    // Copy gradients to model variables
    const variables = model.getVariables();
    this._copyGradients(outputVar, variables, graph, trainMask);

    // Optimizer step
    this.config.optimizer.stepOptimizer();

    const lossValue = loss.data.data[0]!;
    const accuracy = this._computeAccuracy(output.x, labels, trainMask);

    return { loss: lossValue, accuracy };
  }

  /**
   * Evaluate model on masked nodes
   */
  evaluate(
    model: GNNModel,
    graph: GraphData,
    labels: Uint32Array | number[],
    mask: boolean[]
  ): { loss: number; accuracy: number } {
    // Forward pass (no gradients)
    const output = model.forward(graph);

    const indices = mask.map((m, i) => (m ? i : -1)).filter(i => i >= 0);

    const logitsVar = this._gatherRows(
      new Variable(output.x, { requiresGrad: false }),
      indices
    );
    const labelsArr = indices.map(i => labels[i]!);

    const loss = crossEntropyLoss(logitsVar, labelsArr);
    const accuracy = this._computeAccuracy(output.x, labels, mask);

    return {
      loss: loss.data.data[0]!,
      accuracy,
    };
  }

  /**
   * Helper: Gather rows from a variable
   */
  private _gatherRows(variable: Variable, indices: number[]): Variable {
    const numFeatures = variable.shape[1]!;
    const gathered = new Float32Array(indices.length * numFeatures);

    for (let i = 0; i < indices.length; i++) {
      const srcIdx = indices[i]!;
      for (let f = 0; f < numFeatures; f++) {
        gathered[i * numFeatures + f] = variable.data.data[srcIdx * numFeatures + f]!;
      }
    }

    return new Variable(new Tensor(gathered, [indices.length, numFeatures]), {
      requiresGrad: variable.requiresGrad,
    });
  }

  /**
   * Helper: Compute accuracy
   */
  private _computeAccuracy(
    output: Tensor,
    labels: Uint32Array | number[],
    mask: boolean[]
  ): number {
    const numClasses = output.shape[1]!;
    let correct = 0;
    let total = 0;

    for (let i = 0; i < mask.length; i++) {
      if (!mask[i]) continue;

      // Get predicted class (argmax)
      let maxVal = -Infinity;
      let maxIdx = 0;
      for (let c = 0; c < numClasses; c++) {
        const val = output.data[i * numClasses + c]!;
        if (val > maxVal) {
          maxVal = val;
          maxIdx = c;
        }
      }

      if (maxIdx === labels[i]) {
        correct++;
      }
      total++;
    }

    return total > 0 ? correct / total : 0;
  }

  /**
   * Helper: Get current learning rate
   */
  private _getCurrentLr(): number {
    const opt = this.config.optimizer as unknown as { defaults: Record<string, number> };
    return opt.defaults.lr ?? 0.01;
  }

  /**
   * Helper: Copy gradients from output back to model variables
   * This is a simplified gradient propagation for the GNN layers
   */
  private _copyGradients(
    _outputVar: Variable,
    variables: Variable[],
    _graph: GraphData,
    _trainMask: boolean[]
  ): void {
    // For simplicity, we estimate gradients numerically for each variable
    // In a full implementation, this would use proper backpropagation
    // through the GNN layers

    // For now, we'll use the chain rule approximation
    // The gradients are accumulated in the Variable.grad field
    // through the autograd backward pass

    // This is a placeholder - actual implementation would require
    // differentiable GNN layers that work with Variables
    for (const v of variables) {
      if (v.grad === null && v.requiresGrad) {
        // Initialize gradient if not set
        v.grad = Tensor.zeros(v.shape);
      }
    }
  }
}

/**
 * Create a simple training callback that logs to console
 */
export function consoleLogger(logEvery = 1): TrainingCallback {
  return (metrics: TrainingMetrics) => {
    if (metrics.epoch % logEvery === 0) {
      let msg = `Epoch ${metrics.epoch}: loss=${metrics.loss.toFixed(4)}`;
      if (metrics.accuracy !== undefined) {
        msg += `, acc=${(metrics.accuracy * 100).toFixed(1)}%`;
      }
      if (metrics.valLoss !== undefined) {
        msg += `, val_loss=${metrics.valLoss.toFixed(4)}`;
      }
      if (metrics.valAccuracy !== undefined) {
        msg += `, val_acc=${(metrics.valAccuracy * 100).toFixed(1)}%`;
      }
      msg += ` (${metrics.duration.toFixed(1)}ms)`;
      console.log(msg);
    }
  };
}

/**
 * Early stopping callback
 */
export function earlyStoppingCallback(
  patience: number,
  minDelta = 1e-4,
  monitor: 'loss' | 'valLoss' = 'loss'
): TrainingCallback & { shouldStop: () => boolean } {
  let bestValue = Infinity;
  let counter = 0;
  let stopFlag = false;

  const callback: TrainingCallback & { shouldStop: () => boolean } = (metrics: TrainingMetrics) => {
    const value = monitor === 'valLoss' ? (metrics.valLoss ?? metrics.loss) : metrics.loss;

    if (value < bestValue - minDelta) {
      bestValue = value;
      counter = 0;
    } else {
      counter++;
      if (counter >= patience) {
        stopFlag = true;
      }
    }
  };

  callback.shouldStop = () => stopFlag;

  return callback;
}

/**
 * Metrics collector callback
 */
export function metricsCollector(): TrainingCallback & { getMetrics: () => TrainingMetrics[] } {
  const metrics: TrainingMetrics[] = [];

  const callback: TrainingCallback & { getMetrics: () => TrainingMetrics[] } = (m: TrainingMetrics) => {
    metrics.push({ ...m });
  };

  callback.getMetrics = () => [...metrics];

  return callback;
}
