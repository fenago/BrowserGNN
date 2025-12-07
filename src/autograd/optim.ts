/**
 * BrowserGNN by Dr. Lee
 * Optimizers for Training
 */

import { Variable } from './variable';

/**
 * Parameter group with optional per-group settings
 */
export interface ParamGroup {
  params: Variable[];
  lr?: number;
  weightDecay?: number;
}

/**
 * Base Optimizer class
 */
export abstract class Optimizer {
  protected paramGroups: ParamGroup[];
  protected defaults: Record<string, number>;
  protected _step = 0;

  constructor(params: Variable[] | ParamGroup[], defaults: Record<string, number> = {}) {
    this.defaults = defaults;

    if (params.length > 0 && 'params' in (params[0] as ParamGroup)) {
      this.paramGroups = params as ParamGroup[];
    } else {
      this.paramGroups = [{ params: params as Variable[] }];
    }
  }

  /**
   * Zero all gradients
   */
  zeroGrad(): void {
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        param.zeroGrad();
      }
    }
  }

  /**
   * Get current step count
   */
  get step(): number {
    return this._step;
  }

  /**
   * Perform a single optimization step
   */
  abstract stepOptimizer(): void;

  /**
   * Get all parameters
   */
  getParams(): Variable[] {
    const allParams: Variable[] = [];
    for (const group of this.paramGroups) {
      allParams.push(...group.params);
    }
    return allParams;
  }

  /**
   * Get state dict for serialization
   */
  abstract stateDict(): Record<string, unknown>;

  /**
   * Load state dict
   */
  abstract loadStateDict(stateDict: Record<string, unknown>): void;
}

/**
 * Stochastic Gradient Descent Optimizer
 *
 * Supports momentum and weight decay.
 *
 * v_t = momentum * v_{t-1} + grad
 * param = param - lr * v_t - lr * weightDecay * param
 */
export class SGD extends Optimizer {
  private momentum: number;
  private dampening: number;
  private nesterov: boolean;
  private velocities: Map<Variable, Float32Array> = new Map();

  constructor(
    params: Variable[] | ParamGroup[],
    options: {
      lr?: number;
      momentum?: number;
      dampening?: number;
      weightDecay?: number;
      nesterov?: boolean;
    } = {}
  ) {
    super(params, {
      lr: options.lr ?? 0.01,
      weightDecay: options.weightDecay ?? 0,
    });

    this.momentum = options.momentum ?? 0;
    this.dampening = options.dampening ?? 0;
    this.nesterov = options.nesterov ?? false;

    if (this.nesterov && (this.momentum <= 0 || this.dampening !== 0)) {
      throw new Error('Nesterov momentum requires momentum > 0 and dampening = 0');
    }
  }

  stepOptimizer(): void {
    this._step++;

    for (const group of this.paramGroups) {
      const lr = group.lr ?? this.defaults.lr!;
      const weightDecay = group.weightDecay ?? this.defaults.weightDecay!;

      for (const param of group.params) {
        if (!param.grad) continue;

        const grad = param.grad;

        // Apply weight decay
        if (weightDecay !== 0) {
          for (let i = 0; i < grad.size; i++) {
            grad.data[i] += weightDecay * param.data.data[i]!;
          }
        }

        // Apply momentum
        if (this.momentum !== 0) {
          let velocity = this.velocities.get(param);

          if (!velocity) {
            // First step: initialize velocity with gradient
            velocity = new Float32Array(grad.size);
            velocity.set(grad.data);
            this.velocities.set(param, velocity);
          } else {
            // Update velocity: v = momentum * v + (1 - dampening) * grad
            for (let i = 0; i < grad.size; i++) {
              velocity[i] = this.momentum * velocity[i]! + (1 - this.dampening) * grad.data[i]!;
            }
          }

          if (this.nesterov) {
            // Nesterov: use grad + momentum * velocity
            for (let i = 0; i < grad.size; i++) {
              param.data.data[i] -= lr * (grad.data[i]! + this.momentum * velocity[i]!);
            }
          } else {
            // Standard momentum: use velocity directly
            for (let i = 0; i < grad.size; i++) {
              param.data.data[i] -= lr * velocity[i]!;
            }
          }
        } else {
          // No momentum: simple gradient descent
          for (let i = 0; i < grad.size; i++) {
            param.data.data[i] -= lr * grad.data[i]!;
          }
        }
      }
    }
  }

  stateDict(): Record<string, unknown> {
    const state: Record<string, unknown> = {
      step: this._step,
      momentum: this.momentum,
      dampening: this.dampening,
      nesterov: this.nesterov,
      defaults: this.defaults,
      velocities: [] as Float32Array[],
    };

    // Save velocities in order
    const velocityList: Float32Array[] = [];
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const v = this.velocities.get(param);
        velocityList.push(v ? new Float32Array(v) : new Float32Array(0));
      }
    }
    state.velocities = velocityList;

    return state;
  }

  loadStateDict(stateDict: Record<string, unknown>): void {
    this._step = stateDict.step as number;
    this.momentum = stateDict.momentum as number;
    this.dampening = stateDict.dampening as number;
    this.nesterov = stateDict.nesterov as boolean;
    this.defaults = stateDict.defaults as Record<string, number>;

    const velocityList = stateDict.velocities as Float32Array[];
    let idx = 0;
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const v = velocityList[idx++];
        if (v && v.length > 0) {
          this.velocities.set(param, v);
        }
      }
    }
  }
}

/**
 * Adam Optimizer
 *
 * Adaptive Moment Estimation with bias correction.
 *
 * m_t = beta1 * m_{t-1} + (1 - beta1) * grad
 * v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2
 * m_hat = m_t / (1 - beta1^t)
 * v_hat = v_t / (1 - beta2^t)
 * param = param - lr * m_hat / (sqrt(v_hat) + eps)
 */
export class Adam extends Optimizer {
  private beta1: number;
  private beta2: number;
  private eps: number;
  private amsgrad: boolean;
  private m: Map<Variable, Float32Array> = new Map(); // First moment
  private v: Map<Variable, Float32Array> = new Map(); // Second moment
  private vMax: Map<Variable, Float32Array> = new Map(); // AMSGrad max

  constructor(
    params: Variable[] | ParamGroup[],
    options: {
      lr?: number;
      beta1?: number;
      beta2?: number;
      eps?: number;
      weightDecay?: number;
      amsgrad?: boolean;
    } = {}
  ) {
    super(params, {
      lr: options.lr ?? 0.001,
      weightDecay: options.weightDecay ?? 0,
    });

    this.beta1 = options.beta1 ?? 0.9;
    this.beta2 = options.beta2 ?? 0.999;
    this.eps = options.eps ?? 1e-8;
    this.amsgrad = options.amsgrad ?? false;
  }

  stepOptimizer(): void {
    this._step++;

    // Bias correction terms
    const biasCorrection1 = 1 - Math.pow(this.beta1, this._step);
    const biasCorrection2 = 1 - Math.pow(this.beta2, this._step);

    for (const group of this.paramGroups) {
      const lr = group.lr ?? this.defaults.lr!;
      const weightDecay = group.weightDecay ?? this.defaults.weightDecay!;

      for (const param of group.params) {
        if (!param.grad) continue;

        const grad = param.grad;

        // Apply weight decay (AdamW style - decoupled)
        if (weightDecay !== 0) {
          for (let i = 0; i < param.data.size; i++) {
            param.data.data[i] -= lr * weightDecay * param.data.data[i]!;
          }
        }

        // Initialize moment estimates if needed
        let mState = this.m.get(param);
        let vState = this.v.get(param);

        if (!mState) {
          mState = new Float32Array(grad.size);
          this.m.set(param, mState);
        }
        if (!vState) {
          vState = new Float32Array(grad.size);
          this.v.set(param, vState);
        }

        // Update biased first moment estimate
        for (let i = 0; i < grad.size; i++) {
          mState[i] = this.beta1 * mState[i]! + (1 - this.beta1) * grad.data[i]!;
        }

        // Update biased second moment estimate
        for (let i = 0; i < grad.size; i++) {
          vState[i] = this.beta2 * vState[i]! + (1 - this.beta2) * grad.data[i]! * grad.data[i]!;
        }

        // Compute denominator
        let denom: Float32Array;

        if (this.amsgrad) {
          let vMaxState = this.vMax.get(param);
          if (!vMaxState) {
            vMaxState = new Float32Array(grad.size);
            this.vMax.set(param, vMaxState);
          }

          // Update vMax
          for (let i = 0; i < grad.size; i++) {
            vMaxState[i] = Math.max(vMaxState[i]!, vState[i]!);
          }

          // Use vMax for denominator
          denom = new Float32Array(grad.size);
          for (let i = 0; i < grad.size; i++) {
            denom[i] = Math.sqrt(vMaxState[i]! / biasCorrection2) + this.eps;
          }
        } else {
          denom = new Float32Array(grad.size);
          for (let i = 0; i < grad.size; i++) {
            denom[i] = Math.sqrt(vState[i]! / biasCorrection2) + this.eps;
          }
        }

        // Update parameters
        const stepSize = lr / biasCorrection1;
        for (let i = 0; i < param.data.size; i++) {
          param.data.data[i] -= stepSize * mState[i]! / denom[i]!;
        }
      }
    }
  }

  stateDict(): Record<string, unknown> {
    const state: Record<string, unknown> = {
      step: this._step,
      beta1: this.beta1,
      beta2: this.beta2,
      eps: this.eps,
      amsgrad: this.amsgrad,
      defaults: this.defaults,
      m: [] as Float32Array[],
      v: [] as Float32Array[],
      vMax: [] as Float32Array[],
    };

    const mList: Float32Array[] = [];
    const vList: Float32Array[] = [];
    const vMaxList: Float32Array[] = [];

    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const mVal = this.m.get(param);
        const vVal = this.v.get(param);
        const vMaxVal = this.vMax.get(param);

        mList.push(mVal ? new Float32Array(mVal) : new Float32Array(0));
        vList.push(vVal ? new Float32Array(vVal) : new Float32Array(0));
        vMaxList.push(vMaxVal ? new Float32Array(vMaxVal) : new Float32Array(0));
      }
    }

    state.m = mList;
    state.v = vList;
    state.vMax = vMaxList;

    return state;
  }

  loadStateDict(stateDict: Record<string, unknown>): void {
    this._step = stateDict.step as number;
    this.beta1 = stateDict.beta1 as number;
    this.beta2 = stateDict.beta2 as number;
    this.eps = stateDict.eps as number;
    this.amsgrad = stateDict.amsgrad as boolean;
    this.defaults = stateDict.defaults as Record<string, number>;

    const mList = stateDict.m as Float32Array[];
    const vList = stateDict.v as Float32Array[];
    const vMaxList = stateDict.vMax as Float32Array[];

    let idx = 0;
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const mVal = mList[idx];
        const vVal = vList[idx];
        const vMaxVal = vMaxList[idx];

        if (mVal && mVal.length > 0) this.m.set(param, mVal);
        if (vVal && vVal.length > 0) this.v.set(param, vVal);
        if (vMaxVal && vMaxVal.length > 0) this.vMax.set(param, vMaxVal);

        idx++;
      }
    }
  }
}

/**
 * AdaGrad Optimizer
 *
 * Adaptive gradient with accumulated squared gradients.
 */
export class Adagrad extends Optimizer {
  private eps: number;
  private sumSquares: Map<Variable, Float32Array> = new Map();

  constructor(
    params: Variable[] | ParamGroup[],
    options: {
      lr?: number;
      eps?: number;
      weightDecay?: number;
    } = {}
  ) {
    super(params, {
      lr: options.lr ?? 0.01,
      weightDecay: options.weightDecay ?? 0,
    });

    this.eps = options.eps ?? 1e-10;
  }

  stepOptimizer(): void {
    this._step++;

    for (const group of this.paramGroups) {
      const lr = group.lr ?? this.defaults.lr!;
      const weightDecay = group.weightDecay ?? this.defaults.weightDecay!;

      for (const param of group.params) {
        if (!param.grad) continue;

        const grad = param.grad;

        // Apply weight decay
        if (weightDecay !== 0) {
          for (let i = 0; i < grad.size; i++) {
            grad.data[i] += weightDecay * param.data.data[i]!;
          }
        }

        // Initialize sum of squares
        let sumSq = this.sumSquares.get(param);
        if (!sumSq) {
          sumSq = new Float32Array(grad.size);
          this.sumSquares.set(param, sumSq);
        }

        // Accumulate squared gradients
        for (let i = 0; i < grad.size; i++) {
          sumSq[i] += grad.data[i]! * grad.data[i]!;
        }

        // Update parameters
        for (let i = 0; i < param.data.size; i++) {
          param.data.data[i] -= lr * grad.data[i]! / (Math.sqrt(sumSq[i]!) + this.eps);
        }
      }
    }
  }

  stateDict(): Record<string, unknown> {
    const sumSquaresList: Float32Array[] = [];
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const ss = this.sumSquares.get(param);
        sumSquaresList.push(ss ? new Float32Array(ss) : new Float32Array(0));
      }
    }

    return {
      step: this._step,
      eps: this.eps,
      defaults: this.defaults,
      sumSquares: sumSquaresList,
    };
  }

  loadStateDict(stateDict: Record<string, unknown>): void {
    this._step = stateDict.step as number;
    this.eps = stateDict.eps as number;
    this.defaults = stateDict.defaults as Record<string, number>;

    const sumSquaresList = stateDict.sumSquares as Float32Array[];
    let idx = 0;
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const ss = sumSquaresList[idx++];
        if (ss && ss.length > 0) {
          this.sumSquares.set(param, ss);
        }
      }
    }
  }
}

/**
 * RMSprop Optimizer
 */
export class RMSprop extends Optimizer {
  private alpha: number;
  private eps: number;
  private momentum: number;
  private centered: boolean;
  private squareAvg: Map<Variable, Float32Array> = new Map();
  private gradAvg: Map<Variable, Float32Array> = new Map();
  private momentumBuffer: Map<Variable, Float32Array> = new Map();

  constructor(
    params: Variable[] | ParamGroup[],
    options: {
      lr?: number;
      alpha?: number;
      eps?: number;
      weightDecay?: number;
      momentum?: number;
      centered?: boolean;
    } = {}
  ) {
    super(params, {
      lr: options.lr ?? 0.01,
      weightDecay: options.weightDecay ?? 0,
    });

    this.alpha = options.alpha ?? 0.99;
    this.eps = options.eps ?? 1e-8;
    this.momentum = options.momentum ?? 0;
    this.centered = options.centered ?? false;
  }

  stepOptimizer(): void {
    this._step++;

    for (const group of this.paramGroups) {
      const lr = group.lr ?? this.defaults.lr!;
      const weightDecay = group.weightDecay ?? this.defaults.weightDecay!;

      for (const param of group.params) {
        if (!param.grad) continue;

        const grad = param.grad;

        // Apply weight decay
        if (weightDecay !== 0) {
          for (let i = 0; i < grad.size; i++) {
            grad.data[i] += weightDecay * param.data.data[i]!;
          }
        }

        // Initialize state
        let sqAvg = this.squareAvg.get(param);
        if (!sqAvg) {
          sqAvg = new Float32Array(grad.size);
          this.squareAvg.set(param, sqAvg);
        }

        // Update running average of squared gradients
        for (let i = 0; i < grad.size; i++) {
          sqAvg[i] = this.alpha * sqAvg[i]! + (1 - this.alpha) * grad.data[i]! * grad.data[i]!;
        }

        let avg: Float32Array;

        if (this.centered) {
          let gAvg = this.gradAvg.get(param);
          if (!gAvg) {
            gAvg = new Float32Array(grad.size);
            this.gradAvg.set(param, gAvg);
          }

          // Update running average of gradients
          for (let i = 0; i < grad.size; i++) {
            gAvg[i] = this.alpha * gAvg[i]! + (1 - this.alpha) * grad.data[i]!;
          }

          // Centered version
          avg = new Float32Array(grad.size);
          for (let i = 0; i < grad.size; i++) {
            avg[i] = sqAvg[i]! - gAvg[i]! * gAvg[i]!;
          }
        } else {
          avg = sqAvg;
        }

        if (this.momentum > 0) {
          let buf = this.momentumBuffer.get(param);
          if (!buf) {
            buf = new Float32Array(grad.size);
            this.momentumBuffer.set(param, buf);
          }

          for (let i = 0; i < grad.size; i++) {
            buf[i] = this.momentum * buf[i]! + grad.data[i]! / (Math.sqrt(avg[i]!) + this.eps);
          }

          for (let i = 0; i < param.data.size; i++) {
            param.data.data[i] -= lr * buf[i]!;
          }
        } else {
          for (let i = 0; i < param.data.size; i++) {
            param.data.data[i] -= lr * grad.data[i]! / (Math.sqrt(avg[i]!) + this.eps);
          }
        }
      }
    }
  }

  stateDict(): Record<string, unknown> {
    const squareAvgList: Float32Array[] = [];
    const gradAvgList: Float32Array[] = [];
    const momentumBufferList: Float32Array[] = [];

    for (const group of this.paramGroups) {
      for (const param of group.params) {
        const sq = this.squareAvg.get(param);
        const ga = this.gradAvg.get(param);
        const mb = this.momentumBuffer.get(param);

        squareAvgList.push(sq ? new Float32Array(sq) : new Float32Array(0));
        gradAvgList.push(ga ? new Float32Array(ga) : new Float32Array(0));
        momentumBufferList.push(mb ? new Float32Array(mb) : new Float32Array(0));
      }
    }

    return {
      step: this._step,
      alpha: this.alpha,
      eps: this.eps,
      momentum: this.momentum,
      centered: this.centered,
      defaults: this.defaults,
      squareAvg: squareAvgList,
      gradAvg: gradAvgList,
      momentumBuffer: momentumBufferList,
    };
  }

  loadStateDict(stateDict: Record<string, unknown>): void {
    this._step = stateDict.step as number;
    this.alpha = stateDict.alpha as number;
    this.eps = stateDict.eps as number;
    this.momentum = stateDict.momentum as number;
    this.centered = stateDict.centered as boolean;
    this.defaults = stateDict.defaults as Record<string, number>;

    const squareAvgList = stateDict.squareAvg as Float32Array[];
    const gradAvgList = stateDict.gradAvg as Float32Array[];
    const momentumBufferList = stateDict.momentumBuffer as Float32Array[];

    let idx = 0;
    for (const group of this.paramGroups) {
      for (const param of group.params) {
        if (squareAvgList[idx]?.length) this.squareAvg.set(param, squareAvgList[idx]!);
        if (gradAvgList[idx]?.length) this.gradAvg.set(param, gradAvgList[idx]!);
        if (momentumBufferList[idx]?.length) this.momentumBuffer.set(param, momentumBufferList[idx]!);
        idx++;
      }
    }
  }
}

/**
 * Learning Rate Scheduler Base
 */
export abstract class LRScheduler {
  protected optimizer: Optimizer;
  protected baseLrs: number[];
  protected lastEpoch: number = 0;

  constructor(optimizer: Optimizer) {
    this.optimizer = optimizer;
    this.baseLrs = [(optimizer as unknown as { defaults: Record<string, number> }).defaults.lr!];
  }

  abstract getLr(): number[];

  step(epoch?: number): void {
    if (epoch !== undefined) {
      this.lastEpoch = epoch;
    } else {
      this.lastEpoch++;
    }

    const lrs = this.getLr();
    const optAny = this.optimizer as unknown as { paramGroups: ParamGroup[] };
    for (let i = 0; i < optAny.paramGroups.length; i++) {
      optAny.paramGroups[i]!.lr = lrs[Math.min(i, lrs.length - 1)]!;
    }
  }

  get currentLr(): number {
    return this.getLr()[0]!;
  }
}

/**
 * Step Learning Rate Scheduler
 *
 * Decays the learning rate by gamma every step_size epochs.
 */
export class StepLR extends LRScheduler {
  private stepSize: number;
  private gamma: number;

  constructor(optimizer: Optimizer, stepSize: number, gamma = 0.1) {
    super(optimizer);
    this.stepSize = stepSize;
    this.gamma = gamma;
  }

  getLr(): number[] {
    const decay = Math.pow(this.gamma, Math.floor(this.lastEpoch / this.stepSize));
    return this.baseLrs.map(lr => lr * decay);
  }
}

/**
 * Exponential Learning Rate Scheduler
 */
export class ExponentialLR extends LRScheduler {
  private gamma: number;

  constructor(optimizer: Optimizer, gamma: number) {
    super(optimizer);
    this.gamma = gamma;
  }

  getLr(): number[] {
    return this.baseLrs.map(lr => lr * Math.pow(this.gamma, this.lastEpoch));
  }
}

/**
 * Cosine Annealing Learning Rate Scheduler
 */
export class CosineAnnealingLR extends LRScheduler {
  private tMax: number;
  private etaMin: number;

  constructor(optimizer: Optimizer, tMax: number, etaMin = 0) {
    super(optimizer);
    this.tMax = tMax;
    this.etaMin = etaMin;
  }

  getLr(): number[] {
    return this.baseLrs.map(baseLr => {
      return this.etaMin + (baseLr - this.etaMin) * (1 + Math.cos(Math.PI * this.lastEpoch / this.tMax)) / 2;
    });
  }
}
