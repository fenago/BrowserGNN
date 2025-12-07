/**
 * BrowserGNN by Dr. Lee
 * Neural Network Module Base
 *
 * Base class for all neural network modules in BrowserGNN.
 */

import { Tensor } from '../core/tensor';
import { GraphData } from '../core/graph';

/**
 * Parameter storage for modules
 */
export interface Parameter {
  name: string;
  tensor: Tensor;
  requiresGrad: boolean;
}

/**
 * Base class for all neural network modules
 */
export abstract class Module {
  protected _training: boolean = true;
  protected _parameters: Map<string, Parameter> = new Map();
  protected _modules: Map<string, Module> = new Map();

  /**
   * Forward pass - must be implemented by subclasses
   */
  abstract forward(input: Tensor | GraphData): Tensor | GraphData | Promise<Tensor | GraphData>;

  /**
   * Register a parameter
   */
  protected registerParameter(name: string, tensor: Tensor, requiresGrad: boolean = true): void {
    this._parameters.set(name, { name, tensor, requiresGrad });
  }

  /**
   * Register a submodule
   */
  protected registerModule(name: string, module: Module): void {
    this._modules.set(name, module);
  }

  /**
   * Get all parameters (including from submodules)
   */
  parameters(): Parameter[] {
    const params: Parameter[] = Array.from(this._parameters.values());

    for (const module of this._modules.values()) {
      params.push(...module.parameters());
    }

    return params;
  }

  /**
   * Get named parameters
   */
  namedParameters(prefix: string = ''): Map<string, Parameter> {
    const params = new Map<string, Parameter>();

    for (const [name, param] of this._parameters) {
      const fullName = prefix ? `${prefix}.${name}` : name;
      params.set(fullName, param);
    }

    for (const [name, module] of this._modules) {
      const subPrefix = prefix ? `${prefix}.${name}` : name;
      for (const [subName, param] of module.namedParameters(subPrefix)) {
        params.set(subName, param);
      }
    }

    return params;
  }

  /**
   * Set training mode
   */
  train(mode: boolean = true): this {
    this._training = mode;
    for (const module of this._modules.values()) {
      module.train(mode);
    }
    return this;
  }

  /**
   * Set evaluation mode
   */
  eval(): this {
    return this.train(false);
  }

  /**
   * Check if in training mode
   */
  get training(): boolean {
    return this._training;
  }

  /**
   * Count total parameters
   */
  numParameters(): number {
    return this.parameters().reduce((sum, p) => sum + p.tensor.size, 0);
  }

  /**
   * Get module summary
   */
  summary(): string {
    const lines: string[] = [];
    lines.push(`${this.constructor.name}`);
    lines.push(`  Parameters: ${this.numParameters().toLocaleString()}`);

    for (const [name, param] of this._parameters) {
      lines.push(`  ${name}: [${param.tensor.shape.join(', ')}]`);
    }

    for (const [name, module] of this._modules) {
      lines.push(`  ${name}: ${module.constructor.name}`);
    }

    return lines.join('\n');
  }
}

/**
 * Sequential container - runs modules in sequence
 */
export class Sequential extends Module {
  private modules: Module[];

  constructor(modules: Module[]) {
    super();
    this.modules = modules;
    modules.forEach((m, i) => this.registerModule(`${i}`, m));
  }

  async forward(input: Tensor | GraphData): Promise<Tensor | GraphData> {
    let output: Tensor | GraphData = input;

    for (const module of this.modules) {
      output = await module.forward(output);
    }

    return output;
  }

  /**
   * Add a module to the end
   */
  add(module: Module): void {
    const idx = this.modules.length;
    this.modules.push(module);
    this.registerModule(`${idx}`, module);
  }

  /**
   * Get number of modules
   */
  get length(): number {
    return this.modules.length;
  }
}
