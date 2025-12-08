/**
 * BrowserGNN by Dr. Lee
 * Model Serialization
 *
 * Save and load trained GNN models to/from JSON format.
 * Enables persistence of trained models in browser storage or file download.
 */

import { Tensor } from '../core/tensor';
import { Module, Parameter } from '../nn/module';

/**
 * Serialized model format
 */
export interface SerializedModel {
  version: string;
  name: string;
  timestamp: string;
  architecture: string;
  parameters: SerializedParameter[];
  metadata?: Record<string, unknown>;
}

/**
 * Serialized parameter format
 */
export interface SerializedParameter {
  name: string;
  shape: number[];
  data: number[];
  requiresGrad: boolean;
}

/**
 * State dict format (like PyTorch)
 */
export type StateDict = Map<string, { tensor: Tensor; requiresGrad: boolean }>;

/**
 * Serialize a module's parameters to a state dict
 */
export function getStateDict(module: Module): StateDict {
  const stateDict: StateDict = new Map();

  for (const [name, param] of module.namedParameters()) {
    stateDict.set(name, {
      tensor: param.tensor.clone(),
      requiresGrad: param.requiresGrad,
    });
  }

  return stateDict;
}

/**
 * Load parameters from a state dict into a module
 */
export function loadStateDict(
  module: Module,
  stateDict: StateDict,
  strict: boolean = true
): { missing: string[]; unexpected: string[] } {
  const missing: string[] = [];
  const unexpected: string[] = [];

  const moduleParams = module.namedParameters();
  const paramNames = new Set(moduleParams.keys());

  // Check for unexpected keys
  for (const name of stateDict.keys()) {
    if (!paramNames.has(name)) {
      unexpected.push(name);
    }
  }

  // Load matching parameters
  for (const [name, param] of moduleParams) {
    const savedParam = stateDict.get(name);

    if (!savedParam) {
      missing.push(name);
      continue;
    }

    // Check shape compatibility
    if (
      param.tensor.shape.length !== savedParam.tensor.shape.length ||
      !param.tensor.shape.every((s, i) => s === savedParam.tensor.shape[i])
    ) {
      if (strict) {
        throw new Error(
          `Shape mismatch for parameter "${name}": ` +
          `expected [${param.tensor.shape}], got [${savedParam.tensor.shape}]`
        );
      }
      missing.push(name);
      continue;
    }

    // Copy data
    param.tensor.data.set(savedParam.tensor.data);
  }

  if (strict && (missing.length > 0 || unexpected.length > 0)) {
    throw new Error(
      `State dict mismatch: missing=[${missing.join(', ')}], ` +
      `unexpected=[${unexpected.join(', ')}]`
    );
  }

  return { missing, unexpected };
}

/**
 * Serialize a module to JSON-compatible format
 */
export function serializeModel(
  module: Module,
  name: string = 'model',
  metadata?: Record<string, unknown>
): SerializedModel {
  const parameters: SerializedParameter[] = [];

  for (const [paramName, param] of module.namedParameters()) {
    parameters.push({
      name: paramName,
      shape: [...param.tensor.shape],
      data: Array.from(param.tensor.data),
      requiresGrad: param.requiresGrad,
    });
  }

  // Compute total number of parameters
  let numParameters = 0;
  for (const param of parameters) {
    numParameters += param.data.length;
  }

  return {
    version: '1.0.0',
    name,
    timestamp: new Date().toISOString(),
    architecture: module.constructor.name,
    parameters,
    metadata: {
      numParameters,
      ...metadata,
    },
  };
}

/**
 * Deserialize parameters from a serialized model format
 */
export function deserializeModel(serialized: SerializedModel): StateDict {
  const stateDict: StateDict = new Map();

  for (const param of serialized.parameters) {
    const tensor = new Tensor(
      new Float32Array(param.data),
      param.shape
    );

    stateDict.set(param.name, {
      tensor,
      requiresGrad: param.requiresGrad,
    });
  }

  return stateDict;
}

/**
 * Save model to JSON string
 */
export function saveModelToJSON(
  module: Module,
  name: string = 'model',
  metadata?: Record<string, unknown>
): string {
  const serialized = serializeModel(module, name, metadata);
  return JSON.stringify(serialized, null, 2);
}

/**
 * Load model from JSON string
 */
export function loadModelFromJSON(
  module: Module,
  json: string,
  strict: boolean = true
): { model: SerializedModel; missing: string[]; unexpected: string[] } {
  const serialized: SerializedModel = JSON.parse(json);
  const stateDict = deserializeModel(serialized);
  const { missing, unexpected } = loadStateDict(module, stateDict, strict);

  return { model: serialized, missing, unexpected };
}

/**
 * Download model as JSON file (browser only)
 */
export function downloadModel(
  module: Module,
  filename: string = 'model.json',
  metadata?: Record<string, unknown>
): void {
  if (typeof window === 'undefined' || typeof document === 'undefined') {
    throw new Error('downloadModel is only available in browser environment');
  }

  const json = saveModelToJSON(module, filename.replace('.json', ''), metadata);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/**
 * Save model to localStorage (browser only)
 */
export function saveModelToStorage(
  module: Module,
  key: string,
  metadata?: Record<string, unknown>
): void {
  if (typeof localStorage === 'undefined') {
    throw new Error('saveModelToStorage is only available in browser environment');
  }

  const json = saveModelToJSON(module, key, metadata);
  localStorage.setItem(`browsergnn_model_${key}`, json);
}

/**
 * Load model from localStorage (browser only)
 */
export function loadModelFromStorage(
  module: Module,
  key: string,
  strict: boolean = true
): { model: SerializedModel; missing: string[]; unexpected: string[] } | null {
  if (typeof localStorage === 'undefined') {
    throw new Error('loadModelFromStorage is only available in browser environment');
  }

  const json = localStorage.getItem(`browsergnn_model_${key}`);
  if (!json) {
    return null;
  }

  return loadModelFromJSON(module, json, strict);
}

/**
 * List all saved models in localStorage
 */
export function listSavedModels(): string[] {
  if (typeof localStorage === 'undefined') {
    return [];
  }

  const models: string[] = [];
  const prefix = 'browsergnn_model_';

  for (let i = 0; i < localStorage.length; i++) {
    const key = localStorage.key(i);
    if (key && key.startsWith(prefix)) {
      models.push(key.slice(prefix.length));
    }
  }

  return models;
}

/**
 * Delete a saved model from localStorage
 */
export function deleteSavedModel(key: string): boolean {
  if (typeof localStorage === 'undefined') {
    return false;
  }

  const fullKey = `browsergnn_model_${key}`;
  if (localStorage.getItem(fullKey)) {
    localStorage.removeItem(fullKey);
    return true;
  }

  return false;
}

/**
 * Calculate model size in bytes
 */
export function getModelSize(module: Module): {
  parameters: number;
  bytes: number;
  formatted: string;
} {
  let parameters = 0;

  for (const param of module.parameters()) {
    parameters += param.tensor.size;
  }

  // Each parameter is a 32-bit float = 4 bytes
  const bytes = parameters * 4;

  let formatted: string;
  if (bytes < 1024) {
    formatted = `${bytes} B`;
  } else if (bytes < 1024 * 1024) {
    formatted = `${(bytes / 1024).toFixed(2)} KB`;
  } else {
    formatted = `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  return { parameters, bytes, formatted };
}
