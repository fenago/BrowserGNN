/**
 * BrowserGNN by Dr. Lee
 * Backend Management
 *
 * Handles compute backend selection and initialization.
 * Supports WebGPU (preferred) with WASM/CPU fallback.
 */

import { WebGPUComputeManager } from './webgpu';
import { WASMKernels } from './wasm';

export type BackendType = 'webgpu' | 'wasm' | 'cpu';

export interface BackendCapabilities {
  webgpu: boolean;
  wasm: boolean;
  sharedArrayBuffer: boolean;
  simd: boolean;
}

export interface BackendConfig {
  preferredBackend?: BackendType;
  enableProfiling?: boolean;
  memoryLimit?: number; // MB
}

/**
 * Backend manager singleton
 */
class BackendManager {
  private static instance: BackendManager;
  private initialized = false;
  private currentBackend: BackendType = 'cpu';
  private capabilities: BackendCapabilities | null = null;
  private gpuDevice: GPUDevice | null = null;
  private computeManager: WebGPUComputeManager | null = null;
  private wasmKernels: WASMKernels | null = null;
  private config: BackendConfig = {};

  private constructor() {}

  static getInstance(): BackendManager {
    if (!BackendManager.instance) {
      BackendManager.instance = new BackendManager();
    }
    return BackendManager.instance;
  }

  /**
   * Check available capabilities
   */
  async checkCapabilities(): Promise<BackendCapabilities> {
    if (this.capabilities) {
      return this.capabilities;
    }

    const caps: BackendCapabilities = {
      webgpu: false,
      wasm: false,
      sharedArrayBuffer: false,
      simd: false,
    };

    // Check WebGPU
    if (typeof navigator !== 'undefined' && 'gpu' in navigator) {
      try {
        const adapter = await navigator.gpu.requestAdapter();
        caps.webgpu = adapter !== null;
      } catch {
        caps.webgpu = false;
      }
    }

    // Check WASM
    caps.wasm = typeof WebAssembly !== 'undefined';

    // Check SharedArrayBuffer (needed for multi-threaded WASM)
    caps.sharedArrayBuffer = typeof SharedArrayBuffer !== 'undefined';

    // Check SIMD support
    if (caps.wasm) {
      try {
        // SIMD detection via feature detection
        caps.simd = WebAssembly.validate(
          new Uint8Array([
            0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0,
            253, 15, 253, 98, 11,
          ])
        );
      } catch {
        caps.simd = false;
      }
    }

    this.capabilities = caps;
    return caps;
  }

  /**
   * Initialize the backend
   */
  async initialize(config?: BackendConfig): Promise<BackendType> {
    if (this.initialized) {
      return this.currentBackend;
    }

    this.config = config ?? {};
    const caps = await this.checkCapabilities();

    // Determine best backend
    if (this.config.preferredBackend) {
      if (this.config.preferredBackend === 'webgpu' && caps.webgpu) {
        await this.initWebGPU();
        this.currentBackend = 'webgpu';
      } else if (this.config.preferredBackend === 'wasm' && caps.wasm) {
        this.initWASM();
        this.currentBackend = 'wasm';
      } else {
        this.currentBackend = 'cpu';
      }
    } else {
      // Auto-select best available
      if (caps.webgpu) {
        await this.initWebGPU();
        this.currentBackend = 'webgpu';
      } else if (caps.wasm) {
        this.initWASM();
        this.currentBackend = 'wasm';
      } else {
        this.currentBackend = 'cpu';
      }
    }

    this.initialized = true;
    console.log(`BrowserGNN: Initialized with ${this.currentBackend} backend`);

    return this.currentBackend;
  }

  /**
   * Initialize WebGPU
   */
  private async initWebGPU(): Promise<void> {
    if (!('gpu' in navigator)) {
      throw new Error('WebGPU not supported');
    }

    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });

    if (!adapter) {
      throw new Error('Failed to get WebGPU adapter');
    }

    // Request device with limits
    const requiredLimits: GPUDeviceDescriptor['requiredLimits'] = {};

    if (this.config.memoryLimit) {
      // Note: maxBufferSize is in bytes
      requiredLimits.maxBufferSize = this.config.memoryLimit * 1024 * 1024;
    }

    this.gpuDevice = await adapter.requestDevice({
      requiredLimits,
    });

    // Create compute manager
    this.computeManager = new WebGPUComputeManager(this.gpuDevice, {
      maxBufferSize: this.config.memoryLimit,
      enableProfiling: this.config.enableProfiling,
    });

    // Handle device loss
    this.gpuDevice.lost.then(info => {
      console.error(`WebGPU device lost: ${info.message}`);
      this.computeManager?.destroy();
      this.computeManager = null;
      this.gpuDevice = null;
      this.initialized = false;
    });
  }

  /**
   * Initialize WASM backend
   */
  private initWASM(): void {
    this.wasmKernels = WASMKernels.getInstance();
  }

  /**
   * Get current backend type
   */
  getBackend(): BackendType {
    return this.currentBackend;
  }

  /**
   * Get WebGPU device (if available)
   */
  getGPUDevice(): GPUDevice | null {
    return this.gpuDevice;
  }

  /**
   * Get WebGPU compute manager (if available)
   */
  getComputeManager(): WebGPUComputeManager | null {
    return this.computeManager;
  }

  /**
   * Get WASM kernels (if available)
   */
  getWASMKernels(): WASMKernels | null {
    return this.wasmKernels;
  }

  /**
   * Check if initialized
   */
  isInitialized(): boolean {
    return this.initialized;
  }

  /**
   * Get backend info string
   */
  getInfo(): string {
    const caps = this.capabilities;
    if (!caps) {
      return 'Backend not initialized';
    }

    return `BrowserGNN Backend Info:
  Current: ${this.currentBackend}
  WebGPU: ${caps.webgpu ? 'available' : 'unavailable'}
  WASM: ${caps.wasm ? 'available' : 'unavailable'}
  SharedArrayBuffer: ${caps.sharedArrayBuffer ? 'available' : 'unavailable'}
  SIMD: ${caps.simd ? 'available' : 'unavailable'}`;
  }

  /**
   * Reset backend (for testing)
   */
  reset(): void {
    this.initialized = false;
    this.computeManager?.destroy();
    this.computeManager = null;
    this.wasmKernels = null;
    this.gpuDevice?.destroy();
    this.gpuDevice = null;
    this.capabilities = null;
    this.currentBackend = 'cpu';
  }
}

// Export singleton accessor
export const backend = BackendManager.getInstance();

/**
 * Initialize BrowserGNN backend
 * Call this before using any GNN operations
 */
export async function initBrowserGNN(config?: BackendConfig): Promise<BackendType> {
  return backend.initialize(config);
}

/**
 * Get current backend type
 */
export function getBackend(): BackendType {
  return backend.getBackend();
}

/**
 * Check backend capabilities
 */
export async function checkCapabilities(): Promise<BackendCapabilities> {
  return backend.checkCapabilities();
}

/**
 * Get backend info
 */
export function getBackendInfo(): string {
  return backend.getInfo();
}

/**
 * Get WebGPU compute manager (if available)
 */
export function getComputeManager(): WebGPUComputeManager | null {
  return backend.getComputeManager();
}

/**
 * Get GPU device (if available)
 */
export function getGPUDevice(): GPUDevice | null {
  return backend.getGPUDevice();
}

/**
 * Get WASM kernels (if available)
 */
export function getWASMKernels(): WASMKernels | null {
  return backend.getWASMKernels();
}

// Re-export WebGPU module types
export { WebGPUComputeManager } from './webgpu';
export type { ComputePipelineConfig, BufferDescriptor, ShaderName } from './webgpu';

// Re-export WASM module types
export { WASMKernels } from './wasm';
export {
  wasmMatmul,
  wasmScatterAdd,
  wasmScatterMean,
  wasmScatterMax,
  wasmGather,
  wasmRelu,
  wasmAdd,
  wasmSpmv,
  wasmSpmm,
} from './wasm';
