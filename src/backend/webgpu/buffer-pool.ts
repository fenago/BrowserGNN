/**
 * BrowserGNN by Dr. Lee
 * GPU Buffer Pool for Memory Management
 *
 * Efficient buffer reuse to minimize GPU memory allocations
 * and improve performance for repeated operations.
 */

export interface BufferDescriptor {
  size: number;
  usage: GPUBufferUsageFlags;
  label?: string;
}

interface PooledBuffer {
  buffer: GPUBuffer;
  size: number;
  usage: GPUBufferUsageFlags;
  lastUsed: number;
  inUse: boolean;
}

/**
 * GPU Buffer Pool
 *
 * Manages a pool of GPU buffers for reuse, reducing allocation overhead.
 * Implements a simple least-recently-used (LRU) eviction policy.
 */
export class GPUBufferPool {
  private device: GPUDevice;
  private buffers: Map<string, PooledBuffer[]>;
  private maxPoolSize: number;
  private currentPoolSize: number;
  private hitCount: number;
  private missCount: number;

  constructor(device: GPUDevice, maxPoolSizeMB: number = 256) {
    this.device = device;
    this.buffers = new Map();
    this.maxPoolSize = maxPoolSizeMB * 1024 * 1024; // Convert to bytes
    this.currentPoolSize = 0;
    this.hitCount = 0;
    this.missCount = 0;
  }

  /**
   * Get a buffer from the pool or create a new one
   */
  acquire(descriptor: BufferDescriptor): GPUBuffer {
    const key = this.getKey(descriptor.usage);
    const pool = this.buffers.get(key) ?? [];

    // Find a suitable buffer (exact or larger size)
    for (let i = 0; i < pool.length; i++) {
      const pooled = pool[i]!;
      if (!pooled.inUse && pooled.size >= descriptor.size) {
        pooled.inUse = true;
        pooled.lastUsed = Date.now();
        this.hitCount++;
        return pooled.buffer;
      }
    }

    // No suitable buffer found, create new one
    this.missCount++;

    // Round up size to power of 2 for better reuse
    const allocSize = this.roundUpToPowerOf2(Math.max(descriptor.size, 256));

    // Check if we need to evict buffers
    if (this.currentPoolSize + allocSize > this.maxPoolSize) {
      this.evict(allocSize);
    }

    const buffer = this.device.createBuffer({
      size: allocSize,
      usage: descriptor.usage,
      label: descriptor.label ?? 'pooled-buffer',
      mappedAtCreation: false,
    });

    const pooled: PooledBuffer = {
      buffer,
      size: allocSize,
      usage: descriptor.usage,
      lastUsed: Date.now(),
      inUse: true,
    };

    if (!this.buffers.has(key)) {
      this.buffers.set(key, []);
    }
    this.buffers.get(key)!.push(pooled);
    this.currentPoolSize += allocSize;

    return buffer;
  }

  /**
   * Release a buffer back to the pool
   */
  release(buffer: GPUBuffer): void {
    for (const pool of this.buffers.values()) {
      for (const pooled of pool) {
        if (pooled.buffer === buffer) {
          pooled.inUse = false;
          pooled.lastUsed = Date.now();
          return;
        }
      }
    }
    // Buffer not in pool, destroy it
    buffer.destroy();
  }

  /**
   * Create a staging buffer for CPU-GPU data transfer
   */
  createStagingBuffer(size: number, forRead: boolean = false): GPUBuffer {
    const usage = forRead
      ? GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST
      : GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC;

    return this.acquire({
      size,
      usage,
      label: forRead ? 'staging-read' : 'staging-write',
    });
  }

  /**
   * Create a uniform buffer
   */
  createUniformBuffer(size: number, label?: string): GPUBuffer {
    // Uniform buffers need 256-byte alignment
    const alignedSize = Math.ceil(size / 256) * 256;

    return this.acquire({
      size: alignedSize,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: label ?? 'uniform',
    });
  }

  /**
   * Create a storage buffer
   */
  createStorageBuffer(size: number, label?: string): GPUBuffer {
    return this.acquire({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
      label: label ?? 'storage',
    });
  }

  /**
   * Evict least recently used buffers
   */
  private evict(requiredSize: number): void {
    // Collect all unused buffers with their last used time
    const candidates: { key: string; index: number; pooled: PooledBuffer }[] = [];

    for (const [key, pool] of this.buffers.entries()) {
      for (let i = 0; i < pool.length; i++) {
        const pooled = pool[i]!;
        if (!pooled.inUse) {
          candidates.push({ key, index: i, pooled });
        }
      }
    }

    // Sort by last used (oldest first)
    candidates.sort((a, b) => a.pooled.lastUsed - b.pooled.lastUsed);

    // Evict until we have enough space
    let freedSize = 0;
    const toRemove: { key: string; index: number }[] = [];

    for (const candidate of candidates) {
      if (freedSize >= requiredSize) break;

      candidate.pooled.buffer.destroy();
      freedSize += candidate.pooled.size;
      this.currentPoolSize -= candidate.pooled.size;
      toRemove.push({ key: candidate.key, index: candidate.index });
    }

    // Remove evicted buffers from pools (in reverse index order to maintain indices)
    toRemove.sort((a, b) => b.index - a.index);
    for (const { key, index } of toRemove) {
      const pool = this.buffers.get(key);
      if (pool) {
        pool.splice(index, 1);
      }
    }
  }

  /**
   * Get pool key for buffer usage
   */
  private getKey(usage: GPUBufferUsageFlags): string {
    return `usage_${usage}`;
  }

  /**
   * Round up to nearest power of 2
   */
  private roundUpToPowerOf2(n: number): number {
    if (n <= 0) return 256;
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
  }

  /**
   * Get pool statistics
   */
  getStats(): {
    poolSizeBytes: number;
    numBuffers: number;
    numInUse: number;
    hitRate: number;
  } {
    let numBuffers = 0;
    let numInUse = 0;

    for (const pool of this.buffers.values()) {
      numBuffers += pool.length;
      numInUse += pool.filter(p => p.inUse).length;
    }

    const totalRequests = this.hitCount + this.missCount;
    const hitRate = totalRequests > 0 ? this.hitCount / totalRequests : 0;

    return {
      poolSizeBytes: this.currentPoolSize,
      numBuffers,
      numInUse,
      hitRate,
    };
  }

  /**
   * Clear all buffers in the pool
   */
  clear(): void {
    for (const pool of this.buffers.values()) {
      for (const pooled of pool) {
        pooled.buffer.destroy();
      }
    }
    this.buffers.clear();
    this.currentPoolSize = 0;
    this.hitCount = 0;
    this.missCount = 0;
  }

  /**
   * Destroy the pool
   */
  destroy(): void {
    this.clear();
  }
}

/**
 * Upload data to a GPU buffer
 */
export async function uploadToBuffer(
  device: GPUDevice,
  buffer: GPUBuffer,
  data: ArrayBuffer | ArrayBufferView,
  offset: number = 0
): Promise<void> {
  device.queue.writeBuffer(
    buffer,
    offset,
    data instanceof ArrayBuffer ? data : data.buffer,
    data instanceof ArrayBuffer ? 0 : data.byteOffset,
    data instanceof ArrayBuffer ? data.byteLength : data.byteLength
  );
}

/**
 * Read data from a GPU buffer
 */
export async function readFromBuffer(
  device: GPUDevice,
  buffer: GPUBuffer,
  size: number
): Promise<Float32Array> {
  // Create staging buffer for readback
  const stagingBuffer = device.createBuffer({
    size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    label: 'staging-readback',
  });

  // Copy from storage to staging
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, size);
  device.queue.submit([commandEncoder.finish()]);

  // Wait for GPU work to complete
  await device.queue.onSubmittedWorkDone();

  // Map and read
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const copyArrayBuffer = stagingBuffer.getMappedRange();
  const result = new Float32Array(copyArrayBuffer.slice(0));
  stagingBuffer.unmap();
  stagingBuffer.destroy();

  return result;
}
