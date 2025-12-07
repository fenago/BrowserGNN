/**
 * BrowserGNN by Dr. Lee
 * WebGPU Compute Pipeline Infrastructure
 *
 * Manages shader compilation, pipeline creation, and compute dispatch
 * for high-performance graph neural network operations.
 */

import { GPUBufferPool, uploadToBuffer, readFromBuffer } from './buffer-pool';
import {
  SHADER_REGISTRY,
  ShaderName,
  SPMM_ROW_PARALLEL_SHADER,
  SCATTER_ADD_SHADER,
  SCATTER_MAX_SHADER,
  GATHER_SHADER,
  MATMUL_SHADER,
  RELU_SHADER,
  ADD_SHADER,
  ATTENTION_SCORE_SHADER,
} from './shaders';

export interface ComputePipelineConfig {
  maxBufferSize?: number; // MB
  enableProfiling?: boolean;
}

interface CompiledPipeline {
  pipeline: GPUComputePipeline;
  bindGroupLayout: GPUBindGroupLayout;
  entryPoint: string;
}

/**
 * WebGPU Compute Manager
 *
 * Central manager for all WebGPU compute operations.
 * Handles shader compilation, pipeline caching, and execution.
 */
export class WebGPUComputeManager {
  private device: GPUDevice;
  private bufferPool: GPUBufferPool;
  private pipelines: Map<string, CompiledPipeline>;
  private shaderModules: Map<string, GPUShaderModule>;
  private profiling: boolean;
  private querySet: GPUQuerySet | null = null;
  private queryBuffer: GPUBuffer | null = null;

  constructor(device: GPUDevice, config?: ComputePipelineConfig) {
    this.device = device;
    this.bufferPool = new GPUBufferPool(device, config?.maxBufferSize ?? 256);
    this.pipelines = new Map();
    this.shaderModules = new Map();
    this.profiling = config?.enableProfiling ?? false;

    // Initialize profiling resources if enabled
    if (this.profiling && 'createQuerySet' in device) {
      this.querySet = device.createQuerySet({
        type: 'timestamp',
        count: 2,
      });
      this.queryBuffer = device.createBuffer({
        size: 16,
        usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
      });
    }
  }

  /**
   * Get or compile a shader module
   */
  private getShaderModule(name: string, code: string): GPUShaderModule {
    if (this.shaderModules.has(name)) {
      return this.shaderModules.get(name)!;
    }

    const module = this.device.createShaderModule({
      label: `shader-${name}`,
      code,
    });

    this.shaderModules.set(name, module);
    return module;
  }

  /**
   * Get or create a compute pipeline
   */
  private getPipeline(
    name: string,
    shaderCode: string,
    entryPoint: string = 'main',
    bindGroupLayoutEntries: GPUBindGroupLayoutEntry[]
  ): CompiledPipeline {
    const key = `${name}_${entryPoint}`;

    if (this.pipelines.has(key)) {
      return this.pipelines.get(key)!;
    }

    const shaderModule = this.getShaderModule(name, shaderCode);

    const bindGroupLayout = this.device.createBindGroupLayout({
      label: `layout-${name}`,
      entries: bindGroupLayoutEntries,
    });

    const pipelineLayout = this.device.createPipelineLayout({
      label: `pipeline-layout-${name}`,
      bindGroupLayouts: [bindGroupLayout],
    });

    const pipeline = this.device.createComputePipeline({
      label: `pipeline-${name}`,
      layout: pipelineLayout,
      compute: {
        module: shaderModule,
        entryPoint,
      },
    });

    const compiled: CompiledPipeline = {
      pipeline,
      bindGroupLayout,
      entryPoint,
    };

    this.pipelines.set(key, compiled);
    return compiled;
  }

  /**
   * Sparse Matrix-Dense Matrix Multiplication (SpMM) using CSR format
   *
   * Computes: output = sparse @ dense
   * Where sparse is in CSR format [numRows, numCols] and dense is [numCols, numFeatures]
   *
   * This is the core operation for GNN message passing.
   */
  async spmm(
    rowPtr: Uint32Array,
    colIndices: Uint32Array,
    values: Float32Array,
    dense: Float32Array,
    numRows: number,
    numCols: number,
    numFeatures: number
  ): Promise<Float32Array> {
    const nnz = values.length;

    // Create bind group layout
    const layoutEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ];

    const { pipeline, bindGroupLayout } = this.getPipeline(
      'spmmRowParallel',
      SPMM_ROW_PARALLEL_SHADER,
      'main',
      layoutEntries
    );

    // Create buffers
    const uniformBuffer = this.bufferPool.createUniformBuffer(16, 'spmm-uniforms');
    const rowPtrBuffer = this.bufferPool.createStorageBuffer(rowPtr.byteLength, 'spmm-rowptr');
    const colIndicesBuffer = this.bufferPool.createStorageBuffer(colIndices.byteLength, 'spmm-colidx');
    const valuesBuffer = this.bufferPool.createStorageBuffer(values.byteLength, 'spmm-values');
    const denseBuffer = this.bufferPool.createStorageBuffer(dense.byteLength, 'spmm-dense');
    const outputBuffer = this.bufferPool.createStorageBuffer(numRows * numFeatures * 4, 'spmm-output');

    // Upload data
    const maxNnzPerRow = Math.max(...Array.from({ length: numRows }, (_, i) =>
      (rowPtr[i + 1] ?? 0) - (rowPtr[i] ?? 0)
    ));

    const uniforms = new Uint32Array([numRows, numFeatures, maxNnzPerRow, 0]);
    await uploadToBuffer(this.device, uniformBuffer, uniforms);
    await uploadToBuffer(this.device, rowPtrBuffer, rowPtr);
    await uploadToBuffer(this.device, colIndicesBuffer, colIndices);
    await uploadToBuffer(this.device, valuesBuffer, values);
    await uploadToBuffer(this.device, denseBuffer, dense);

    // Zero output buffer
    const zeros = new Float32Array(numRows * numFeatures);
    await uploadToBuffer(this.device, outputBuffer, zeros);

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      label: 'spmm-bindgroup',
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: rowPtrBuffer } },
        { binding: 2, resource: { buffer: colIndicesBuffer } },
        { binding: 3, resource: { buffer: valuesBuffer } },
        { binding: 4, resource: { buffer: denseBuffer } },
        { binding: 5, resource: { buffer: outputBuffer } },
      ],
    });

    // Dispatch compute
    const commandEncoder = this.device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);

    // Workgroup size is (64, 4), so we dispatch enough groups to cover all rows
    const workgroupsX = numRows;
    const workgroupsY = 1;
    computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
    computePass.end();

    this.device.queue.submit([commandEncoder.finish()]);

    // Read back results
    const result = await readFromBuffer(this.device, outputBuffer, numRows * numFeatures * 4);

    // Release buffers
    this.bufferPool.release(uniformBuffer);
    this.bufferPool.release(rowPtrBuffer);
    this.bufferPool.release(colIndicesBuffer);
    this.bufferPool.release(valuesBuffer);
    this.bufferPool.release(denseBuffer);
    this.bufferPool.release(outputBuffer);

    return result;
  }

  /**
   * Scatter-Add operation for message aggregation
   */
  async scatterAdd(
    src: Float32Array,
    indices: Uint32Array,
    numNodes: number,
    numFeatures: number
  ): Promise<Float32Array> {
    const numMessages = indices.length;

    const layoutEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ];

    const { pipeline, bindGroupLayout } = this.getPipeline(
      'scatterAdd',
      SCATTER_ADD_SHADER,
      'main',
      layoutEntries
    );

    // Create buffers
    const uniformBuffer = this.bufferPool.createUniformBuffer(16, 'scatter-uniforms');
    const indicesBuffer = this.bufferPool.createStorageBuffer(indices.byteLength, 'scatter-indices');
    const srcBuffer = this.bufferPool.createStorageBuffer(src.byteLength, 'scatter-src');
    const dstBuffer = this.bufferPool.createStorageBuffer(numNodes * numFeatures * 4, 'scatter-dst');

    // Upload data
    const uniforms = new Uint32Array([numMessages, numNodes, numFeatures, 0]);
    await uploadToBuffer(this.device, uniformBuffer, uniforms);
    await uploadToBuffer(this.device, indicesBuffer, indices);
    await uploadToBuffer(this.device, srcBuffer, src);

    // Zero output buffer
    const zeros = new Float32Array(numNodes * numFeatures);
    await uploadToBuffer(this.device, dstBuffer, zeros);

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: indicesBuffer } },
        { binding: 2, resource: { buffer: srcBuffer } },
        { binding: 3, resource: { buffer: dstBuffer } },
      ],
    });

    // Dispatch
    const commandEncoder = this.device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(Math.ceil(numMessages / 256));
    computePass.end();

    this.device.queue.submit([commandEncoder.finish()]);

    const result = await readFromBuffer(this.device, dstBuffer, numNodes * numFeatures * 4);

    // Release buffers
    this.bufferPool.release(uniformBuffer);
    this.bufferPool.release(indicesBuffer);
    this.bufferPool.release(srcBuffer);
    this.bufferPool.release(dstBuffer);

    return result;
  }

  /**
   * Scatter-Max operation for max-pooling aggregation
   */
  async scatterMax(
    src: Float32Array,
    indices: Uint32Array,
    numNodes: number,
    numFeatures: number
  ): Promise<Float32Array> {
    const numMessages = indices.length;

    const layoutEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ];

    const { pipeline, bindGroupLayout } = this.getPipeline(
      'scatterMax',
      SCATTER_MAX_SHADER,
      'main',
      layoutEntries
    );

    // Create buffers
    const uniformBuffer = this.bufferPool.createUniformBuffer(16, 'scatter-uniforms');
    const indicesBuffer = this.bufferPool.createStorageBuffer(indices.byteLength, 'scatter-indices');
    const srcBuffer = this.bufferPool.createStorageBuffer(src.byteLength, 'scatter-src');
    const dstBuffer = this.bufferPool.createStorageBuffer(numNodes * numFeatures * 4, 'scatter-dst');

    // Upload data
    const uniforms = new Uint32Array([numMessages, numNodes, numFeatures, 0]);
    await uploadToBuffer(this.device, uniformBuffer, uniforms);
    await uploadToBuffer(this.device, indicesBuffer, indices);
    await uploadToBuffer(this.device, srcBuffer, src);

    // Initialize output buffer to -infinity
    const initVals = new Float32Array(numNodes * numFeatures).fill(-Infinity);
    await uploadToBuffer(this.device, dstBuffer, initVals);

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: indicesBuffer } },
        { binding: 2, resource: { buffer: srcBuffer } },
        { binding: 3, resource: { buffer: dstBuffer } },
      ],
    });

    // Dispatch
    const commandEncoder = this.device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(Math.ceil(numMessages / 256));
    computePass.end();

    this.device.queue.submit([commandEncoder.finish()]);

    const result = await readFromBuffer(this.device, dstBuffer, numNodes * numFeatures * 4);

    // Replace -Infinity with 0
    for (let i = 0; i < result.length; i++) {
      if (result[i] === -Infinity) {
        result[i] = 0;
      }
    }

    // Release buffers
    this.bufferPool.release(uniformBuffer);
    this.bufferPool.release(indicesBuffer);
    this.bufferPool.release(srcBuffer);
    this.bufferPool.release(dstBuffer);

    return result;
  }

  /**
   * Gather operation for collecting node features
   */
  async gather(
    src: Float32Array,
    indices: Uint32Array,
    numFeatures: number
  ): Promise<Float32Array> {
    const numIndices = indices.length;

    const layoutEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ];

    const { pipeline, bindGroupLayout } = this.getPipeline(
      'gather',
      GATHER_SHADER,
      'main',
      layoutEntries
    );

    // Create buffers
    const uniformBuffer = this.bufferPool.createUniformBuffer(16, 'gather-uniforms');
    const indicesBuffer = this.bufferPool.createStorageBuffer(indices.byteLength, 'gather-indices');
    const srcBuffer = this.bufferPool.createStorageBuffer(src.byteLength, 'gather-src');
    const dstBuffer = this.bufferPool.createStorageBuffer(numIndices * numFeatures * 4, 'gather-dst');

    // Upload data
    const uniforms = new Uint32Array([numIndices, numFeatures, 0, 0]);
    await uploadToBuffer(this.device, uniformBuffer, uniforms);
    await uploadToBuffer(this.device, indicesBuffer, indices);
    await uploadToBuffer(this.device, srcBuffer, src);

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: indicesBuffer } },
        { binding: 2, resource: { buffer: srcBuffer } },
        { binding: 3, resource: { buffer: dstBuffer } },
      ],
    });

    // Dispatch
    const commandEncoder = this.device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(Math.ceil(numIndices / 256));
    computePass.end();

    this.device.queue.submit([commandEncoder.finish()]);

    const result = await readFromBuffer(this.device, dstBuffer, numIndices * numFeatures * 4);

    // Release buffers
    this.bufferPool.release(uniformBuffer);
    this.bufferPool.release(indicesBuffer);
    this.bufferPool.release(srcBuffer);
    this.bufferPool.release(dstBuffer);

    return result;
  }

  /**
   * Dense matrix multiplication: C = A @ B
   */
  async matmul(
    a: Float32Array,
    b: Float32Array,
    m: number,
    k: number,
    n: number
  ): Promise<Float32Array> {
    const layoutEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ];

    const { pipeline, bindGroupLayout } = this.getPipeline(
      'matmul',
      MATMUL_SHADER,
      'main',
      layoutEntries
    );

    // Create buffers
    const uniformBuffer = this.bufferPool.createUniformBuffer(16, 'matmul-uniforms');
    const aBuffer = this.bufferPool.createStorageBuffer(a.byteLength, 'matmul-a');
    const bBuffer = this.bufferPool.createStorageBuffer(b.byteLength, 'matmul-b');
    const cBuffer = this.bufferPool.createStorageBuffer(m * n * 4, 'matmul-c');

    // Upload data
    const uniforms = new Uint32Array([m, n, k, 0]);
    await uploadToBuffer(this.device, uniformBuffer, uniforms);
    await uploadToBuffer(this.device, aBuffer, a);
    await uploadToBuffer(this.device, bBuffer, b);

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: aBuffer } },
        { binding: 2, resource: { buffer: bBuffer } },
        { binding: 3, resource: { buffer: cBuffer } },
      ],
    });

    // Dispatch - tile size is 16x16
    const commandEncoder = this.device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(Math.ceil(n / 16), Math.ceil(m / 16));
    computePass.end();

    this.device.queue.submit([commandEncoder.finish()]);

    const result = await readFromBuffer(this.device, cBuffer, m * n * 4);

    // Release buffers
    this.bufferPool.release(uniformBuffer);
    this.bufferPool.release(aBuffer);
    this.bufferPool.release(bBuffer);
    this.bufferPool.release(cBuffer);

    return result;
  }

  /**
   * ReLU activation in-place
   */
  async relu(data: Float32Array): Promise<Float32Array> {
    const size = data.length;

    const layoutEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ];

    const { pipeline, bindGroupLayout } = this.getPipeline(
      'relu',
      RELU_SHADER,
      'main',
      layoutEntries
    );

    // Create buffers
    const uniformBuffer = this.bufferPool.createUniformBuffer(16, 'relu-uniforms');
    const dataBuffer = this.bufferPool.createStorageBuffer(data.byteLength, 'relu-data');

    // Upload data
    const uniforms = new Uint32Array([size, 0, 0, 0]);
    await uploadToBuffer(this.device, uniformBuffer, uniforms);
    await uploadToBuffer(this.device, dataBuffer, data);

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: dataBuffer } },
      ],
    });

    // Dispatch
    const commandEncoder = this.device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(Math.ceil(size / 256));
    computePass.end();

    this.device.queue.submit([commandEncoder.finish()]);

    const result = await readFromBuffer(this.device, dataBuffer, data.byteLength);

    // Release buffers
    this.bufferPool.release(uniformBuffer);
    this.bufferPool.release(dataBuffer);

    return result;
  }

  /**
   * Element-wise addition: result = a + b
   */
  async add(a: Float32Array, b: Float32Array): Promise<Float32Array> {
    if (a.length !== b.length) {
      throw new Error('Arrays must have same length');
    }

    const size = a.length;

    const layoutEntries: GPUBindGroupLayoutEntry[] = [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
      { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
    ];

    const { pipeline, bindGroupLayout } = this.getPipeline(
      'add',
      ADD_SHADER,
      'main',
      layoutEntries
    );

    // Create buffers
    const uniformBuffer = this.bufferPool.createUniformBuffer(16, 'add-uniforms');
    const aBuffer = this.bufferPool.createStorageBuffer(a.byteLength, 'add-a');
    const bBuffer = this.bufferPool.createStorageBuffer(b.byteLength, 'add-b');
    const resultBuffer = this.bufferPool.createStorageBuffer(a.byteLength, 'add-result');

    // Upload data
    const uniforms = new Uint32Array([size, 0, 0, 0]);
    await uploadToBuffer(this.device, uniformBuffer, uniforms);
    await uploadToBuffer(this.device, aBuffer, a);
    await uploadToBuffer(this.device, bBuffer, b);

    // Create bind group
    const bindGroup = this.device.createBindGroup({
      layout: bindGroupLayout,
      entries: [
        { binding: 0, resource: { buffer: uniformBuffer } },
        { binding: 1, resource: { buffer: aBuffer } },
        { binding: 2, resource: { buffer: bBuffer } },
        { binding: 3, resource: { buffer: resultBuffer } },
      ],
    });

    // Dispatch
    const commandEncoder = this.device.createCommandEncoder();
    const computePass = commandEncoder.beginComputePass();
    computePass.setPipeline(pipeline);
    computePass.setBindGroup(0, bindGroup);
    computePass.dispatchWorkgroups(Math.ceil(size / 256));
    computePass.end();

    this.device.queue.submit([commandEncoder.finish()]);

    const result = await readFromBuffer(this.device, resultBuffer, a.byteLength);

    // Release buffers
    this.bufferPool.release(uniformBuffer);
    this.bufferPool.release(aBuffer);
    this.bufferPool.release(bBuffer);
    this.bufferPool.release(resultBuffer);

    return result;
  }

  /**
   * Get buffer pool statistics
   */
  getStats(): ReturnType<GPUBufferPool['getStats']> {
    return this.bufferPool.getStats();
  }

  /**
   * Clear pipeline and shader caches
   */
  clearCache(): void {
    this.pipelines.clear();
    this.shaderModules.clear();
  }

  /**
   * Destroy the compute manager
   */
  destroy(): void {
    this.bufferPool.destroy();
    this.pipelines.clear();
    this.shaderModules.clear();
    this.querySet?.destroy();
    this.queryBuffer?.destroy();
  }
}
