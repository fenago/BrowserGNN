/**
 * BrowserGNN by Dr. Lee
 * Serialization Tests
 */

import { describe, it, expect } from 'vitest';
import {
  serializeModel,
  deserializeModel,
  getStateDict,
  loadStateDict,
  saveModelToJSON,
  loadModelFromJSON,
  getModelSize,
} from '../src/utils/serialization';
import { Tensor } from '../src/core/tensor';
import { GCNConv } from '../src/layers/gcn';
import { Sequential, Linear, ReLU } from '../src/nn';

describe('Serialization', () => {
  describe('serializeModel / deserializeModel', () => {
    it('should serialize a GCN layer', () => {
      const layer = new GCNConv({
        inChannels: 16,
        outChannels: 8,
      });

      const serialized = serializeModel(layer, 'test-gcn');

      expect(serialized.name).toBe('test-gcn');
      expect(serialized.version).toBeDefined();
      expect(serialized.parameters.length).toBeGreaterThan(0);
      expect(serialized.metadata?.numParameters).toBe(layer.numParameters());
    });

    it('should serialize with custom metadata', () => {
      const layer = new GCNConv({
        inChannels: 16,
        outChannels: 8,
      });

      const serialized = serializeModel(layer, 'test-gcn', {
        description: 'Test model',
        author: 'Dr. Lee',
      });

      expect(serialized.metadata?.description).toBe('Test model');
      expect(serialized.metadata?.author).toBe('Dr. Lee');
    });

    it('should deserialize to state dict', () => {
      const layer = new GCNConv({
        inChannels: 16,
        outChannels: 8,
      });

      const serialized = serializeModel(layer, 'test-gcn');
      const stateDict = deserializeModel(serialized);

      expect(stateDict).toBeDefined();
      expect(stateDict.size).toBeGreaterThan(0);
    });

    it('should preserve parameter values through serialization', () => {
      const layer = new GCNConv({
        inChannels: 4,
        outChannels: 2,
      });

      // Modify a parameter value
      const params = layer.parameters();
      params[0].tensor.data[0] = 999.5;

      // Serialize and deserialize
      const serialized = serializeModel(layer, 'test');
      const stateDict = deserializeModel(serialized);

      // Check the value is preserved (use Map .values() iterator)
      const firstParam = stateDict.values().next().value;
      expect(firstParam.tensor.data[0]).toBe(999.5);
    });
  });

  describe('getStateDict / loadStateDict', () => {
    it('should get state dict from module', () => {
      const layer = new GCNConv({
        inChannels: 8,
        outChannels: 4,
      });

      const stateDict = getStateDict(layer);

      expect(stateDict.get('weight')).toBeDefined();
      expect(stateDict.get('weight')!.tensor.shape).toEqual([8, 4]);
    });

    it('should load state dict into module', () => {
      const layer1 = new GCNConv({
        inChannels: 4,
        outChannels: 2,
      });

      // Modify weights
      const stateDict = getStateDict(layer1);
      const weight = stateDict.get('weight')!;
      weight.tensor.data.fill(0.5);

      const layer2 = new GCNConv({
        inChannels: 4,
        outChannels: 2,
      });

      const result = loadStateDict(layer2, stateDict, false);

      expect(result.missing.length).toBe(0);
      expect(result.unexpected.length).toBe(0);

      // Check weights are loaded
      const newStateDict = getStateDict(layer2);
      expect(newStateDict.get('weight')!.tensor.data[0]).toBe(0.5);
    });

    it('should report missing keys', () => {
      const layer = new GCNConv({
        inChannels: 4,
        outChannels: 2,
        bias: true,
      });

      const partialStateDict = getStateDict(layer);
      partialStateDict.delete('bias');

      const layer2 = new GCNConv({
        inChannels: 4,
        outChannels: 2,
        bias: true,
      });

      const result = loadStateDict(layer2, partialStateDict, false);

      expect(result.missing).toContain('bias');
    });

    it('should report unexpected keys', () => {
      const layer = new GCNConv({
        inChannels: 4,
        outChannels: 2,
        bias: false,
      });

      const stateDict = getStateDict(layer);
      // Add an extra key
      stateDict.set('extra_key', {
        tensor: new Tensor(new Float32Array(4), [4]),
        requiresGrad: false,
      });

      const layer2 = new GCNConv({
        inChannels: 4,
        outChannels: 2,
        bias: false,
      });

      const result = loadStateDict(layer2, stateDict, false);

      expect(result.unexpected).toContain('extra_key');
    });
  });

  describe('saveModelToJSON / loadModelFromJSON', () => {
    it('should save and load model as JSON', () => {
      const layer = new GCNConv({
        inChannels: 4,
        outChannels: 2,
      });

      // Modify a weight
      const params = layer.parameters();
      params[0].tensor.data[0] = 123.456;

      const json = saveModelToJSON(layer, 'json-test');

      // Create new layer and load
      const layer2 = new GCNConv({
        inChannels: 4,
        outChannels: 2,
      });

      const result = loadModelFromJSON(layer2, json, false);

      expect(result.missing.length).toBe(0);
      expect(result.unexpected.length).toBe(0);
      expect(result.model.name).toBe('json-test');

      // Check value is restored
      const params2 = layer2.parameters();
      expect(params2[0].tensor.data[0]).toBeCloseTo(123.456, 3);
    });

    it('should handle Sequential models', () => {
      const model = new Sequential([
        new Linear({ inFeatures: 10, outFeatures: 8 }),
        new ReLU(),
        new Linear({ inFeatures: 8, outFeatures: 4 }),
      ]);

      const json = saveModelToJSON(model, 'sequential-test');
      const parsed = JSON.parse(json);

      expect(parsed.name).toBe('sequential-test');
      expect(parsed.parameters.length).toBe(4); // 2 weights + 2 biases
    });
  });

  describe('getModelSize', () => {
    it('should calculate model size correctly', () => {
      const layer = new GCNConv({
        inChannels: 16,
        outChannels: 8,
      });

      const size = getModelSize(layer);

      // 16*8 weight + 8 bias = 136 parameters
      expect(size.parameters).toBe(136);
      // 136 * 4 bytes = 544 bytes
      expect(size.bytes).toBe(544);
      expect(size.formatted).toBe('544 B');
    });

    it('should format larger sizes correctly', () => {
      const model = new Sequential([
        new Linear({ inFeatures: 256, outFeatures: 512 }),
        new Linear({ inFeatures: 512, outFeatures: 256 }),
      ]);

      const size = getModelSize(model);

      // (256*512 + 512) + (512*256 + 256) = 262912 params
      expect(size.parameters).toBe(262912);
      // 262912 * 4 = 1051648 bytes = ~1 MB
      expect(size.bytes).toBe(1051648);
      expect(size.formatted).toContain('MB');
    });
  });
});
