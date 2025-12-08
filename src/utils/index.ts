/**
 * BrowserGNN by Dr. Lee
 * Utilities
 *
 * Export all utility functions.
 */

export {
  // Serialization
  serializeModel,
  deserializeModel,
  getStateDict,
  loadStateDict,
  saveModelToJSON,
  loadModelFromJSON,
  downloadModel,
  saveModelToStorage,
  loadModelFromStorage,
  listSavedModels,
  deleteSavedModel,
  getModelSize,
  type SerializedModel,
  type SerializedParameter,
  type StateDict,
} from './serialization';
