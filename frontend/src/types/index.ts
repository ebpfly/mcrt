// === Material Types ===

export interface MaterialInfo {
  material_id: string;
  name: string;
  shelf: string;
  book: string;
  page: string;
  references?: string;
  comments?: string;
  wavelength_range_um?: [number, number];
}

export interface OpticalConstants {
  wavelength_um: number[];
  n: number[];
  k: number[];
  material_info?: MaterialInfo;
}

// === Layer/Particle Types ===

export interface ParticleConfig {
  material_id: number;
  diameter_um: number;
  volume_fraction: number;
  std_dev: number;
}

export interface LayerConfig {
  matrix_id: number;
  thickness_um: number;
  particles: ParticleConfig[];
}

// === Simulation Types ===

export type SimulationStatus = 'idle' | 'running' | 'paused' | 'completed' | 'error';

export interface SimulationResults {
  wavelength_um: number[];
  reflectance: number[];
  absorptance: number[];
  transmittance: number[];
}

export interface SimulationSession {
  session_id: string;
  status: SimulationStatus;
  batches_completed: number;
  total_batches: number;
  photons_completed: number;
  photons_target: number;
  progress_percent: number;
  created_at: string;
  updated_at: string;
  error_message?: string;
  results?: SimulationResults;
}

export interface SimulationConfig {
  particle_materials: Record<string, OpticalConstants>;
  matrix_materials: Record<string, OpticalConstants>;
  layers: LayerConfig[];
  wavelength_start_um: number;
  wavelength_end_um: number;
  wavelength_interval_um: number;
  photons_target: number;
  n_batches: number;
}

// === SSE Event Types ===

export interface BatchUpdateEvent {
  batch_number: number;
  total_batches: number;
  photons_completed: number;
  photons_target: number;
  progress_percent: number;
  results: SimulationResults;
  timestamp: string;
}

export interface SimulationCompleteEvent {
  session_id: string;
  status: SimulationStatus;
  results: SimulationResults;
  timestamp: string;
}

export interface SimulationErrorEvent {
  session_id: string;
  error: string;
  timestamp: string;
}

// === UI Types ===

export interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  timestamp: number;
}

// === Health Check ===

export interface HealthStatus {
  status: string;
  version: string;
  fos_available: boolean;
  database_available: boolean;
  active_sessions: number;
}

// === Comparison Types ===

export interface SavedRun {
  id: string;
  name: string;
  config: SimulationConfig;
  results: SimulationResults;
  timestamp: string;
}
