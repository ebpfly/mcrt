import { configureStore, createSlice, PayloadAction } from '@reduxjs/toolkit';
import { TypedUseSelectorHook, useDispatch, useSelector } from 'react-redux';
import {
  MaterialInfo,
  OpticalConstants,
  LayerConfig,
  ParticleConfig,
  SimulationSession,
  SimulationResults,
  SimulationStatus,
  HealthStatus,
  Notification,
  SavedRun,
} from '../types';

// === UI Slice ===

interface UIState {
  drawerOpen: boolean;
  activeTab: number;
  notifications: Notification[];
  loading: boolean;
}

const initialUIState: UIState = {
  drawerOpen: false,
  activeTab: 0,
  notifications: [],
  loading: false,
};

const uiSlice = createSlice({
  name: 'ui',
  initialState: initialUIState,
  reducers: {
    setDrawerOpen: (state, action: PayloadAction<boolean>) => {
      state.drawerOpen = action.payload;
    },
    setActiveTab: (state, action: PayloadAction<number>) => {
      state.activeTab = action.payload;
    },
    addNotification: (state, action: PayloadAction<Omit<Notification, 'id' | 'timestamp'>>) => {
      state.notifications.push({
        ...action.payload,
        id: Math.random().toString(36).substr(2, 9),
        timestamp: Date.now(),
      });
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    setLoading: (state, action: PayloadAction<boolean>) => {
      state.loading = action.payload;
    },
  },
});

// === Health Slice ===

interface HealthState {
  status: HealthStatus | null;
  availableMaterials: MaterialInfo[];
  availableShelves: string[];
  lastCheck: number | null;
}

const initialHealthState: HealthState = {
  status: null,
  availableMaterials: [],
  availableShelves: [],
  lastCheck: null,
};

const healthSlice = createSlice({
  name: 'health',
  initialState: initialHealthState,
  reducers: {
    setHealthStatus: (state, action: PayloadAction<HealthStatus>) => {
      state.status = action.payload;
      state.lastCheck = Date.now();
    },
    setAvailableMaterials: (state, action: PayloadAction<MaterialInfo[]>) => {
      state.availableMaterials = action.payload;
    },
    setAvailableShelves: (state, action: PayloadAction<string[]>) => {
      state.availableShelves = action.payload;
    },
  },
});

// === Medium Slice ===

interface MediumState {
  layers: LayerConfig[];
  particleMaterials: Record<number, OpticalConstants>;
  matrixMaterials: Record<number, OpticalConstants>;
  selectedMaterialIds: {
    particle: Record<number, string>;
    matrix: Record<number, string>;
  };
}

const initialMediumState: MediumState = {
  layers: [
    {
      matrix_id: 1,
      thickness_um: 2000,  // 2mm optically thick layer
      particles: [
        {
          material_id: 1,
          diameter_um: 3,  // 3µm - validated against Adding-Doubling exact solution
          volume_fraction: 40,  // 40% - validated against Adding-Doubling
          std_dev: 0.0,  // Monodisperse for validation comparison
        },
      ],
    },
  ],
  particleMaterials: {},
  matrixMaterials: {},
  selectedMaterialIds: {
    particle: {},
    matrix: {},
  },
};

const mediumSlice = createSlice({
  name: 'medium',
  initialState: initialMediumState,
  reducers: {
    setLayers: (state, action: PayloadAction<LayerConfig[]>) => {
      state.layers = action.payload;
    },
    addLayer: (state, action: PayloadAction<LayerConfig>) => {
      state.layers.push(action.payload);
    },
    updateLayer: (state, action: PayloadAction<{ index: number; layer: LayerConfig }>) => {
      state.layers[action.payload.index] = action.payload.layer;
    },
    removeLayer: (state, action: PayloadAction<number>) => {
      state.layers.splice(action.payload, 1);
    },
    addParticleToLayer: (state, action: PayloadAction<{ layerIndex: number; particle: ParticleConfig }>) => {
      state.layers[action.payload.layerIndex].particles.push(action.payload.particle);
    },
    updateParticle: (
      state,
      action: PayloadAction<{ layerIndex: number; particleIndex: number; particle: ParticleConfig }>
    ) => {
      state.layers[action.payload.layerIndex].particles[action.payload.particleIndex] = action.payload.particle;
    },
    removeParticle: (state, action: PayloadAction<{ layerIndex: number; particleIndex: number }>) => {
      state.layers[action.payload.layerIndex].particles.splice(action.payload.particleIndex, 1);
    },
    setParticleMaterial: (state, action: PayloadAction<{ id: number; material: OpticalConstants }>) => {
      state.particleMaterials[action.payload.id] = action.payload.material;
    },
    setMatrixMaterial: (state, action: PayloadAction<{ id: number; material: OpticalConstants }>) => {
      state.matrixMaterials[action.payload.id] = action.payload.material;
    },
    setSelectedMaterialId: (
      state,
      action: PayloadAction<{ type: 'particle' | 'matrix'; id: number; materialId: string }>
    ) => {
      state.selectedMaterialIds[action.payload.type][action.payload.id] = action.payload.materialId;
    },
  },
});

// === Simulation Slice ===

interface SimulationState {
  wavelengthStart: number;
  wavelengthEnd: number;
  wavelengthInterval: number;
  photonsTarget: number;
  nBatches: number;
  currentSession: SimulationSession | null;
  status: SimulationStatus;
  results: SimulationResults | null;
  progressPercent: number;
}

const initialSimulationState: SimulationState = {
  wavelengthStart: 7.0,
  wavelengthEnd: 14.0,
  wavelengthInterval: 0.1,
  photonsTarget: 10000,
  nBatches: 1,
  currentSession: null,
  status: 'idle',
  results: null,
  progressPercent: 0,
};

const simulationSlice = createSlice({
  name: 'simulation',
  initialState: initialSimulationState,
  reducers: {
    setWavelengthRange: (state, action: PayloadAction<{ start: number; end: number; interval: number }>) => {
      state.wavelengthStart = action.payload.start;
      state.wavelengthEnd = action.payload.end;
      state.wavelengthInterval = action.payload.interval;
    },
    setPhotonsTarget: (state, action: PayloadAction<number>) => {
      state.photonsTarget = action.payload;
    },
    setNBatches: (state, action: PayloadAction<number>) => {
      state.nBatches = action.payload;
    },
    setCurrentSession: (state, action: PayloadAction<SimulationSession | null>) => {
      state.currentSession = action.payload;
      if (action.payload) {
        state.status = action.payload.status;
        state.progressPercent = action.payload.progress_percent;
        if (action.payload.results) {
          state.results = action.payload.results;
        }
      }
    },
    setStatus: (state, action: PayloadAction<SimulationStatus>) => {
      state.status = action.payload;
    },
    setResults: (state, action: PayloadAction<SimulationResults>) => {
      state.results = action.payload;
    },
    setProgress: (state, action: PayloadAction<number>) => {
      state.progressPercent = action.payload;
    },
    updateFromBatch: (
      state,
      action: PayloadAction<{ progress: number; results: SimulationResults }>
    ) => {
      state.progressPercent = action.payload.progress;
      state.results = action.payload.results;
    },
    resetSimulation: (state) => {
      state.currentSession = null;
      state.status = 'idle';
      state.results = null;
      state.progressPercent = 0;
    },
  },
});

// === Comparison Slice ===

interface ComparisonState {
  savedRuns: SavedRun[];
  selectedRunIds: string[];
}

const initialComparisonState: ComparisonState = {
  savedRuns: [],
  selectedRunIds: [],
};

const comparisonSlice = createSlice({
  name: 'comparison',
  initialState: initialComparisonState,
  reducers: {
    addSavedRun: (state, action: PayloadAction<SavedRun>) => {
      state.savedRuns.push(action.payload);
    },
    removeSavedRun: (state, action: PayloadAction<string>) => {
      state.savedRuns = state.savedRuns.filter(r => r.id !== action.payload);
      state.selectedRunIds = state.selectedRunIds.filter(id => id !== action.payload);
    },
    toggleRunSelection: (state, action: PayloadAction<string>) => {
      const index = state.selectedRunIds.indexOf(action.payload);
      if (index === -1) {
        state.selectedRunIds.push(action.payload);
      } else {
        state.selectedRunIds.splice(index, 1);
      }
    },
    clearSelection: (state) => {
      state.selectedRunIds = [];
    },
  },
});

// === Store Configuration ===

export const store = configureStore({
  reducer: {
    ui: uiSlice.reducer,
    health: healthSlice.reducer,
    medium: mediumSlice.reducer,
    simulation: simulationSlice.reducer,
    comparison: comparisonSlice.reducer,
  },
});

// === Exports ===

export const uiActions = uiSlice.actions;
export const healthActions = healthSlice.actions;
export const mediumActions = mediumSlice.actions;
export const simulationActions = simulationSlice.actions;
export const comparisonActions = comparisonSlice.actions;

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;

export const useAppDispatch = () => useDispatch<AppDispatch>();
export const useAppSelector: TypedUseSelectorHook<RootState> = useSelector;
