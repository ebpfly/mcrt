import axios, { AxiosInstance } from 'axios';
import {
  HealthStatus,
  MaterialInfo,
  OpticalConstants,
  SimulationConfig,
  SimulationSession,
} from '../types';

const API_BASE = '/api/v1';

class APIService {
  private client: AxiosInstance;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE,
      headers: {
        'Content-Type': 'application/json',
      },
    });
  }

  // === Health ===

  async getHealth(): Promise<HealthStatus> {
    const response = await this.client.get<HealthStatus>('/health');
    return response.data;
  }

  // === Materials ===

  async listMaterials(params?: {
    shelf?: string;
    search?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ materials: MaterialInfo[]; total: number }> {
    const response = await this.client.get('/materials', { params });
    return response.data;
  }

  async listShelves(): Promise<string[]> {
    const response = await this.client.get<{ shelves: string[] }>('/materials/shelves');
    return response.data.shelves;
  }

  async getMaterial(
    materialId: string,
    wavelengthRange?: { min: number; max: number }
  ): Promise<OpticalConstants> {
    const params = wavelengthRange
      ? {
          wavelength_min_um: wavelengthRange.min,
          wavelength_max_um: wavelengthRange.max,
        }
      : undefined;
    const response = await this.client.get<OpticalConstants>(
      `/materials/${encodeURIComponent(materialId)}`,
      { params }
    );
    return response.data;
  }

  async createCustomMaterial(material: {
    name: string;
    wavelength_um: number[];
    n: number[];
    k: number[];
    description?: string;
  }): Promise<void> {
    await this.client.post('/materials/custom', material);
  }

  async getCustomMaterial(name: string): Promise<OpticalConstants> {
    const response = await this.client.get<OpticalConstants>(
      `/materials/custom/${encodeURIComponent(name)}`
    );
    return response.data;
  }

  // === Simulation ===

  async startSimulation(config: SimulationConfig): Promise<SimulationSession> {
    const response = await this.client.post<SimulationSession>('/simulation/start', config);
    return response.data;
  }

  async getSimulation(sessionId: string): Promise<SimulationSession> {
    const response = await this.client.get<SimulationSession>(`/simulation/${sessionId}`);
    return response.data;
  }

  async continueSimulation(
    sessionId: string,
    nBatches?: number
  ): Promise<SimulationSession> {
    const params = nBatches ? { n_batches: nBatches } : undefined;
    const response = await this.client.post<SimulationSession>(
      `/simulation/${sessionId}/continue`,
      null,
      { params }
    );
    return response.data;
  }

  async stopSimulation(sessionId: string): Promise<void> {
    await this.client.post(`/simulation/${sessionId}/stop`);
  }

  async deleteSimulation(sessionId: string): Promise<void> {
    await this.client.delete(`/simulation/${sessionId}`);
  }

  async listSimulations(): Promise<{ sessions: SimulationSession[]; total: number }> {
    const response = await this.client.get('/simulations');
    return response.data;
  }

  // === State Management ===

  async exportState(sessionId: string, format: 'json' | 'compact' = 'json'): Promise<Blob> {
    const response = await this.client.get(`/simulation/${sessionId}/state`, {
      params: { format },
      responseType: 'blob',
    });
    return response.data;
  }

  async restoreState(file: File): Promise<{ session_id: string; status: string; message: string }> {
    const formData = new FormData();
    formData.append('file', file);
    const response = await this.client.post('/simulation/restore', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  }

  async exportResultsCSV(sessionId: string): Promise<Blob> {
    const response = await this.client.get(`/simulation/${sessionId}/results/csv`, {
      responseType: 'blob',
    });
    return response.data;
  }

  // === SSE Stream URL ===

  getStreamURL(sessionId: string): string {
    return `${API_BASE}/simulation/${sessionId}/stream`;
  }

  // === Reference Data ===

  async listReferenceData(): Promise<{
    references: Array<{
      id: string;
      name: string;
      source: string;
      description: string;
      measurement_type: string;
      particle_size: string;
      wavelength_range_um: [number, number];
    }>;
    total: number;
  }> {
    const response = await this.client.get('/reference');
    return response.data;
  }

  async getReferenceData(referenceId: string): Promise<{
    id: string;
    name: string;
    source: string;
    description: string;
    measurement_type: string;
    particle_size: string;
    wavelength_range_um: [number, number];
    data: {
      wavelength_um: number[];
      reflectance: number[];
    };
  }> {
    const response = await this.client.get(`/reference/${referenceId}`);
    return response.data;
  }
}

export const api = new APIService();
export default api;
