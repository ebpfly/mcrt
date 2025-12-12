import { useState, useCallback } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Alert,
  CircularProgress,
  Divider,
  Card,
  CardContent,
  Autocomplete,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Calculate as CalculateIcon,
} from '@mui/icons-material';
import api from '../services/api';
import { useAppSelector } from '../store';
import { MaterialInfo, ThinFilmResults, OpticalConstants } from '../types';

interface LayerState {
  id: string;
  thickness_nm: number;
  material: MaterialInfo | null;
  opticalConstants: OpticalConstants | null;
}

export default function ThinFilmPage() {
  const materials = useAppSelector((state) => state.health.availableMaterials);

  // Layer configuration
  const [layers, setLayers] = useState<LayerState[]>([
    { id: '1', thickness_nm: 100, material: null, opticalConstants: null },
  ]);

  // Substrate configuration
  const [substrateMaterial, setSubstrateMaterial] = useState<MaterialInfo | null>(null);
  const [substrateOC, setSubstrateOC] = useState<OpticalConstants | null>(null);

  // Wavelength configuration
  const [wavelengthStart, setWavelengthStart] = useState(0.4);
  const [wavelengthEnd, setWavelengthEnd] = useState(0.8);
  const wavelengthPoints = 100;

  // Angle and polarization
  const [angleDeg, setAngleDeg] = useState(0);
  const [polarization, setPolarization] = useState<'s' | 'p' | 'unpolarized'>('unpolarized');

  // Results and status
  const [results, setResults] = useState<ThinFilmResults | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Generate wavelength array
  const generateWavelengths = useCallback(() => {
    const wavelengths: number[] = [];
    const step = (wavelengthEnd - wavelengthStart) / (wavelengthPoints - 1);
    for (let i = 0; i < wavelengthPoints; i++) {
      wavelengths.push(wavelengthStart + i * step);
    }
    return wavelengths;
  }, [wavelengthStart, wavelengthEnd, wavelengthPoints]);

  // Add layer
  const addLayer = () => {
    setLayers([
      ...layers,
      { id: Date.now().toString(), thickness_nm: 100, material: null, opticalConstants: null },
    ]);
  };

  // Remove layer
  const removeLayer = (id: string) => {
    if (layers.length > 1) {
      setLayers(layers.filter((l) => l.id !== id));
    }
  };

  // Update layer thickness
  const updateLayerThickness = (id: string, thickness: number) => {
    setLayers(layers.map((l) => (l.id === id ? { ...l, thickness_nm: thickness } : l)));
  };

  // Update layer material
  const updateLayerMaterial = async (id: string, material: MaterialInfo | null) => {
    if (!material) {
      setLayers(layers.map((l) => (l.id === id ? { ...l, material: null, opticalConstants: null } : l)));
      return;
    }

    try {
      const oc = await api.getMaterial(material.material_id);
      setLayers(layers.map((l) => (l.id === id ? { ...l, material, opticalConstants: oc } : l)));
    } catch {
      setError(`Failed to load material: ${material.name}`);
    }
  };

  // Update substrate material
  const updateSubstrateMaterial = async (material: MaterialInfo | null) => {
    if (!material) {
      setSubstrateMaterial(null);
      setSubstrateOC(null);
      return;
    }

    try {
      const oc = await api.getMaterial(material.material_id);
      setSubstrateMaterial(material);
      setSubstrateOC(oc);
    } catch {
      setError(`Failed to load material: ${material.name}`);
    }
  };

  // Interpolate optical constants to wavelength grid
  const interpolateOC = (oc: OpticalConstants, wavelengths: number[]): { n: number[]; k: number[] } => {
    const n: number[] = [];
    const k: number[] = [];

    for (const wl of wavelengths) {
      let i = 0;
      while (i < oc.wavelength_um.length - 1 && oc.wavelength_um[i + 1] < wl) {
        i++;
      }

      if (i >= oc.wavelength_um.length - 1) {
        n.push(oc.n[oc.n.length - 1]);
        k.push(oc.k[oc.k.length - 1]);
      } else if (wl <= oc.wavelength_um[0]) {
        n.push(oc.n[0]);
        k.push(oc.k[0]);
      } else {
        const t = (wl - oc.wavelength_um[i]) / (oc.wavelength_um[i + 1] - oc.wavelength_um[i]);
        n.push(oc.n[i] + t * (oc.n[i + 1] - oc.n[i]));
        k.push(oc.k[i] + t * (oc.k[i + 1] - oc.k[i]));
      }
    }

    return { n, k };
  };

  // Calculate
  const handleCalculate = async () => {
    setError(null);

    if (!substrateOC) {
      setError('Please select a substrate material');
      return;
    }

    for (const layer of layers) {
      if (!layer.opticalConstants) {
        setError('Please select materials for all layers');
        return;
      }
    }

    setLoading(true);

    try {
      const wavelengths = generateWavelengths();

      const substrateInterp = interpolateOC(substrateOC, wavelengths);

      const layerConfigs = layers.map((layer) => {
        const interp = interpolateOC(layer.opticalConstants!, wavelengths);
        return {
          thickness_nm: layer.thickness_nm,
          n: interp.n,
          k: interp.k,
        };
      });

      const result = await api.calculateThinFilm({
        wavelength_um: wavelengths,
        layers: layerConfigs,
        substrate_n: substrateInterp.n,
        substrate_k: substrateInterp.k,
        angle_deg: angleDeg,
        polarization,
      });

      setResults(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Calculation failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Thin Film Calculator
      </Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 3 }}>
        Calculate reflectance, transmittance, and absorptance of multilayer thin films using the Transfer Matrix Method.
      </Typography>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Configuration Panel */}
        <Grid item xs={12} md={5}>
          <Paper sx={{ p: 2 }}>
            {/* Substrate */}
            <Typography variant="h6" gutterBottom>
              Substrate
            </Typography>
            <Autocomplete
              size="small"
              options={materials}
              getOptionLabel={(option) => option.name}
              value={substrateMaterial}
              onChange={(_, value) => updateSubstrateMaterial(value)}
              renderInput={(params) => <TextField {...params} label="Substrate Material" />}
              renderOption={(props, option) => (
                <li {...props} key={option.material_id}>
                  <Box>
                    <Typography variant="body2">{option.name}</Typography>
                    <Typography variant="caption" color="text.secondary">
                      {option.shelf}/{option.book}
                    </Typography>
                  </Box>
                </li>
              )}
            />

            <Divider sx={{ my: 2 }} />

            {/* Thin Film Layers */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
              <Typography variant="h6">Thin Film Layers</Typography>
              <Button startIcon={<AddIcon />} onClick={addLayer} size="small">
                Add Layer
              </Button>
            </Box>

            {layers.map((layer, index) => (
              <Card key={layer.id} variant="outlined" sx={{ mb: 2 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                    <Typography variant="subtitle2">Layer {index + 1}</Typography>
                    {layers.length > 1 && (
                      <IconButton size="small" onClick={() => removeLayer(layer.id)}>
                        <DeleteIcon fontSize="small" />
                      </IconButton>
                    )}
                  </Box>
                  <Autocomplete
                    size="small"
                    options={materials}
                    getOptionLabel={(option) => option.name}
                    value={layer.material}
                    onChange={(_, value) => updateLayerMaterial(layer.id, value)}
                    renderInput={(params) => <TextField {...params} label="Material" />}
                    renderOption={(props, option) => (
                      <li {...props} key={option.material_id}>
                        <Box>
                          <Typography variant="body2">{option.name}</Typography>
                          <Typography variant="caption" color="text.secondary">
                            {option.shelf}/{option.book}
                          </Typography>
                        </Box>
                      </li>
                    )}
                  />
                  <TextField
                    fullWidth
                    size="small"
                    label="Thickness (nm)"
                    type="number"
                    value={layer.thickness_nm}
                    onChange={(e) => updateLayerThickness(layer.id, parseFloat(e.target.value) || 0)}
                    sx={{ mt: 1 }}
                  />
                </CardContent>
              </Card>
            ))}

            <Divider sx={{ my: 2 }} />

            {/* Wavelength Range */}
            <Typography variant="h6" gutterBottom>
              Wavelength Range
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  size="small"
                  label="Start (um)"
                  type="number"
                  value={wavelengthStart}
                  onChange={(e) => setWavelengthStart(parseFloat(e.target.value) || 0.4)}
                  inputProps={{ step: 0.1 }}
                />
              </Grid>
              <Grid item xs={6}>
                <TextField
                  fullWidth
                  size="small"
                  label="End (um)"
                  type="number"
                  value={wavelengthEnd}
                  onChange={(e) => setWavelengthEnd(parseFloat(e.target.value) || 0.8)}
                  inputProps={{ step: 0.1 }}
                />
              </Grid>
            </Grid>

            <Divider sx={{ my: 2 }} />

            {/* Angle and Polarization */}
            <Typography variant="h6" gutterBottom>
              Incidence Angle: {angleDeg}°
            </Typography>
            <Box sx={{ px: 1 }}>
              <Slider
                value={angleDeg}
                onChange={(_, v) => setAngleDeg(v as number)}
                min={0}
                max={89}
                valueLabelDisplay="auto"
                valueLabelFormat={(v) => `${v}°`}
              />
            </Box>

            <FormControl fullWidth size="small" sx={{ mt: 2 }}>
              <InputLabel>Polarization</InputLabel>
              <Select
                value={polarization}
                label="Polarization"
                onChange={(e) => setPolarization(e.target.value as 's' | 'p' | 'unpolarized')}
              >
                <MenuItem value="unpolarized">Unpolarized</MenuItem>
                <MenuItem value="s">S-polarized (TE)</MenuItem>
                <MenuItem value="p">P-polarized (TM)</MenuItem>
              </Select>
            </FormControl>

            <Button
              fullWidth
              variant="contained"
              startIcon={loading ? <CircularProgress size={20} /> : <CalculateIcon />}
              onClick={handleCalculate}
              disabled={loading}
              sx={{ mt: 3 }}
            >
              {loading ? 'Calculating...' : 'Calculate'}
            </Button>
          </Paper>
        </Grid>

        {/* Results Panel */}
        <Grid item xs={12} md={7}>
          <Paper sx={{ p: 2, height: '100%', minHeight: 500 }}>
            <Typography variant="h6" gutterBottom>
              Results
            </Typography>

            {results ? (
              <Box sx={{ mt: 2 }}>
                <Box sx={{ display: 'flex', gap: 2, mb: 2, flexWrap: 'wrap' }}>
                  <Typography variant="body2">
                    <strong>R:</strong> {(results.reflectance[Math.floor(results.reflectance.length / 2)] * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2">
                    <strong>T:</strong> {(results.transmittance[Math.floor(results.transmittance.length / 2)] * 100).toFixed(1)}%
                  </Typography>
                  <Typography variant="body2">
                    <strong>A:</strong> {(results.absorptance[Math.floor(results.absorptance.length / 2)] * 100).toFixed(1)}%
                  </Typography>
                </Box>
                <Box sx={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.875rem' }}>
                    <thead>
                      <tr style={{ borderBottom: '2px solid #ccc' }}>
                        <th style={{ padding: '8px', textAlign: 'left' }}>Wavelength (nm)</th>
                        <th style={{ padding: '8px', textAlign: 'right' }}>R (%)</th>
                        <th style={{ padding: '8px', textAlign: 'right' }}>T (%)</th>
                        <th style={{ padding: '8px', textAlign: 'right' }}>A (%)</th>
                      </tr>
                    </thead>
                    <tbody>
                      {results.wavelength_um.filter((_, i) => i % 10 === 0).map((wl, i) => {
                        const idx = i * 10;
                        return (
                          <tr key={wl} style={{ borderBottom: '1px solid #eee' }}>
                            <td style={{ padding: '6px' }}>{(wl * 1000).toFixed(0)}</td>
                            <td style={{ padding: '6px', textAlign: 'right' }}>{(results.reflectance[idx] * 100).toFixed(2)}</td>
                            <td style={{ padding: '6px', textAlign: 'right' }}>{(results.transmittance[idx] * 100).toFixed(2)}</td>
                            <td style={{ padding: '6px', textAlign: 'right' }}>{(results.absorptance[idx] * 100).toFixed(2)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </Box>
              </Box>
            ) : (
              <Box
                sx={{
                  height: 400,
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: 'text.secondary',
                }}
              >
                <Typography>Configure layers and click Calculate to see results</Typography>
              </Box>
            )}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
