import { useState, useCallback } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  IconButton,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Slider,
  Alert,
  CircularProgress,
  Stack,
  Autocomplete,
} from '@mui/material';
import {
  Add as AddIcon,
  Delete as DeleteIcon,
  Calculate as CalculateIcon,
} from '@mui/icons-material';
import api from '../services/api';
import { useAppSelector } from '../store';
import { MaterialInfo, ThinFilmResults, OpticalConstants, SimulationResults } from '../types';
import ReflectanceChart from '../components/charts/ReflectanceChart';

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

  // Convert ThinFilmResults to SimulationResults format for the chart
  const chartResults: SimulationResults | null = results
    ? {
        wavelength_um: results.wavelength_um,
        reflectance: results.reflectance,
        transmittance: results.transmittance,
        absorptance: results.absorptance,
      }
    : null;

  // Export CSV
  const handleExportCSV = () => {
    if (!results) return;

    const headers = ['wavelength_nm', 'reflectance', 'transmittance', 'absorptance'];
    const rows = results.wavelength_um.map((wl, i) => [
      (wl * 1000).toFixed(1),
      results.reflectance[i].toFixed(6),
      results.transmittance[i].toFixed(6),
      results.absorptance[i].toFixed(6),
    ]);

    const csv = [headers.join(','), ...rows.map((r) => r.join(','))].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'thinfilm_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Box>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Left Column - Configuration */}
        <Grid item xs={12} md={4}>
          <Stack spacing={2}>
            {/* Controls */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Controls
                </Typography>
                <Button
                  fullWidth
                  variant="contained"
                  color="primary"
                  startIcon={loading ? <CircularProgress size={20} color="inherit" /> : <CalculateIcon />}
                  onClick={handleCalculate}
                  disabled={loading}
                >
                  {loading ? 'Calculating...' : 'Calculate'}
                </Button>
              </CardContent>
            </Card>

            {/* Substrate Material */}
            <Card>
              <CardContent>
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
              </CardContent>
            </Card>

            {/* Thin Film Layers */}
            <Card>
              <CardContent>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                  <Typography variant="h6">Thin Film Layers</Typography>
                  <Button startIcon={<AddIcon />} onClick={addLayer} size="small">
                    Add Layer
                  </Button>
                </Box>

                <Stack spacing={2}>
                  {layers.map((layer, index) => (
                    <Box
                      key={layer.id}
                      sx={{
                        p: 1.5,
                        border: '1px solid',
                        borderColor: 'divider',
                        borderRadius: 1,
                      }}
                    >
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                        <Typography variant="subtitle2">Layer {index + 1}</Typography>
                        {layers.length > 1 && (
                          <IconButton size="small" onClick={() => removeLayer(layer.id)}>
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        )}
                      </Box>
                      <Stack spacing={1}>
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
                        />
                      </Stack>
                    </Box>
                  ))}
                </Stack>
              </CardContent>
            </Card>

            {/* Wavelength Settings */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Wavelength Range
                </Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <TextField
                      fullWidth
                      size="small"
                      label="Start (μm)"
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
                      label="End (μm)"
                      type="number"
                      value={wavelengthEnd}
                      onChange={(e) => setWavelengthEnd(parseFloat(e.target.value) || 0.8)}
                      inputProps={{ step: 0.1 }}
                    />
                  </Grid>
                </Grid>
              </CardContent>
            </Card>

            {/* Angle and Polarization */}
            <Card>
              <CardContent>
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
              </CardContent>
            </Card>
          </Stack>
        </Grid>

        {/* Right Column - Results */}
        <Grid item xs={12} md={8}>
          <Card sx={{ height: '100%', minHeight: 600 }}>
            <CardContent>
              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                <Typography variant="h6">Results</Typography>
                {results && (
                  <Typography variant="body2" color="text.secondary">
                    {angleDeg}° incidence, {polarization}
                  </Typography>
                )}
              </Box>

              <ReflectanceChart results={chartResults} />

              {results && (
                <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                  <Button size="small" variant="outlined" onClick={handleExportCSV}>
                    Export CSV
                  </Button>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
