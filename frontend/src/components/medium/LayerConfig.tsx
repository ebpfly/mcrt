import {
  Box,
  TextField,
  Typography,
  Slider,
  InputAdornment,
  Alert,
} from '@mui/material';
import { useAppDispatch, useAppSelector, mediumActions } from '../../store';

// Volume fraction thresholds for warnings
const VF_DEPENDENT_SCATTERING = 30; // Above this, independent scattering assumption weakens
const VF_PHYSICAL_MAX = 64; // Random close packing limit for spheres

export default function LayerConfig() {
  const dispatch = useAppDispatch();
  const layers = useAppSelector((state) => state.medium.layers);

  // For simplicity, we handle the first layer only
  const layer = layers[0];
  const particle = layer?.particles[0];

  const handleLayerChange = (field: string, value: number) => {
    if (!layer) return;
    dispatch(
      mediumActions.updateLayer({
        index: 0,
        layer: { ...layer, [field]: value },
      })
    );
  };

  const handleParticleChange = (field: string, value: number) => {
    if (!particle) return;
    dispatch(
      mediumActions.updateParticle({
        layerIndex: 0,
        particleIndex: 0,
        particle: { ...particle, [field]: value },
      })
    );
  };

  if (!layer || !particle) {
    return <Typography color="text.secondary">No layer configured</Typography>;
  }

  return (
    <Box>
      {/* Layer Thickness */}
      <Typography variant="subtitle2" gutterBottom>
        Layer Thickness
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <Slider
          value={layer.thickness_um}
          min={10}
          max={1000}
          step={10}
          onChange={(_, value) => handleLayerChange('thickness_um', value as number)}
          sx={{ flexGrow: 1 }}
        />
        <TextField
          size="small"
          type="number"
          value={layer.thickness_um}
          onChange={(e) => handleLayerChange('thickness_um', Number(e.target.value))}
          InputProps={{
            endAdornment: <InputAdornment position="end">um</InputAdornment>,
          }}
          sx={{ width: 120 }}
        />
      </Box>

      {/* Particle Diameter */}
      <Typography variant="subtitle2" gutterBottom>
        Particle Diameter
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <Slider
          value={particle.diameter_um}
          min={0.1}
          max={100}
          step={0.1}
          onChange={(_, value) => handleParticleChange('diameter_um', value as number)}
          valueLabelDisplay="auto"
          valueLabelFormat={(v) => v.toFixed(2)}
          sx={{ flexGrow: 1 }}
        />
        <TextField
          size="small"
          type="number"
          value={particle.diameter_um.toFixed(2)}
          onChange={(e) => handleParticleChange('diameter_um', Number(e.target.value))}
          inputProps={{ step: 0.1 }}
          InputProps={{
            endAdornment: <InputAdornment position="end">um</InputAdornment>,
          }}
          sx={{ width: 140 }}
        />
      </Box>

      {/* Volume Fraction */}
      <Typography variant="subtitle2" gutterBottom>
        Volume Fraction
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 1 }}>
        <Slider
          value={particle.volume_fraction}
          min={0}
          max={VF_PHYSICAL_MAX}
          step={1}
          onChange={(_, value) => handleParticleChange('volume_fraction', value as number)}
          marks={[
            { value: VF_DEPENDENT_SCATTERING, label: '' },
          ]}
          sx={{
            flexGrow: 1,
            '& .MuiSlider-mark': {
              backgroundColor: 'warning.main',
              height: 12,
              width: 2,
            },
          }}
        />
        <TextField
          size="small"
          type="number"
          value={particle.volume_fraction}
          onChange={(e) => {
            const val = Math.min(VF_PHYSICAL_MAX, Math.max(0, Number(e.target.value)));
            handleParticleChange('volume_fraction', val);
          }}
          InputProps={{
            endAdornment: <InputAdornment position="end">%</InputAdornment>,
          }}
          sx={{ width: 120 }}
        />
      </Box>
      {particle.volume_fraction > VF_DEPENDENT_SCATTERING && particle.volume_fraction <= VF_PHYSICAL_MAX && (
        <Alert severity="warning" sx={{ mb: 2, py: 0 }}>
          Above {VF_DEPENDENT_SCATTERING}%: independent scattering assumption weakens. Results may be approximate.
        </Alert>
      )}
      {particle.volume_fraction > VF_PHYSICAL_MAX && (
        <Alert severity="error" sx={{ mb: 2, py: 0 }}>
          Above {VF_PHYSICAL_MAX}%: exceeds random close packing limit for spheres.
        </Alert>
      )}
      {particle.volume_fraction <= VF_DEPENDENT_SCATTERING && (
        <Box sx={{ mb: 2 }} />
      )}

      {/* Size Distribution Std Dev */}
      <Typography variant="subtitle2" gutterBottom>
        Size Distribution Std Dev
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Slider
          value={particle.std_dev}
          min={0}
          max={1}
          step={0.01}
          onChange={(_, value) => handleParticleChange('std_dev', value as number)}
          sx={{ flexGrow: 1 }}
        />
        <TextField
          size="small"
          type="number"
          value={particle.std_dev}
          onChange={(e) => handleParticleChange('std_dev', Number(e.target.value))}
          inputProps={{ step: 0.01 }}
          sx={{ width: 120 }}
        />
      </Box>
    </Box>
  );
}
