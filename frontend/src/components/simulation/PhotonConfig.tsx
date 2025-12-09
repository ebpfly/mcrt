import {
  Box,
  TextField,
  Typography,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { useAppDispatch, useAppSelector, simulationActions } from '../../store';

export default function PhotonConfig() {
  const dispatch = useAppDispatch();
  const { photonsTarget, nBatches } = useAppSelector((state) => state.simulation);

  const photonPresets = [
    { label: '100 (Test)', value: 100 },
    { label: '1K (Debug)', value: 1000 },
    { label: '10K (Quick)', value: 10000 },
    { label: '50K (Fast)', value: 50000 },
    { label: '100K (Standard)', value: 100000 },
    { label: '500K (High Quality)', value: 500000 },
    { label: '1M (Publication)', value: 1000000 },
  ];

  const handlePhotonPreset = (value: number) => {
    dispatch(simulationActions.setPhotonsTarget(value));
  };

  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom>
        Total Photons
      </Typography>

      <FormControl size="small" fullWidth sx={{ mb: 2 }}>
        <InputLabel>Preset</InputLabel>
        <Select
          value={photonPresets.find((p) => p.value === photonsTarget)?.value || ''}
          label="Preset"
          onChange={(e) => handlePhotonPreset(e.target.value as number)}
        >
          {photonPresets.map((preset) => (
            <MenuItem key={preset.value} value={preset.value}>
              {preset.label}
            </MenuItem>
          ))}
        </Select>
      </FormControl>

      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
        <Slider
          value={Math.log10(photonsTarget)}
          min={3}
          max={6}
          step={0.1}
          onChange={(_, value) =>
            dispatch(simulationActions.setPhotonsTarget(Math.round(Math.pow(10, value as number))))
          }
          sx={{ flexGrow: 1 }}
        />
        <TextField
          size="small"
          type="number"
          value={photonsTarget}
          onChange={(e) => dispatch(simulationActions.setPhotonsTarget(Number(e.target.value)))}
          sx={{ width: 130 }}
        />
      </Box>

      <Typography variant="subtitle2" gutterBottom>
        Number of Batches
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Slider
          value={nBatches}
          min={1}
          max={50}
          step={1}
          onChange={(_, value) => dispatch(simulationActions.setNBatches(value as number))}
          sx={{ flexGrow: 1 }}
        />
        <TextField
          size="small"
          type="number"
          value={nBatches}
          onChange={(e) => dispatch(simulationActions.setNBatches(Number(e.target.value)))}
          sx={{ width: 80 }}
        />
      </Box>

      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
        {Math.floor(photonsTarget / nBatches).toLocaleString()} photons per batch
      </Typography>
    </Box>
  );
}
