import {
  Box,
  TextField,
  Typography,
  Slider,
  InputAdornment,
} from '@mui/material';
import { useAppDispatch, useAppSelector, simulationActions } from '../../store';

export default function WavelengthConfig() {
  const dispatch = useAppDispatch();
  const { wavelengthStart, wavelengthEnd, wavelengthInterval } = useAppSelector(
    (state) => state.simulation
  );

  const handleRangeChange = (_: Event, newValue: number | number[]) => {
    const [start, end] = newValue as number[];
    dispatch(
      simulationActions.setWavelengthRange({
        start,
        end,
        interval: wavelengthInterval,
      })
    );
  };

  const handleIntervalChange = (value: number) => {
    dispatch(
      simulationActions.setWavelengthRange({
        start: wavelengthStart,
        end: wavelengthEnd,
        interval: value,
      })
    );
  };

  const nWavelengths = Math.floor((wavelengthEnd - wavelengthStart) / wavelengthInterval) + 1;

  return (
    <Box>
      <Typography variant="subtitle2" gutterBottom>
        Wavelength Range
      </Typography>

      <Box sx={{ px: 1, mb: 2 }}>
        <Slider
          value={[wavelengthStart, wavelengthEnd]}
          onChange={handleRangeChange}
          min={0.3}
          max={16.0}
          step={0.1}
          valueLabelDisplay="auto"
          valueLabelFormat={(value) => `${value.toFixed(1)} µm`}
        />
      </Box>

      <Box sx={{ display: 'flex', gap: 2, mb: 2 }}>
        <TextField
          size="small"
          label="Start"
          type="number"
          value={wavelengthStart}
          onChange={(e) =>
            dispatch(
              simulationActions.setWavelengthRange({
                start: Number(e.target.value),
                end: wavelengthEnd,
                interval: wavelengthInterval,
              })
            )
          }
          inputProps={{ step: 0.01 }}
          InputProps={{
            endAdornment: <InputAdornment position="end">um</InputAdornment>,
          }}
        />
        <TextField
          size="small"
          label="End"
          type="number"
          value={wavelengthEnd}
          onChange={(e) =>
            dispatch(
              simulationActions.setWavelengthRange({
                start: wavelengthStart,
                end: Number(e.target.value),
                interval: wavelengthInterval,
              })
            )
          }
          inputProps={{ step: 0.01 }}
          InputProps={{
            endAdornment: <InputAdornment position="end">um</InputAdornment>,
          }}
        />
      </Box>

      <Typography variant="subtitle2" gutterBottom>
        Wavelength Interval
      </Typography>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        <Slider
          value={wavelengthInterval * 1000}
          min={10}
          max={500}
          step={10}
          onChange={(_, value) => handleIntervalChange((value as number) / 1000)}
          sx={{ flexGrow: 1 }}
        />
        <TextField
          size="small"
          type="number"
          value={Math.round(wavelengthInterval * 1000)}
          onChange={(e) => handleIntervalChange(Number(e.target.value) / 1000)}
          inputProps={{ step: 10 }}
          InputProps={{
            endAdornment: <InputAdornment position="end">nm</InputAdornment>,
          }}
          sx={{ width: 160 }}
        />
      </Box>

      <Typography variant="caption" color="text.secondary" sx={{ mt: 1, display: 'block' }}>
        {nWavelengths} wavelength points
      </Typography>
    </Box>
  );
}
