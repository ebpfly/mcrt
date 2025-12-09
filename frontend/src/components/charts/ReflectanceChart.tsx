import { useMemo, useState } from 'react';
import { Box, FormControlLabel, Checkbox, Typography, Tooltip as MuiTooltip } from '@mui/material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
} from 'chart.js';
import { Line } from 'react-chartjs-2';
import { SimulationResults, SavedRun, OpticalConstants } from '../../types';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

// Color palette for saved runs
const SAVED_RUN_COLORS = [
  { r: 156, g: 39, b: 176 },   // purple
  { r: 255, g: 152, b: 0 },    // orange
  { r: 0, g: 150, b: 136 },    // teal
  { r: 233, g: 30, b: 99 },    // pink
  { r: 63, g: 81, b: 181 },    // indigo
  { r: 139, g: 195, b: 74 },   // light green
  { r: 121, g: 85, b: 72 },    // brown
  { r: 0, g: 188, b: 212 },    // cyan
];

interface ReferenceData {
  id: string;
  name: string;
  source: string;
  data: {
    wavelength_um: number[];
    reflectance: number[];
  };
}

interface ReflectanceChartProps {
  results: SimulationResults | null;
  savedRuns?: SavedRun[];
  visibleRunIds?: string[];
  referenceData?: ReferenceData | null;
}

export default function ReflectanceChart({ results, savedRuns = [], visibleRunIds = [], referenceData = null }: ReflectanceChartProps) {
  const [showReflectance, setShowReflectance] = useState(true);
  const [showAbsorptance, setShowAbsorptance] = useState(true);
  const [showTransmittance, setShowTransmittance] = useState(true);
  const [showReference, setShowReference] = useState(true);

  // Get visible saved runs
  const visibleSavedRuns = useMemo(() => {
    return savedRuns.filter(run => visibleRunIds.includes(run.id));
  }, [savedRuns, visibleRunIds]);

  // Find common wavelength range for all data
  const wavelengthLabels = useMemo(() => {
    if (results) {
      return results.wavelength_um.map((w) => w.toFixed(3));
    }
    if (visibleSavedRuns.length > 0) {
      return visibleSavedRuns[0].results.wavelength_um.map((w) => w.toFixed(3));
    }
    return [];
  }, [results, visibleSavedRuns]);

  const chartData = useMemo(() => {
    if (!results && visibleSavedRuns.length === 0) {
      return {
        labels: [],
        datasets: [],
      };
    }

    const datasets: Array<{
      label: string;
      data: number[];
      borderColor: string;
      backgroundColor: string;
      fill: boolean;
      tension: number;
      pointRadius: number;
      borderWidth: number;
      borderDash?: number[];
    }> = [];

    // Current results (solid lines)
    if (results) {
      if (showReflectance) {
        datasets.push({
          label: 'Reflectance',
          data: results.reflectance,
          borderColor: 'rgb(59, 130, 246)',
          backgroundColor: 'rgba(59, 130, 246, 0.1)',
          fill: false,
          tension: 0.1,
          pointRadius: 0,
          borderWidth: 2,
        });
      }

      if (showAbsorptance) {
        datasets.push({
          label: 'Absorptance',
          data: results.absorptance,
          borderColor: 'rgb(239, 68, 68)',
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          fill: false,
          tension: 0.1,
          pointRadius: 0,
          borderWidth: 2,
        });
      }

      if (showTransmittance) {
        datasets.push({
          label: 'Transmittance',
          data: results.transmittance,
          borderColor: 'rgb(34, 197, 94)',
          backgroundColor: 'rgba(34, 197, 94, 0.1)',
          fill: false,
          tension: 0.1,
          pointRadius: 0,
          borderWidth: 2,
        });
      }
    }

    // Reference data (thick dotted black line)
    if (showReference && referenceData && wavelengthLabels.length > 0) {
      // Interpolate reference data to match simulation wavelengths
      const refWavelengths = referenceData.data.wavelength_um;
      const refReflectances = referenceData.data.reflectance;

      const interpolatedRef = wavelengthLabels.map((wlStr) => {
        const wl = parseFloat(wlStr);
        // Find surrounding points for linear interpolation
        let i = 0;
        while (i < refWavelengths.length - 1 && refWavelengths[i + 1] < wl) {
          i++;
        }
        if (wl <= refWavelengths[0]) return refReflectances[0];
        if (wl >= refWavelengths[refWavelengths.length - 1]) return refReflectances[refReflectances.length - 1];
        // Linear interpolation
        const t = (wl - refWavelengths[i]) / (refWavelengths[i + 1] - refWavelengths[i]);
        return refReflectances[i] + t * (refReflectances[i + 1] - refReflectances[i]);
      });

      datasets.push({
        label: `Reference: ${referenceData.name}`,
        data: interpolatedRef,
        borderColor: 'rgb(0, 0, 0)',
        backgroundColor: 'rgba(0, 0, 0, 0.1)',
        fill: false,
        tension: 0.1,
        pointRadius: 0,
        borderWidth: 2,
        borderDash: [3, 3],
      });
    }

    // Saved runs (dashed lines with unique colors)
    visibleSavedRuns.forEach((run, index) => {
      const color = SAVED_RUN_COLORS[index % SAVED_RUN_COLORS.length];
      const colorStr = `rgb(${color.r}, ${color.g}, ${color.b})`;
      const colorAlpha = `rgba(${color.r}, ${color.g}, ${color.b}, 0.1)`;

      if (showReflectance) {
        datasets.push({
          label: `${run.name} - R`,
          data: run.results.reflectance,
          borderColor: colorStr,
          backgroundColor: colorAlpha,
          fill: false,
          tension: 0.1,
          pointRadius: 0,
          borderWidth: 2,
          borderDash: [5, 5],
        });
      }

      if (showAbsorptance) {
        datasets.push({
          label: `${run.name} - A`,
          data: run.results.absorptance,
          borderColor: colorStr,
          backgroundColor: colorAlpha,
          fill: false,
          tension: 0.1,
          pointRadius: 0,
          borderWidth: 2,
          borderDash: [10, 5],
        });
      }

      if (showTransmittance) {
        datasets.push({
          label: `${run.name} - T`,
          data: run.results.transmittance,
          borderColor: colorStr,
          backgroundColor: colorAlpha,
          fill: false,
          tension: 0.1,
          pointRadius: 0,
          borderWidth: 2,
          borderDash: [2, 2],
        });
      }
    });

    return {
      labels: wavelengthLabels,
      datasets,
    };
  }, [results, visibleSavedRuns, wavelengthLabels, showReflectance, showAbsorptance, showTransmittance, showReference, referenceData]);

  const options = useMemo(
    () => ({
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top' as const,
        },
        title: {
          display: false,
        },
        tooltip: {
          mode: 'index' as const,
          intersect: false,
          callbacks: {
            title: (items: Array<{ label: string }>) => {
              if (items.length > 0) {
                return `Wavelength: ${items[0].label} um`;
              }
              return '';
            },
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            label: (context: any) => {
              const label = context.dataset?.label || 'Value';
              const value = context.parsed?.y ?? 0;
              return `${label}: ${(value * 100).toFixed(2)}%`;
            },
          },
        },
      },
      scales: {
        x: {
          title: {
            display: true,
            text: 'Wavelength (um)',
          },
          ticks: {
            maxTicksLimit: 10,
          },
        },
        y: {
          title: {
            display: true,
            text: 'Value',
          },
          min: 0,
          max: 1,
          ticks: {
            callback: (value: string | number) => `${(Number(value) * 100).toFixed(0)}%`,
          },
        },
      },
      interaction: {
        mode: 'nearest' as const,
        axis: 'x' as const,
        intersect: false,
      },
    }),
    []
  );

  if (!results && visibleSavedRuns.length === 0) {
    return (
      <Box
        sx={{
          height: 400,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          bgcolor: 'grey.50',
          borderRadius: 1,
        }}
      >
        <Typography color="text.secondary">
          Run a simulation to see results
        </Typography>
      </Box>
    );
  }

  return (
    <Box>
      <Box sx={{ display: 'flex', gap: 2, mb: 2, flexWrap: 'wrap' }}>
        <FormControlLabel
          control={
            <Checkbox
              checked={showReflectance}
              onChange={(e) => setShowReflectance(e.target.checked)}
              size="small"
              sx={{ color: 'rgb(59, 130, 246)' }}
            />
          }
          label="Reflectance"
        />
        <FormControlLabel
          control={
            <Checkbox
              checked={showAbsorptance}
              onChange={(e) => setShowAbsorptance(e.target.checked)}
              size="small"
              sx={{ color: 'rgb(239, 68, 68)' }}
            />
          }
          label="Absorptance"
        />
        <FormControlLabel
          control={
            <Checkbox
              checked={showTransmittance}
              onChange={(e) => setShowTransmittance(e.target.checked)}
              size="small"
              sx={{ color: 'rgb(34, 197, 94)' }}
            />
          }
          label="Transmittance"
        />
        {referenceData && (
          <MuiTooltip title={`${referenceData.source}: ${referenceData.name}`} arrow>
            <FormControlLabel
              control={
                <Checkbox
                  checked={showReference}
                  onChange={(e) => setShowReference(e.target.checked)}
                  size="small"
                  sx={{ color: 'rgb(0, 0, 0)' }}
                />
              }
              label="Reference"
            />
          </MuiTooltip>
        )}
      </Box>
      <Box sx={{ height: 400 }}>
        <Line data={chartData} options={options} />
      </Box>
    </Box>
  );
}
