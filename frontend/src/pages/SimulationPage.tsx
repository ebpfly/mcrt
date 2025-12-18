import { useState, useCallback, useEffect } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Divider,
  Stack,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  Checkbox,
  TextField,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  Refresh as ResetIcon,
  Save as SaveIcon,
  Upload as LoadIcon,
  Delete as DeleteIcon,
  Bookmark as BookmarkIcon,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector, simulationActions, uiActions, comparisonActions } from '../store';
import api from '../services/api';
import sseClient from '../services/sse';
import MaterialSelector from '../components/material/MaterialSelector';
import LayerConfig from '../components/medium/LayerConfig';
import MediumVisualization3D from '../components/medium/MediumVisualization3D';
import WavelengthConfig from '../components/simulation/WavelengthConfig';
import PhotonConfig from '../components/simulation/PhotonConfig';
import ReflectanceChart from '../components/charts/ReflectanceChart';

interface ReferenceData {
  id: string;
  name: string;
  source: string;
  data: {
    wavelength_um: number[];
    reflectance: number[];
  };
}

export default function SimulationPage() {
  const dispatch = useAppDispatch();
  const [isLoading, setIsLoading] = useState(false);
  const [saveDialogOpen, setSaveDialogOpen] = useState(false);
  const [runName, setRunName] = useState('');
  const [referenceData, setReferenceData] = useState<ReferenceData | null>(null);

  const simulation = useAppSelector((state) => state.simulation);
  const medium = useAppSelector((state) => state.medium);
  const comparison = useAppSelector((state) => state.comparison);

  // Load reference data on mount
  useEffect(() => {
    const loadReferenceData = async () => {
      try {
        // Load Adding-Doubling exact solution for Fe2O3 (validated MC reference)
        const refData = await api.getReferenceData('adding_doubling_fe2o3');
        setReferenceData(refData);
      } catch (error) {
        console.log('No reference data available');
      }
    };
    loadReferenceData();
  }, []);

  const handleStartSimulation = useCallback(async () => {
    if (Object.keys(medium.particleMaterials).length === 0) {
      dispatch(uiActions.addNotification({ type: 'error', message: 'Please select a particle material' }));
      return;
    }
    if (Object.keys(medium.matrixMaterials).length === 0) {
      dispatch(uiActions.addNotification({ type: 'error', message: 'Please select a matrix material' }));
      return;
    }

    setIsLoading(true);
    dispatch(simulationActions.setStatus('running'));

    try {
      const config = {
        particle_materials: Object.fromEntries(
          Object.entries(medium.particleMaterials).map(([k, v]) => [
            k,
            { wavelength_um: v.wavelength_um.slice(), n: v.n.slice(), k: v.k.slice() },
          ])
        ),
        matrix_materials: Object.fromEntries(
          Object.entries(medium.matrixMaterials).map(([k, v]) => [
            k,
            { wavelength_um: v.wavelength_um.slice(), n: v.n.slice(), k: v.k.slice() },
          ])
        ),
        layers: medium.layers,
        wavelength_start_um: simulation.wavelengthStart,
        wavelength_end_um: simulation.wavelengthEnd,
        wavelength_interval_um: simulation.wavelengthInterval,
        photons_target: simulation.photonsTarget,
        n_batches: simulation.nBatches,
      };

      const session = await api.startSimulation(config);
      dispatch(simulationActions.setCurrentSession(session));

      // Connect to SSE stream
      const streamUrl = api.getStreamURL(session.session_id);
      sseClient.connect(streamUrl, session.session_id, {
        onBatchComplete: (data) => {
          dispatch(
            simulationActions.updateFromBatch({
              progress: data.progress_percent,
              results: data.results,
            })
          );
        },
        onSimulationComplete: (data) => {
          dispatch(simulationActions.setStatus(data.status));
          dispatch(simulationActions.setResults(data.results));
          dispatch(uiActions.addNotification({ type: 'success', message: 'Simulation completed!' }));
          setIsLoading(false);
        },
        onError: (data) => {
          dispatch(simulationActions.setStatus('error'));
          dispatch(uiActions.addNotification({ type: 'error', message: data.error }));
          setIsLoading(false);
        },
        onConnectionError: () => {
          dispatch(uiActions.addNotification({ type: 'error', message: 'Connection lost' }));
          setIsLoading(false);
        },
      });
    } catch (error) {
      dispatch(simulationActions.setStatus('error'));
      dispatch(uiActions.addNotification({ type: 'error', message: 'Failed to start simulation' }));
      setIsLoading(false);
    }
  }, [dispatch, medium, simulation]);

  const handleStopSimulation = useCallback(async () => {
    if (simulation.currentSession) {
      try {
        await api.stopSimulation(simulation.currentSession.session_id);
        dispatch(simulationActions.setStatus('paused'));
        sseClient.disconnect();
        setIsLoading(false);
      } catch (error) {
        dispatch(uiActions.addNotification({ type: 'error', message: 'Failed to stop simulation' }));
      }
    }
  }, [dispatch, simulation.currentSession]);

  const handleResetSimulation = useCallback(() => {
    sseClient.disconnect();
    dispatch(simulationActions.resetSimulation());
    setIsLoading(false);
  }, [dispatch]);

  const handleExportState = useCallback(async () => {
    if (!simulation.currentSession) return;
    try {
      const blob = await api.exportState(simulation.currentSession.session_id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `mcrt_state_${simulation.currentSession.session_id.slice(0, 8)}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      dispatch(uiActions.addNotification({ type: 'error', message: 'Failed to export state' }));
    }
  }, [dispatch, simulation.currentSession]);

  const handleImportState = useCallback(async () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      try {
        const result = await api.restoreState(file);
        const session = await api.getSimulation(result.session_id);
        dispatch(simulationActions.setCurrentSession(session));
        dispatch(uiActions.addNotification({ type: 'success', message: result.message }));
      } catch (error) {
        dispatch(uiActions.addNotification({ type: 'error', message: 'Failed to import state' }));
      }
    };
    input.click();
  }, [dispatch]);

  const handleSaveRun = useCallback(() => {
    if (!simulation.results) return;

    const config = {
      particle_materials: Object.fromEntries(
        Object.entries(medium.particleMaterials).map(([k, v]) => [
          k,
          { wavelength_um: v.wavelength_um.slice(), n: v.n.slice(), k: v.k.slice() },
        ])
      ),
      matrix_materials: Object.fromEntries(
        Object.entries(medium.matrixMaterials).map(([k, v]) => [
          k,
          { wavelength_um: v.wavelength_um.slice(), n: v.n.slice(), k: v.k.slice() },
        ])
      ),
      layers: medium.layers,
      wavelength_start_um: simulation.wavelengthStart,
      wavelength_end_um: simulation.wavelengthEnd,
      wavelength_interval_um: simulation.wavelengthInterval,
      photons_target: simulation.photonsTarget,
      n_batches: simulation.nBatches,
    };

    dispatch(comparisonActions.addSavedRun({
      id: Math.random().toString(36).substr(2, 9),
      name: runName || `Run ${comparison.savedRuns.length + 1}`,
      config,
      results: simulation.results,
      timestamp: new Date().toISOString(),
    }));

    setSaveDialogOpen(false);
    setRunName('');
    dispatch(uiActions.addNotification({ type: 'success', message: 'Run saved for comparison' }));
  }, [dispatch, simulation, medium, comparison.savedRuns.length, runName]);

  const isRunning = simulation.status === 'running';
  const canStart = simulation.status === 'idle' || simulation.status === 'error';

  return (
    <Box>
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
                <Stack spacing={1}>
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    {canStart ? (
                      <Button
                        variant="contained"
                        color="primary"
                        startIcon={<PlayIcon />}
                        onClick={handleStartSimulation}
                        disabled={isLoading || Object.keys(medium.particleMaterials).length === 0 || Object.keys(medium.matrixMaterials).length === 0}
                        fullWidth
                      >
                        {Object.keys(medium.particleMaterials).length === 0 || Object.keys(medium.matrixMaterials).length === 0
                          ? 'Loading Materials...'
                          : 'Start Simulation'}
                      </Button>
                    ) : isRunning ? (
                      <Button
                        variant="contained"
                        color="secondary"
                        startIcon={<StopIcon />}
                        onClick={handleStopSimulation}
                        fullWidth
                      >
                        Stop
                      </Button>
                    ) : (
                      <Button
                        variant="outlined"
                        startIcon={<ResetIcon />}
                        onClick={handleResetSimulation}
                        fullWidth
                      >
                        Reset
                      </Button>
                    )}
                  </Box>
                  <Divider />
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    <Button
                      variant="outlined"
                      size="small"
                      startIcon={<SaveIcon />}
                      onClick={handleExportState}
                      disabled={!simulation.currentSession}
                    >
                      Save
                    </Button>
                    <Button
                      variant="outlined"
                      size="small"
                      startIcon={<LoadIcon />}
                      onClick={handleImportState}
                    >
                      Load
                    </Button>
                  </Box>
                </Stack>
              </CardContent>
            </Card>

            {/* Material Selection */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Materials
                </Typography>
                <MaterialSelector />
              </CardContent>
            </Card>

            {/* Layer Configuration */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Medium Configuration
                </Typography>
                <LayerConfig />
              </CardContent>
            </Card>

            {/* 3D Visualization */}
            <MediumVisualization3D />

            {/* Simulation Settings */}
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Simulation Settings
                </Typography>
                <Stack spacing={2}>
                  <WavelengthConfig />
                  <Divider />
                  <PhotonConfig />
                </Stack>
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
                {simulation.status !== 'idle' && (
                  <Typography variant="body2" color="text.secondary">
                    {simulation.progressPercent.toFixed(1)}% complete
                  </Typography>
                )}
              </Box>

              {isRunning && (
                <LinearProgress
                  variant="determinate"
                  value={simulation.progressPercent}
                  sx={{ mb: 2 }}
                />
              )}

              <ReflectanceChart
                results={simulation.results}
                savedRuns={comparison.savedRuns}
                visibleRunIds={comparison.selectedRunIds}
                referenceData={referenceData}
              />

              {simulation.results && (
                <Box sx={{ mt: 2, display: 'flex', gap: 2 }}>
                  <Button
                    size="small"
                    variant="outlined"
                    onClick={async () => {
                      if (simulation.currentSession) {
                        const blob = await api.exportResultsCSV(simulation.currentSession.session_id);
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'results.csv';
                        a.click();
                        URL.revokeObjectURL(url);
                      }
                    }}
                  >
                    Export CSV
                  </Button>
                  <Button
                    size="small"
                    variant="outlined"
                    startIcon={<BookmarkIcon />}
                    onClick={() => setSaveDialogOpen(true)}
                  >
                    Save for Comparison
                  </Button>
                </Box>
              )}

              {/* Saved Runs Panel */}
              {comparison.savedRuns.length > 0 && (
                <Box sx={{ mt: 3 }}>
                  <Divider sx={{ mb: 2 }} />
                  <Typography variant="subtitle2" gutterBottom>
                    Saved Runs ({comparison.savedRuns.length})
                  </Typography>
                  <List dense>
                    {comparison.savedRuns.map((run) => (
                      <ListItem key={run.id} sx={{ py: 0 }}>
                        <Checkbox
                          edge="start"
                          checked={comparison.selectedRunIds.includes(run.id)}
                          onChange={() => dispatch(comparisonActions.toggleRunSelection(run.id))}
                          size="small"
                        />
                        <ListItemText
                          primary={run.name}
                          secondary={`${run.config.wavelength_start_um}-${run.config.wavelength_end_um} um, ${run.config.photons_target} photons`}
                        />
                        <ListItemSecondaryAction>
                          <IconButton
                            edge="end"
                            size="small"
                            onClick={() => dispatch(comparisonActions.removeSavedRun(run.id))}
                          >
                            <DeleteIcon fontSize="small" />
                          </IconButton>
                        </ListItemSecondaryAction>
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Save Run Dialog */}
      <Dialog open={saveDialogOpen} onClose={() => setSaveDialogOpen(false)}>
        <DialogTitle>Save Run for Comparison</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Run Name"
            fullWidth
            variant="outlined"
            value={runName}
            onChange={(e) => setRunName(e.target.value)}
            placeholder={`Run ${comparison.savedRuns.length + 1}`}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setSaveDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleSaveRun} variant="contained">Save</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
