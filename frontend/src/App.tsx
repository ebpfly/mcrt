import { useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box, Container, Alert, Snackbar } from '@mui/material';
import { useAppDispatch, useAppSelector, uiActions, healthActions } from './store';
import api from './services/api';
import Layout from './components/Layout';
import SimulationPage from './pages/SimulationPage';
import ComparisonPage from './pages/ComparisonPage';
import DocumentationPage from './pages/DocumentationPage';
import ThinFilmPage from './pages/ThinFilmPage';

function App() {
  const dispatch = useAppDispatch();
  const notifications = useAppSelector((state) => state.ui.notifications);
  const health = useAppSelector((state) => state.health.status);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const status = await api.getHealth();
        dispatch(healthActions.setHealthStatus(status));
      } catch (error) {
        dispatch(
          uiActions.addNotification({
            type: 'error',
            message: 'Failed to connect to API server',
          })
        );
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [dispatch]);

  useEffect(() => {
    const loadMaterials = async () => {
      try {
        const shelves = await api.listShelves();
        dispatch(healthActions.setAvailableShelves(shelves));

        const { materials } = await api.listMaterials({ limit: 5000 });
        dispatch(healthActions.setAvailableMaterials(materials));
      } catch (error) {
        console.error('Failed to load materials:', error);
      }
    };

    if (health?.database_available) {
      loadMaterials();
    }
  }, [dispatch, health?.database_available]);

  const handleCloseNotification = (id: string) => {
    dispatch(uiActions.removeNotification(id));
  };

  return (
    <Layout>
      <Box sx={{ py: 2 }}>
        <Container maxWidth="xl">
          {health && !health.fos_available && (
            <Alert severity="warning" sx={{ mb: 2 }}>
              FOS simulation engine not available. Please check the server configuration.
            </Alert>
          )}

          <Routes>
            <Route path="/" element={<Navigate to="/simulation" replace />} />
            <Route path="/simulation" element={<SimulationPage />} />
            <Route path="/thinfilm" element={<ThinFilmPage />} />
            <Route path="/comparison" element={<ComparisonPage />} />
            <Route path="/docs" element={<DocumentationPage />} />
          </Routes>
        </Container>
      </Box>

      {notifications.map((notification) => (
        <Snackbar
          key={notification.id}
          open={true}
          autoHideDuration={6000}
          onClose={() => handleCloseNotification(notification.id)}
          anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
        >
          <Alert
            onClose={() => handleCloseNotification(notification.id)}
            severity={notification.type}
            sx={{ width: '100%' }}
          >
            {notification.message}
          </Alert>
        </Snackbar>
      ))}
    </Layout>
  );
}

export default App;
