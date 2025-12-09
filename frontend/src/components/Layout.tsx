import { ReactNode, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  AppBar,
  Box,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Chip,
  Divider,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Science as ScienceIcon,
  CompareArrows as CompareIcon,
  MenuBook as DocsIcon,
  CheckCircle as HealthyIcon,
  Error as ErrorIcon,
} from '@mui/icons-material';
import { useAppSelector } from '../store';

const drawerWidth = 240;

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const health = useAppSelector((state) => state.health.status);
  const simulationStatus = useAppSelector((state) => state.simulation.status);

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const menuItems = [
    { text: 'Simulation', icon: <ScienceIcon />, path: '/simulation' },
    { text: 'Comparison', icon: <CompareIcon />, path: '/comparison' },
    { text: 'Documentation', icon: <DocsIcon />, path: '/docs' },
  ];

  const drawer = (
    // @ts-expect-error MUI's sx prop union type is too complex for TypeScript
    <Box>
      <Toolbar>
        <Typography variant="h6" noWrap component="div" sx={{ fontWeight: 'bold' }}>
          MCRT
        </Typography>
      </Toolbar>
      <Divider />
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => {
                navigate(item.path);
                setMobileOpen(false);
              }}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>
      <Divider />
      <Box sx={{ p: 2 }}>
        <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
          Status
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
          <Chip
            size="small"
            icon={health?.fos_available ? <HealthyIcon /> : <ErrorIcon />}
            label={health?.fos_available ? 'FOS Ready' : 'FOS Unavailable'}
            color={health?.fos_available ? 'success' : 'error'}
            variant="outlined"
          />
          <Chip
            size="small"
            icon={health?.database_available ? <HealthyIcon /> : <ErrorIcon />}
            label={health?.database_available ? 'DB Ready' : 'DB Unavailable'}
            color={health?.database_available ? 'success' : 'error'}
            variant="outlined"
          />
          {simulationStatus !== 'idle' && (
            <Chip
              size="small"
              label={`Sim: ${simulationStatus}`}
              color={
                simulationStatus === 'running'
                  ? 'primary'
                  : simulationStatus === 'completed'
                  ? 'success'
                  : simulationStatus === 'error'
                  ? 'error'
                  : 'default'
              }
              variant="outlined"
            />
          )}
        </Box>
      </Box>
    </Box>
  );

  return (
    // @ts-expect-error MUI's sx prop union type is too complex for TypeScript
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Monte Carlo Radiative Transfer
          </Typography>
          <Typography variant="body2" color="inherit">
            v{health?.version || '0.1.0'}
          </Typography>
        </Toolbar>
      </AppBar>

      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{ keepMounted: true }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          minHeight: '100vh',
          bgcolor: 'background.default',
        }}
      >
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
}
