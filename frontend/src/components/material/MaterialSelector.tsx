import { useState, useEffect } from 'react';
import {
  Box,
  TextField,
  Autocomplete,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Typography,
  Divider,
  Chip,
} from '@mui/material';
import { useAppDispatch, useAppSelector, mediumActions, uiActions } from '../../store';
import api from '../../services/api';
import { MaterialInfo } from '../../types';

// Default materials to load on startup
const DEFAULT_PARTICLE_MATERIAL = 'main/Fe2O3/Querry-o'; // Fe2O3 (0.21-91 µm) - matches Adding-Doubling validation reference

// Air material - synthetic n=1.0, k=0 across 0.3-16 µm range
const AIR_MATERIAL = {
  wavelength_um: [0.3, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0],
  n: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
  k: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
};

export default function MaterialSelector() {
  const dispatch = useAppDispatch();
  const materials = useAppSelector((state) => state.health.availableMaterials);
  const shelves = useAppSelector((state) => state.health.availableShelves);
  const selectedIds = useAppSelector((state) => state.medium.selectedMaterialIds);
  const particleMaterials = useAppSelector((state) => state.medium.particleMaterials);
  const matrixMaterials = useAppSelector((state) => state.medium.matrixMaterials);

  const [selectedShelf, setSelectedShelf] = useState<string>('');
  const [searchText, setSearchText] = useState('');
  const [filteredMaterials, setFilteredMaterials] = useState<MaterialInfo[]>([]);
  const [defaultsLoaded, setDefaultsLoaded] = useState(false);

  // Load default materials on startup
  useEffect(() => {
    const loadDefaults = async () => {
      if (defaultsLoaded) return;
      if (materials.length === 0) return; // Wait for materials to load
      if (Object.keys(particleMaterials).length > 0 || Object.keys(matrixMaterials).length > 0) {
        setDefaultsLoaded(true);
        return; // Already have materials loaded
      }

      try {
        // Load particle material (quartz)
        const particleOc = await api.getMaterial(DEFAULT_PARTICLE_MATERIAL);
        dispatch(mediumActions.setParticleMaterial({ id: 1, material: particleOc }));
        dispatch(mediumActions.setSelectedMaterialId({ type: 'particle', id: 1, materialId: DEFAULT_PARTICLE_MATERIAL }));

        // Use synthetic air for matrix (n=1.0, k=0)
        dispatch(mediumActions.setMatrixMaterial({ id: 1, material: AIR_MATERIAL }));
        dispatch(mediumActions.setSelectedMaterialId({ type: 'matrix', id: 1, materialId: 'air' }));

        setDefaultsLoaded(true);
      } catch (error) {
        console.error('Failed to load default materials:', error);
        setDefaultsLoaded(true); // Don't retry on error
      }
    };

    loadDefaults();
  }, [dispatch, materials, particleMaterials, matrixMaterials, defaultsLoaded]);

  useEffect(() => {
    let filtered = materials;
    if (selectedShelf) {
      filtered = filtered.filter((m) => m.shelf === selectedShelf);
    }
    if (searchText) {
      const search = searchText.toLowerCase();
      filtered = filtered.filter(
        (m) => m.name.toLowerCase().includes(search) || m.material_id.toLowerCase().includes(search)
      );
    }
    setFilteredMaterials(filtered.slice(0, 100));
  }, [materials, selectedShelf, searchText]);

  const handleSelectMaterial = async (
    type: 'particle' | 'matrix',
    id: number,
    material: MaterialInfo | null
  ) => {
    if (!material) return;

    try {
      const oc = await api.getMaterial(material.material_id);
      if (type === 'particle') {
        dispatch(mediumActions.setParticleMaterial({ id, material: oc }));
      } else {
        dispatch(mediumActions.setMatrixMaterial({ id, material: oc }));
      }
      dispatch(
        mediumActions.setSelectedMaterialId({ type, id, materialId: material.material_id })
      );
    } catch (error) {
      dispatch(uiActions.addNotification({ type: 'error', message: 'Failed to load material' }));
    }
  };

  const getSelectedMaterial = (type: 'particle' | 'matrix', id: number): MaterialInfo | null => {
    const materialId = selectedIds[type][id];
    if (!materialId) return null;
    return materials.find((m) => m.material_id === materialId) || null;
  };

  return (
    <Box>
      {/* Filters */}
      <Box sx={{ mb: 2 }}>
        <FormControl size="small" fullWidth sx={{ mb: 1 }}>
          <InputLabel>Category</InputLabel>
          <Select
            value={selectedShelf}
            label="Category"
            onChange={(e) => setSelectedShelf(e.target.value)}
          >
            <MenuItem value="">All</MenuItem>
            {shelves.map((shelf) => (
              <MenuItem key={shelf} value={shelf}>
                {shelf}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
        <TextField
          size="small"
          fullWidth
          placeholder="Search materials..."
          value={searchText}
          onChange={(e) => setSearchText(e.target.value)}
        />
      </Box>

      <Divider sx={{ my: 2 }} />

      {/* Particle Material */}
      <Typography variant="subtitle2" gutterBottom>
        Particle Material
      </Typography>
      <Autocomplete
        size="small"
        options={filteredMaterials}
        getOptionLabel={(option) => option.name.replace(/<[^>]*>/g, '')}
        value={getSelectedMaterial('particle', 1)}
        onChange={(_, value) => handleSelectMaterial('particle', 1, value)}
        renderInput={(params) => <TextField {...params} placeholder="Select particle material" />}
        renderOption={(props, option) => (
          <li {...props} key={option.material_id}>
            <Box>
              <Typography variant="body2" dangerouslySetInnerHTML={{ __html: option.name }} />
              <Typography variant="caption" color="text.secondary">
                {option.shelf}/{option.book}
              </Typography>
            </Box>
          </li>
        )}
        sx={{ mb: 2 }}
      />
      {selectedIds.particle[1] && (
        <Chip
          size="small"
          label={selectedIds.particle[1]}
          onDelete={() => {
            dispatch(mediumActions.setSelectedMaterialId({ type: 'particle', id: 1, materialId: '' }));
          }}
          sx={{ mb: 2 }}
        />
      )}

      <Divider sx={{ my: 2 }} />

      {/* Matrix Material */}
      <Typography variant="subtitle2" gutterBottom>
        Matrix Material
      </Typography>
      <Autocomplete
        size="small"
        options={filteredMaterials}
        getOptionLabel={(option) => option.name.replace(/<[^>]*>/g, '')}
        value={getSelectedMaterial('matrix', 1)}
        onChange={(_, value) => handleSelectMaterial('matrix', 1, value)}
        renderInput={(params) => <TextField {...params} placeholder="Select matrix material" />}
        renderOption={(props, option) => (
          <li {...props} key={option.material_id}>
            <Box>
              <Typography variant="body2" dangerouslySetInnerHTML={{ __html: option.name }} />
              <Typography variant="caption" color="text.secondary">
                {option.shelf}/{option.book}
              </Typography>
            </Box>
          </li>
        )}
      />
      {selectedIds.matrix[1] && (
        <Chip
          size="small"
          label={selectedIds.matrix[1]}
          onDelete={() => {
            dispatch(mediumActions.setSelectedMaterialId({ type: 'matrix', id: 1, materialId: '' }));
          }}
          sx={{ mt: 1 }}
        />
      )}
    </Box>
  );
}
