import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Box, Sphere, Environment, PerspectiveCamera } from '@react-three/drei';
import { Box as MuiBox, Typography, Paper, Slider, FormControlLabel, Switch } from '@mui/material';
import { useAppSelector } from '../../store';
import * as THREE from 'three';
import { useState } from 'react';

// Particle with position and radius
interface ParticleData {
  position: THREE.Vector3;
  radius: number;
}

// Generate random particle positions and sizes based on volume fraction and size distribution
function generateParticles(
  count: number,
  layerWidth: number,
  layerHeight: number,
  layerDepth: number,
  meanRadius: number,
  radiusStd: number,
  settled: boolean = false,
  seed: number = 42
): ParticleData[] {
  const particles: ParticleData[] = [];
  const random = seededRandom(seed);

  // Box-Muller transform for normal distribution
  const normalRandom = () => {
    const u1 = random();
    const u2 = random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };

  // Pre-generate all particle radii
  const radii: number[] = [];
  for (let i = 0; i < count; i++) {
    let radius = meanRadius;
    if (radiusStd > 0) {
      radius = meanRadius + normalRandom() * radiusStd;
      radius = Math.max(meanRadius * 0.2, Math.min(meanRadius * 3, radius));
    }
    radii.push(radius);
  }

  if (settled) {
    // Settling mode: drop particles from random positions and let them settle
    // Sort by radius (larger particles first for better packing)
    const sortedIndices = radii
      .map((r, i) => ({ r, i }))
      .sort((a, b) => b.r - a.r)
      .map((x) => x.i);

    for (const idx of sortedIndices) {
      const radius = radii[idx];
      const maxAttempts = 50;
      let placed = false;

      for (let attempt = 0; attempt < maxAttempts && !placed; attempt++) {
        // Random x, z position
        const x = (random() - 0.5) * (layerWidth - 2 * radius);
        const z = (random() - 0.5) * (layerDepth - 2 * radius);

        // Start from top and find settling position
        let y = -layerHeight / 2 + radius; // Start at bottom

        // Find the highest point this particle can rest on
        for (const p of particles) {
          const dx = x - p.position.x;
          const dz = z - p.position.z;
          const horizontalDist = Math.sqrt(dx * dx + dz * dz);
          const minDist = radius + p.radius;

          if (horizontalDist < minDist) {
            // This particle is above another - calculate resting height
            const verticalOffset = Math.sqrt(
              Math.max(0, minDist * minDist - horizontalDist * horizontalDist)
            );
            const restingY = p.position.y + verticalOffset;
            y = Math.max(y, restingY);
          }
        }

        // Check if position is valid (within bounds and no overlap)
        if (y + radius <= layerHeight / 2) {
          const newPos = new THREE.Vector3(x, y, z);
          let collision = false;

          for (const p of particles) {
            if (newPos.distanceTo(p.position) < (radius + p.radius) * 0.99) {
              collision = true;
              break;
            }
          }

          if (!collision) {
            particles.push({ position: newPos, radius });
            placed = true;
          }
        }
      }
    }
  } else {
    // Random placement mode - try hard to place all particles
    const maxAttempts = count * 200;
    let attempts = 0;
    let radiusIdx = 0;

    while (particles.length < count && attempts < maxAttempts) {
      attempts++;

      const radius = radii[radiusIdx % radii.length];

      const x = (random() - 0.5) * (layerWidth - 2 * radius);
      const y = (random() - 0.5) * (layerHeight - 2 * radius);
      const z = (random() - 0.5) * (layerDepth - 2 * radius);

      const newPos = new THREE.Vector3(x, y, z);

      // Check for collisions - use tight tolerance (1.01x) for dense packing
      let collision = false;
      for (const p of particles) {
        if (newPos.distanceTo(p.position) < (radius + p.radius) * 1.01) {
          collision = true;
          break;
        }
      }

      if (!collision) {
        particles.push({ position: newPos, radius });
        radiusIdx++;
      }
    }
  }

  return particles;
}

// Seeded random number generator for reproducible particle positions
function seededRandom(seed: number): () => number {
  let state = seed;
  return () => {
    state = (state * 1103515245 + 12345) & 0x7fffffff;
    return state / 0x7fffffff;
  };
}

// Calculate number of particles from volume fraction
function calculateParticleCount(
  volumeFraction: number, // percentage (0-100)
  layerVolume: number,
  particleRadius: number
): number {
  const particleVolume = (4 / 3) * Math.PI * Math.pow(particleRadius, 3);
  const totalParticleVolume = (volumeFraction / 100) * layerVolume;
  return Math.round(totalParticleVolume / particleVolume);
}

// Individual particle component with optional animation
function Particle({
  position,
  radius,
  color,
  opacity = 0.9
}: {
  position: THREE.Vector3;
  radius: number;
  color: string;
  opacity?: number;
}) {
  return (
    <Sphere args={[radius, 16, 16]} position={position}>
      <meshStandardMaterial
        color={color}
        transparent
        opacity={opacity}
        roughness={0.3}
        metalness={0.1}
      />
    </Sphere>
  );
}

// Layer box component
function LayerBox({
  width,
  height,
  depth,
  color,
  opacity = 0.15
}: {
  width: number;
  height: number;
  depth: number;
  color: string;
  opacity?: number;
}) {
  return (
    <Box args={[width, height, depth]}>
      <meshStandardMaterial
        color={color}
        transparent
        opacity={opacity}
        side={THREE.DoubleSide}
        depthWrite={false}
      />
    </Box>
  );
}

// Wireframe box for layer boundary
function LayerWireframe({
  width,
  height,
  depth
}: {
  width: number;
  height: number;
  depth: number;
}) {
  return (
    <lineSegments>
      <edgesGeometry args={[new THREE.BoxGeometry(width, height, depth)]} />
      <lineBasicMaterial color="#666666" />
    </lineSegments>
  );
}

// Main 3D scene component
function Scene({
  layerThickness,
  particleDiameter,
  particleDiameterStd,
  volumeFraction,
  particleColor,
  matrixColor,
  showMatrix,
  autoRotate,
  settled
}: {
  layerThickness: number;
  particleDiameter: number;
  particleDiameterStd: number;
  volumeFraction: number;
  particleColor: string;
  matrixColor: string;
  showMatrix: boolean;
  autoRotate: boolean;
  settled: boolean;
}) {
  const groupRef = useRef<THREE.Group>(null);

  // Auto-rotate animation
  useFrame((_, delta) => {
    if (autoRotate && groupRef.current) {
      groupRef.current.rotation.y += delta * 0.3;
    }
  });

  // Scale everything for visualization (micrometers are too small)
  // Use a representative scale where the layer is ~4 units
  const scale = 4 / Math.max(layerThickness, 1);

  const scaledThickness = layerThickness * scale;
  const scaledMeanRadius = (particleDiameter / 2) * scale;
  const scaledRadiusStd = (particleDiameterStd / 2) * scale;

  // Layer dimensions (square cross-section)
  const layerWidth = scaledThickness;
  const layerHeight = scaledThickness;
  const layerDepth = scaledThickness;

  // Calculate particles with positions and sizes
  // Strategy: calculate display volume first, then generate particles directly in it
  const { particles, displayVolume } = useMemo(() => {
    const fullVolume = layerWidth * layerHeight * layerDepth;
    const maxParticles = settled ? 300 : 800;

    // Calculate how many particles we'd need at actual size for full volume
    const trueCount = calculateParticleCount(volumeFraction, fullVolume, scaledMeanRadius);

    // Determine display volume and target count
    let displayWidth, displayHeight, displayDepth;
    let targetCount;

    if (trueCount > maxParticles) {
      // Calculate volume that maxParticles would fill at target volume fraction
      const singleParticleVolume = (4 / 3) * Math.PI * Math.pow(scaledMeanRadius, 3);
      const maxParticleVolume = maxParticles * singleParticleVolume;
      const targetVolume = maxParticleVolume / (volumeFraction / 100);

      // Scale the display dimensions uniformly
      const linearScale = Math.pow(targetVolume / fullVolume, 1 / 3);
      displayWidth = layerWidth * linearScale;
      displayHeight = layerHeight * linearScale;
      displayDepth = layerDepth * linearScale;
      targetCount = maxParticles;
    } else {
      // Use full volume
      displayWidth = layerWidth;
      displayHeight = layerHeight;
      displayDepth = layerDepth;
      targetCount = trueCount;
    }

    // Generate particles directly in the display volume with proper collision detection
    const generatedParticles = generateParticles(
      Math.max(1, targetCount),
      displayWidth,
      displayHeight,
      displayDepth,
      scaledMeanRadius,
      scaledRadiusStd,
      settled
    );

    return {
      particles: generatedParticles,
      displayVolume: {
        width: displayWidth,
        height: displayHeight,
        depth: displayDepth
      }
    };
  }, [layerWidth, layerHeight, layerDepth, scaledMeanRadius, scaledRadiusStd, volumeFraction, settled]);

  return (
    <group ref={groupRef}>
      {/* Matrix material (semi-transparent box) - uses display volume size */}
      {showMatrix && (
        <LayerBox
          width={displayVolume.width}
          height={displayVolume.height}
          depth={displayVolume.depth}
          color={matrixColor}
          opacity={0.2}
        />
      )}

      {/* Layer boundary wireframe - uses display volume size */}
      <LayerWireframe
        width={displayVolume.width}
        height={displayVolume.height}
        depth={displayVolume.depth}
      />

      {/* Particles */}
      {particles.map((p, i) => (
        <Particle
          key={i}
          position={p.position}
          radius={p.radius}
          color={particleColor}
        />
      ))}
    </group>
  );
}

// Main exported component
export default function MediumVisualization3D() {
  const layers = useAppSelector((state) => state.medium.layers);

  // Get first layer config (for now, visualize single layer)
  const layer = layers[0];
  const particle = layer?.particles[0];

  // Visualization controls
  const [showMatrix, setShowMatrix] = useState(true);
  const [autoRotate, setAutoRotate] = useState(true);
  const [particleScale, setParticleScale] = useState(1);
  const [settled, setSettled] = useState(false);

  // Default values if no config
  const layerThickness = layer?.thickness_um ?? 100;
  const baseDiameter = particle?.diameter_um ?? 1;
  const particleDiameter = baseDiameter * particleScale;
  // std_dev is relative (0-1), convert to absolute diameter std
  const particleDiameterStd = (particle?.std_dev ?? 0) * baseDiameter * particleScale;
  const volumeFraction = particle?.volume_fraction ?? 10;

  // Colors
  const particleColor = '#4fc3f7'; // Light blue
  const matrixColor = '#e0e0e0'; // Light gray

  // For display purposes, we'll show the actual count from the Scene component
  // This is a simplified estimate - the Scene component calculates the actual value
  const displayInfo = useMemo(() => {
    const scale = 4 / Math.max(layerThickness, 1);
    const scaledMeanRadius = (particleDiameter / 2) * scale;
    const layerVolume = 4 * 4 * 4;
    const maxParticles = settled ? 300 : 800;
    const trueCount = calculateParticleCount(volumeFraction, layerVolume, scaledMeanRadius);
    const targetCount = Math.min(trueCount, maxParticles);

    return {
      targetCount: Math.max(1, targetCount),
      isSubsample: trueCount > maxParticles
    };
  }, [layerThickness, particleDiameter, volumeFraction, settled]);

  return (
    <Paper elevation={2} sx={{ p: 2, height: '100%', minHeight: 400 }}>
      <Typography variant="h6" gutterBottom>
        Medium Visualization
      </Typography>

      {/* Controls */}
      <MuiBox sx={{ mb: 2 }}>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          Layer: {layerThickness} µm | Particle: {particleDiameter.toFixed(2)} µm |
          Volume fraction: {volumeFraction}%
        </Typography>
        <Typography variant="caption" color="text.secondary" display="block" gutterBottom>
          {displayInfo.isSubsample
            ? `(Representative subsample with ~${displayInfo.targetCount} particles)`
            : `(~${displayInfo.targetCount} particles)`
          }
        </Typography>

        <MuiBox sx={{ display: 'flex', gap: 2, alignItems: 'center', flexWrap: 'wrap' }}>
          <FormControlLabel
            control={
              <Switch
                checked={autoRotate}
                onChange={(e) => setAutoRotate(e.target.checked)}
                size="small"
              />
            }
            label="Auto-rotate"
          />
          <FormControlLabel
            control={
              <Switch
                checked={showMatrix}
                onChange={(e) => setShowMatrix(e.target.checked)}
                size="small"
              />
            }
            label="Show matrix"
          />
          <FormControlLabel
            control={
              <Switch
                checked={settled}
                onChange={(e) => setSettled(e.target.checked)}
                size="small"
              />
            }
            label="Settled"
          />
        </MuiBox>

        <MuiBox sx={{ mt: 1 }}>
          <Typography variant="caption" color="text.secondary">
            Particle size scale: {particleScale.toFixed(1)}x
          </Typography>
          <Slider
            value={particleScale}
            onChange={(_, v) => setParticleScale(v as number)}
            min={0.5}
            max={3}
            step={0.1}
            size="small"
          />
        </MuiBox>
      </MuiBox>

      {/* 3D Canvas */}
      <MuiBox sx={{ height: 300, bgcolor: '#1a1a2e', borderRadius: 1, overflow: 'hidden' }}>
        <Canvas>
          <PerspectiveCamera makeDefault position={[6, 4, 6]} />
          <OrbitControls
            enablePan={true}
            enableZoom={true}
            enableRotate={true}
          />

          {/* Lighting */}
          <ambientLight intensity={0.4} />
          <directionalLight position={[10, 10, 5]} intensity={1} />
          <directionalLight position={[-10, -10, -5]} intensity={0.3} />

          {/* Scene */}
          <Scene
            layerThickness={layerThickness}
            particleDiameter={particleDiameter}
            particleDiameterStd={particleDiameterStd}
            volumeFraction={volumeFraction}
            particleColor={particleColor}
            matrixColor={matrixColor}
            showMatrix={showMatrix}
            autoRotate={autoRotate}
            settled={settled}
          />

          {/* Environment for reflections */}
          <Environment preset="studio" />
        </Canvas>
      </MuiBox>
    </Paper>
  );
}
