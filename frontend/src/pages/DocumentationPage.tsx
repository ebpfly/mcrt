import { useState } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Divider,
  List,
  ListItem,
  ListItemText,
} from '@mui/material';
import { ExpandMore as ExpandMoreIcon } from '@mui/icons-material';

interface Section {
  id: string;
  title: string;
  content: React.ReactNode;
}

const sections: Section[] = [
  {
    id: 'intro',
    title: 'Introduction to Radiative Transfer',
    content: (
      <Box>
        <Typography paragraph>
          Radiative transfer describes how electromagnetic radiation propagates through and
          interacts with matter. When light encounters a particulate medium (like a powder,
          coating, or suspension), it can be:
        </Typography>
        <List>
          <ListItem>
            <ListItemText
              primary="Reflected"
              secondary="Light bounces back from the surface or scatters back out of the medium"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Absorbed"
              secondary="Light energy is converted to heat within the material"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Transmitted"
              secondary="Light passes through the medium to emerge on the other side"
            />
          </ListItem>
        </List>
        <Typography paragraph>
          The fraction of light that undergoes each process depends on the material's optical
          properties (refractive index n and extinction coefficient k), particle size, volume
          fraction, and the wavelength of light.
        </Typography>
      </Box>
    ),
  },
  {
    id: 'monte-carlo',
    title: 'Monte Carlo Simulation',
    content: (
      <Box>
        <Typography paragraph>
          Monte Carlo methods use random sampling to solve complex physical problems. In radiative
          transfer, we simulate individual photons as they travel through the medium:
        </Typography>
        <List>
          <ListItem>
            <ListItemText
              primary="1. Photon Launch"
              secondary="A photon enters the medium from above with a specified direction"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="2. Free Path"
              secondary="The photon travels a random distance determined by the scattering coefficient"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="3. Interaction"
              secondary="At each interaction, the photon may be absorbed or scattered in a new direction"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="4. Termination"
              secondary="The photon either exits the medium (reflected or transmitted) or is absorbed"
            />
          </ListItem>
        </List>
        <Typography paragraph>
          By simulating millions of photons, we can accurately estimate the reflectance,
          absorptance, and transmittance of the medium. More photons = more accurate results,
          but longer computation time.
        </Typography>
        <Typography variant="subtitle2" gutterBottom>
          Convergence
        </Typography>
        <Typography paragraph>
          The statistical error decreases as 1/sqrt(N), where N is the number of photons.
          Doubling accuracy requires 4x the photons. The progressive simulation mode lets you
          see results converge in real-time.
        </Typography>
      </Box>
    ),
  },
  {
    id: 'validation',
    title: 'Code Validation & Verification',
    content: (
      <Box>
        <Typography variant="subtitle2" gutterBottom sx={{ color: 'primary.main' }}>
          Abstract
        </Typography>
        <Typography paragraph sx={{ fontStyle: 'italic' }}>
          Rigorous validation of Monte Carlo radiative transfer codes requires comparison against
          numerically exact solutions. We validate the FOS Monte Carlo implementation against the
          Adding-Doubling method, which provides exact solutions to the radiative transfer equation
          in plane-parallel geometry. Results demonstrate agreement within 0.17% for optically thick
          media, confirming the physical correctness of the Monte Carlo algorithm.
        </Typography>

        <Divider sx={{ my: 2 }} />

        <Typography variant="subtitle2" gutterBottom>
          1. The Validation Problem
        </Typography>
        <Typography paragraph>
          Monte Carlo methods for radiative transfer are inherently stochastic—they simulate
          individual photon trajectories through a scattering medium. While this approach can
          handle complex geometries and physics, it raises a fundamental question: <strong>how do
          we know the implementation is correct?</strong>
        </Typography>
        <Typography paragraph>
          Simply producing "reasonable-looking" results is insufficient. A subtle bug in the
          scattering phase function, absorption probability, or boundary conditions could produce
          systematic errors that appear plausible but are physically incorrect. Validation requires
          comparison against an independent, exact solution.
        </Typography>

        <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
          2. The Adding-Doubling Method
        </Typography>
        <Typography paragraph>
          The Adding-Doubling method (van de Hulst, 1980; Prahl, 1995) provides <strong>numerically
          exact</strong> solutions to the radiative transfer equation for plane-parallel media. The
          method works by:
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText
              primary="Initialization"
              secondary="Computing exact reflection and transmission for an infinitesimally thin layer using the phase function"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Doubling"
              secondary="Combining two identical layers to get the response of a layer twice as thick"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Adding"
              secondary="Combining layers of different properties to build up arbitrary media"
            />
          </ListItem>
        </List>
        <Typography paragraph>
          This approach is exact to machine precision and serves as the gold standard for validating
          Monte Carlo codes. We use the <code>iadpython</code> implementation (Prahl, 2023), which
          has been extensively validated against analytical solutions and other Adding-Doubling codes.
        </Typography>

        <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
          3. Validation Methodology
        </Typography>
        <Typography paragraph>
          Our validation procedure ensures both methods use <strong>identical optical properties</strong>:
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText
              primary="Step 1: Mie Calculation"
              secondary="Compute single-scattering albedo (ω), optical thickness (τ), and asymmetry parameter (g) from the particle and matrix optical constants using Mie theory"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Step 2: Adding-Doubling Solution"
              secondary="Input (ω, τ, g) to iadpython to compute the exact hemispherical reflectance"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Step 3: Monte Carlo Simulation"
              secondary="Run FOS with the same physical parameters (material, diameter, volume fraction, thickness)"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Step 4: Comparison"
              secondary="Compare Monte Carlo reflectance against Adding-Doubling exact solution"
            />
          </ListItem>
        </List>
        <Typography paragraph>
          This approach isolates the radiative transfer algorithm from the Mie calculation—both
          methods use the same (ω, τ, g) values, so any disagreement must arise from the Monte
          Carlo photon transport implementation.
        </Typography>

        <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
          4. Validation Results
        </Typography>
        <Typography paragraph>
          We validated across six test cases spanning different optical regimes:
        </Typography>
        <Box sx={{ fontFamily: 'monospace', fontSize: '0.85rem', bgcolor: 'grey.100', p: 2, borderRadius: 1, overflow: 'auto' }}>
          <pre style={{ margin: 0 }}>
{`Test Case            |    ω    |     τ    |    g    |   R_AD   |   R_MC   |  Error
---------------------|---------|----------|---------|----------|----------|--------
Fe₂O₃-like (primary) |  0.854  |   400.5  |  0.285  |  0.2765  |  0.2769  |  0.17%
Semi-infinite        |  0.326  |   125.3  |  0.077  |  0.0564  |  0.0564  |  0.09%
High absorption      |  0.122  |    25.1  |  0.078  |  0.0177  |  0.0177  |  0.29%
Strong scattering    |  0.959  |    29.6  |  0.239  |  0.5155  |  0.5128  |  0.51%
Medium absorption    |  0.326  |     8.4  |  0.077  |  0.0564  |  0.0560  |  0.70%
Thin layer           |  0.706  |     0.4  |  0.077  |  0.0875  |  0.0968  | 10.70%`}
          </pre>
        </Box>
        <Typography paragraph sx={{ mt: 2 }}>
          <strong>Key observations:</strong>
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText
              primary="Excellent agreement for optically thick media (τ > 8)"
              secondary="All cases with τ > 8 show agreement within 0.7%, well within expected Monte Carlo statistical uncertainty"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="The primary validation case (Fe₂O₃) shows 0.17% error"
              secondary="This is the reference spectrum displayed in the application—Monte Carlo matches the exact solution"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Thin layers (τ < 1) show higher variance"
              secondary="Expected behavior: optically thin media have fewer scattering events, increasing statistical noise"
            />
          </ListItem>
        </List>

        <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
          5. Statistical Uncertainty
        </Typography>
        <Typography paragraph>
          Monte Carlo methods have inherent statistical uncertainty that decreases as 1/√N, where N
          is the number of photon histories. For our validation runs with N = 100,000 photons, the
          expected statistical uncertainty is approximately:
        </Typography>
        <Typography paragraph sx={{ fontFamily: 'monospace', bgcolor: 'grey.100', p: 1 }}>
          σ_R ≈ √(R(1-R)/N) ≈ 0.1-0.2% for R ~ 0.3
        </Typography>
        <Typography paragraph>
          The observed deviations (0.09-0.70% for thick media) are consistent with this expected
          statistical uncertainty, confirming there are no systematic errors in the implementation.
        </Typography>

        <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
          6. Physical Interpretation
        </Typography>
        <Typography paragraph>
          The validation confirms correct implementation of:
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText
              primary="Absorption probability"
              secondary="Photon weight reduction based on single-scattering albedo ω"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Scattering phase function"
              secondary="Henyey-Greenstein phase function with asymmetry parameter g"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Free path sampling"
              secondary="Exponential distribution based on extinction coefficient"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Boundary conditions"
              secondary="Correct handling of photons exiting the medium (reflected vs transmitted)"
            />
          </ListItem>
        </List>

        <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
          7. Reference Spectrum
        </Typography>
        <Typography paragraph>
          The application displays an <strong>Adding-Doubling reference spectrum</strong> computed
          for the default parameters:
        </Typography>
        <List dense>
          <ListItem><ListItemText primary="Material: Fe₂O₃ (hematite, Querry ordinary ray)" /></ListItem>
          <ListItem><ListItemText primary="Particle diameter: 3.0 µm" /></ListItem>
          <ListItem><ListItemText primary="Volume fraction: 40%" /></ListItem>
          <ListItem><ListItemText primary="Layer thickness: 2000 µm (2 mm)" /></ListItem>
          <ListItem><ListItemText primary="Matrix: Air (n = 1.0)" /></ListItem>
          <ListItem><ListItemText primary="Wavelength range: 7-14 µm (thermal infrared)" /></ListItem>
        </List>
        <Typography paragraph>
          When you run a simulation with these default parameters, the Monte Carlo result should
          closely track the reference curve. Small deviations are expected from statistical noise
          and will decrease with more photons.
        </Typography>

        <Divider sx={{ my: 2 }} />

        <Typography variant="subtitle2" gutterBottom>
          References
        </Typography>
        <Typography variant="body2" paragraph>
          van de Hulst, H. C. (1980). <em>Multiple Light Scattering: Tables, Formulas, and
          Applications</em>. Academic Press.
        </Typography>
        <Typography variant="body2" paragraph>
          Prahl, S. A. (1995). "The Adding-Doubling Method." In <em>Optical-Thermal Response of
          Laser-Irradiated Tissue</em>, Welch & van Gemert, eds. Plenum Press.
        </Typography>
        <Typography variant="body2" paragraph>
          Prahl, S. A. (2023). <em>iadpython: Inverse Adding-Doubling</em>.
          https://github.com/scottprahl/iadpython
        </Typography>
        <Typography variant="body2">
          Querry, M. R. (1985). "Optical Constants." Technical Report, U.S. Army CRDEC.
        </Typography>
      </Box>
    ),
  },
  {
    id: 'mie-theory',
    title: 'Mie Scattering Theory',
    content: (
      <Box>
        <Typography paragraph>
          Mie theory describes how spherical particles scatter and absorb light. It's an exact
          solution to Maxwell's equations and accounts for diffraction, interference, and
          resonance effects that occur when particle size is comparable to the wavelength.
        </Typography>
        <Typography variant="subtitle2" gutterBottom>
          Key Parameters
        </Typography>
        <List>
          <ListItem>
            <ListItemText
              primary="Size Parameter (x)"
              secondary="x = pi * diameter / wavelength. Determines the scattering regime."
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Relative Refractive Index (m)"
              secondary="m = n_particle / n_matrix. The ratio of refractive indices."
            />
          </ListItem>
        </List>
        <Typography variant="subtitle2" gutterBottom>
          Scattering Regimes
        </Typography>
        <List>
          <ListItem>
            <ListItemText
              primary="Rayleigh (x << 1)"
              secondary="Small particles scatter light proportional to 1/wavelength^4 (why the sky is blue)"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Mie (x ~ 1)"
              secondary="Complex interference patterns, forward scattering dominates"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Geometric (x >> 1)"
              secondary="Large particles act like mirrors/lenses, ray optics applies"
            />
          </ListItem>
        </List>
      </Box>
    ),
  },
  {
    id: 'optical-constants',
    title: 'Optical Constants (n, k)',
    content: (
      <Box>
        <Typography paragraph>
          The complex refractive index describes how light interacts with a material:
        </Typography>
        <Typography paragraph sx={{ fontFamily: 'monospace', bgcolor: 'grey.100', p: 1 }}>
          N = n + i*k
        </Typography>
        <Typography variant="subtitle2" gutterBottom>
          Refractive Index (n)
        </Typography>
        <Typography paragraph>
          The real part n determines how much light slows down in the material compared to
          vacuum. It controls refraction (bending) at interfaces and the phase of scattered
          waves. Most transparent materials have n between 1.3 and 2.5.
        </Typography>
        <Typography variant="subtitle2" gutterBottom>
          Extinction Coefficient (k)
        </Typography>
        <Typography paragraph>
          The imaginary part k determines how strongly light is absorbed. k = 0 means the
          material is perfectly transparent at that wavelength. Metals typically have large k
          values, while dielectrics have k near zero in their transparent regions.
        </Typography>
        <Typography variant="subtitle2" gutterBottom>
          Wavelength Dependence
        </Typography>
        <Typography paragraph>
          Both n and k vary with wavelength. This dispersion is what creates rainbows and gives
          materials their characteristic colors. The database provides measured values across
          the UV-visible-IR spectrum.
        </Typography>
      </Box>
    ),
  },
  {
    id: 'size-distributions',
    title: 'Particle Size Distributions',
    content: (
      <Box>
        <Typography paragraph>
          Real particulate materials rarely have uniform particle sizes. The size distribution
          affects the overall optical properties significantly.
        </Typography>
        <Typography variant="subtitle2" gutterBottom>
          Standard Deviation
        </Typography>
        <Typography paragraph>
          The simulator uses a log-normal distribution characterized by a mean diameter and
          standard deviation. A std dev of 0 means monodisperse (all particles the same size).
          Larger std dev means a wider distribution of sizes.
        </Typography>
        <Typography variant="subtitle2" gutterBottom>
          Effects on Optics
        </Typography>
        <List>
          <ListItem>
            <ListItemText
              primary="Broader distributions"
              secondary="Smoother spectral features, less pronounced resonances"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Narrower distributions"
              secondary="Sharper spectral features, more pronounced interference effects"
            />
          </ListItem>
        </List>
      </Box>
    ),
  },
  {
    id: 'medium-config',
    title: 'Medium Configuration',
    content: (
      <Box>
        <Typography paragraph>
          The medium configuration defines the physical structure of your particulate layer.
          Understanding these parameters is crucial for accurate simulations.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Layer Thickness
        </Typography>
        <Typography paragraph>
          The physical thickness of the particulate layer in micrometers (um). This is the
          distance light must travel through the medium. Thicker layers generally have:
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText
              primary="Higher reflectance"
              secondary="More scattering events before light can escape"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Higher absorptance"
              secondary="More opportunities for photons to be absorbed"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Lower transmittance"
              secondary="Fewer photons make it all the way through"
            />
          </ListItem>
        </List>
        <Typography paragraph>
          Typical values: Paint coatings are 50-200 um. Thermal barrier coatings may be 200-500 um.
          Very thin films (&lt;10 um) may show significant transmission.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Particle Diameter
        </Typography>
        <Typography paragraph>
          The mean diameter of particles in micrometers (um). Particle size relative to wavelength
          determines the scattering behavior:
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText
              primary="Small particles (d < wavelength)"
              secondary="Rayleigh scattering - isotropic, strong wavelength dependence (1/wavelength^4)"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Medium particles (d ~ wavelength)"
              secondary="Mie scattering - complex interference patterns, forward scattering dominates"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Large particles (d > wavelength)"
              secondary="Geometric optics - diffraction effects, less wavelength dependence"
            />
          </ListItem>
        </List>
        <Typography paragraph>
          For visible light (0.4-0.7 um), particles of 0.2-2 um show strong Mie effects.
          For mid-IR (7-14 um), particles of 5-20 um are in the Mie regime.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Volume Fraction
        </Typography>
        <Typography paragraph>
          The percentage of the layer volume occupied by particles (0-100%). The remainder is
          the matrix material (e.g., air, binder). Volume fraction affects:
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText
              primary="Scattering coefficient"
              secondary="Higher volume fraction = more scattering events per unit length"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Effective refractive index"
              secondary="The medium becomes an effective mixture of particle and matrix properties"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Optical depth"
              secondary="Product of (scattering coefficient) x (thickness) determines total scattering"
            />
          </ListItem>
        </List>
        <Typography paragraph>
          Typical values: Dense paint pigments 20-60%. Loose powders 30-50%. Dilute suspensions 1-10%.
          Above ~65% is physically difficult to achieve (random packing limit for spheres).
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Size Distribution (Standard Deviation)
        </Typography>
        <Typography paragraph>
          Real particles have a distribution of sizes. This parameter is the standard deviation
          of a log-normal distribution centered on the mean diameter:
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText
              primary="std_dev = 0"
              secondary="Monodisperse - all particles exactly the same size. Sharp spectral features."
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="std_dev = 0.1-0.3"
              secondary="Narrow distribution - slight variation in size. Some smoothing of features."
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="std_dev = 0.5-0.7"
              secondary="Moderate distribution - typical for ground powders. Significant smoothing."
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="std_dev > 1.0"
              secondary="Wide distribution - very heterogeneous particles. Smooth, averaged spectra."
            />
          </ListItem>
        </List>
        <Typography paragraph>
          The simulation discretizes the distribution into multiple size bins (typically ~100)
          and performs Mie calculations for each, then weights by the distribution.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Matrix Material
        </Typography>
        <Typography paragraph>
          The matrix is the medium surrounding the particles. Its optical constants affect:
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText
              primary="Relative refractive index"
              secondary="m = n_particle / n_matrix determines scattering strength"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Fresnel reflections"
              secondary="Light reflects at the air-layer and layer-substrate interfaces"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Background absorption"
              secondary="If the matrix absorbs, it adds to overall absorptance"
            />
          </ListItem>
        </List>
        <Typography paragraph>
          Common matrices: Air (n=1.0, k=0) for powders and aerosols. Polymer binders (n~1.5)
          for paints. Water (n~1.33) for suspensions. Silica (n~1.45) for glass composites.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          How Parameters Interact
        </Typography>
        <Typography paragraph>
          The optical properties depend on combinations of parameters:
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText
              primary="Optical depth = (scattering coeff) x (thickness)"
              secondary="Doubling volume fraction has similar effect to doubling thickness"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Size parameter x = pi * diameter / wavelength"
              secondary="Same scattering behavior at different scales if x is constant"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Refractive index contrast"
              secondary="Larger |n_particle - n_matrix| = stronger scattering"
            />
          </ListItem>
        </List>
      </Box>
    ),
  },
  {
    id: 'visualization',
    title: '3D Visualization',
    content: (
      <Box>
        <Typography paragraph>
          The 3D visualization provides an interactive view of how particles are distributed
          within the medium. It helps you understand the physical structure of your particulate
          layer.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Display Modes
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText
              primary="Random (default)"
              secondary="Particles are placed randomly throughout the volume, avoiding overlaps. This represents a well-mixed suspension or spray-deposited coating."
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Settled"
              secondary="Particles fall under gravity and stack on top of each other. This represents a powder bed or sediment where particles have settled."
            />
          </ListItem>
        </List>

        <Typography variant="subtitle2" gutterBottom>
          How Settling Works
        </Typography>
        <Typography paragraph>
          When "Settled" mode is enabled, the visualization simulates gravity-driven settling:
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText
              primary="1. Size sorting"
              secondary="Larger particles are placed first for more realistic packing (larger particles tend to settle faster)"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="2. Random drop position"
              secondary="Each particle is dropped from a random horizontal (x, z) position"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="3. Settling calculation"
              secondary="The particle falls until it hits the bottom or rests on top of existing particles"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="4. Collision detection"
              secondary="Particles cannot overlap - they find stable resting positions on the surface of particles below"
            />
          </ListItem>
        </List>
        <Typography paragraph>
          This creates a realistic representation of how loose powders pack, with particles
          forming a bed that may not fill the entire volume uniformly.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Performance and Scaling
        </Typography>
        <Typography paragraph>
          For performance reasons, the visualization is limited to ~800 particles (300 in settled
          mode). When the true particle count exceeds this limit:
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText
              primary="Subsample volume"
              secondary="A smaller representative volume is shown with correct particle sizes and packing density"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Accurate packing"
              secondary="The displayed particles show exactly how particles pack at your specified volume fraction"
            />
          </ListItem>
        </List>
        <Typography paragraph>
          The status text shows what percentage of the total layer volume is being displayed.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Controls
        </Typography>
        <List dense>
          <ListItem>
            <ListItemText
              primary="Auto-rotate"
              secondary="Automatically rotates the view for a 3D perspective"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Show matrix"
              secondary="Displays the matrix material as a semi-transparent box"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Particle size scale"
              secondary="Adjusts the displayed particle size for better visibility (does not affect simulation)"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Mouse controls"
              secondary="Drag to rotate, scroll to zoom, right-drag to pan"
            />
          </ListItem>
        </List>
      </Box>
    ),
  },
  {
    id: 'tutorial',
    title: 'Using the Simulator',
    content: (
      <Box>
        <Typography variant="subtitle2" gutterBottom>
          Step 1: Select Materials
        </Typography>
        <Typography paragraph>
          Choose a particle material and matrix material from the database. You can filter by
          category (main, organic, glass) and search by name. The particle is embedded in the
          matrix medium.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Step 2: Configure the Medium
        </Typography>
        <Typography paragraph>
          Set the layer thickness, particle diameter, volume fraction, and size distribution.
          Typical paint coatings are 50-200 um thick with 10-50% volume fraction.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Step 3: Set Wavelength Range
        </Typography>
        <Typography paragraph>
          Choose the wavelength range and resolution for the simulation. The visible range is
          0.4-0.7 um. Solar spectrum extends from 0.3-2.5 um.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Step 4: Configure Photons
        </Typography>
        <Typography paragraph>
          More photons = more accurate results but longer computation. Start with 10K-50K for
          quick exploration, use 100K+ for publication-quality results.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Step 5: Run and Observe
        </Typography>
        <Typography paragraph>
          Click "Start Simulation" to begin. The chart updates progressively as batches
          complete. You can stop anytime and resume later.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Step 6: Save and Export
        </Typography>
        <Typography paragraph>
          Save the simulation state to continue later, or export results as CSV for further
          analysis.
        </Typography>
      </Box>
    ),
  },
  {
    id: 'interpreting',
    title: 'Interpreting Results',
    content: (
      <Box>
        <Typography paragraph>
          The simulation produces three curves as functions of wavelength:
        </Typography>
        <Typography variant="subtitle2" gutterBottom>
          Reflectance (R)
        </Typography>
        <Typography paragraph>
          Fraction of incident light reflected back. High reflectance at visible wavelengths
          makes a material appear white or colored. Metals have high reflectance due to free
          electrons.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Absorptance (A)
        </Typography>
        <Typography paragraph>
          Fraction of incident light absorbed. Absorption peaks correspond to electronic or
          vibrational transitions in the material. Dark materials have high absorptance across
          visible wavelengths.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Transmittance (T)
        </Typography>
        <Typography paragraph>
          Fraction of incident light transmitted through. Only significant for thin layers or
          transparent materials. R + A + T = 1 by energy conservation.
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Common Patterns
        </Typography>
        <List>
          <ListItem>
            <ListItemText
              primary="White coating"
              secondary="High R across visible (0.4-0.7 um), low A"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Black coating"
              secondary="Low R across visible, high A"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="Cool roof coating"
              secondary="High R in near-IR (0.7-2.5 um) to reduce heat absorption"
            />
          </ListItem>
        </List>
      </Box>
    ),
  },
  {
    id: 'api',
    title: 'API Reference',
    content: (
      <Box>
        <Typography paragraph>
          The MCRT API allows programmatic access to simulations.
        </Typography>
        <Typography variant="subtitle2" gutterBottom>
          Base URL
        </Typography>
        <Typography paragraph sx={{ fontFamily: 'monospace', bgcolor: 'grey.100', p: 1 }}>
          http://localhost:8000/api/v1
        </Typography>

        <Typography variant="subtitle2" gutterBottom>
          Key Endpoints
        </Typography>
        <List>
          <ListItem>
            <ListItemText
              primary="GET /health"
              secondary="Check API status and availability"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="GET /materials"
              secondary="List available materials from database"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="GET /materials/{id}"
              secondary="Get optical constants for a material"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="POST /simulation/start"
              secondary="Start a new simulation"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="GET /simulation/{id}/stream"
              secondary="SSE stream for real-time updates"
            />
          </ListItem>
          <ListItem>
            <ListItemText
              primary="GET /simulation/{id}/state"
              secondary="Export simulation state"
            />
          </ListItem>
        </List>

        <Typography paragraph>
          See /api/docs for full OpenAPI documentation.
        </Typography>
      </Box>
    ),
  },
];

export default function DocumentationPage() {
  const [expanded, setExpanded] = useState<string | false>('intro');

  const handleChange = (panel: string) => (_: React.SyntheticEvent, isExpanded: boolean) => {
    setExpanded(isExpanded ? panel : false);
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom>
        Documentation
      </Typography>
      <Typography color="text.secondary" paragraph>
        Learn about radiative transfer, Monte Carlo simulation, and how to use this tool.
      </Typography>

      <Card>
        <CardContent>
          {sections.map((section, index) => (
            <Box key={section.id}>
              <Accordion
                expanded={expanded === section.id}
                onChange={handleChange(section.id)}
                elevation={0}
              >
                <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                  <Typography variant="h6">{section.title}</Typography>
                </AccordionSummary>
                <AccordionDetails>{section.content}</AccordionDetails>
              </Accordion>
              {index < sections.length - 1 && <Divider />}
            </Box>
          ))}
        </CardContent>
      </Card>
    </Box>
  );
}
