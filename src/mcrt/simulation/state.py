"""State persistence for simulation sessions."""

import json
from datetime import datetime
from pathlib import Path
from typing import BinaryIO
import numpy as np

from mcrt.simulation.manager import SimulationSession


class StatePersistence:
    """Save and restore simulation state."""

    VERSION = "1.0"

    @classmethod
    def save_json(cls, session: SimulationSession, filepath: str | Path) -> Path:
        """Save session state to JSON file.

        Args:
            session: Session to save
            filepath: Output file path

        Returns:
            Path to saved file
        """
        filepath = Path(filepath)

        state = {
            "version": cls.VERSION,
            "saved_at": datetime.now().isoformat(),
            "session": session.to_dict(),
        }

        with open(filepath, "w") as f:
            json.dump(state, f, indent=2)

        return filepath

    @classmethod
    def load_json(cls, filepath: str | Path) -> SimulationSession:
        """Load session state from JSON file.

        Args:
            filepath: Path to state file

        Returns:
            Restored SimulationSession
        """
        filepath = Path(filepath)

        with open(filepath) as f:
            state = json.load(f)

        version = state.get("version", "0.0")
        if version != cls.VERSION:
            # Could add migration logic here
            pass

        return SimulationSession.from_dict(state["session"])

    @classmethod
    def save_compact(cls, session: SimulationSession, filepath: str | Path) -> Path:
        """Save session state in compact binary format (npz + json).

        This is more efficient for large result arrays.

        Args:
            session: Session to save
            filepath: Output file path (will add .npz and .json)

        Returns:
            Path to main metadata file
        """
        filepath = Path(filepath)
        base = filepath.with_suffix("")

        # Save arrays to npz
        arrays = {}
        if session.current_result:
            arrays["wavelength_um"] = session.current_result.wavelength_um
            arrays["reflectance"] = session.current_result.reflectance
            arrays["absorptance"] = session.current_result.absorptance
            arrays["transmittance"] = session.current_result.transmittance

        # Save material arrays
        for i, (k, mat) in enumerate(session.particle_materials.items()):
            arrays[f"particle_{k}_wavelength"] = mat.wavelength_um
            arrays[f"particle_{k}_n"] = mat.n
            arrays[f"particle_{k}_k"] = mat.k

        for i, (k, mat) in enumerate(session.matrix_materials.items()):
            arrays[f"matrix_{k}_wavelength"] = mat.wavelength_um
            arrays[f"matrix_{k}_n"] = mat.n
            arrays[f"matrix_{k}_k"] = mat.k

        np.savez_compressed(f"{base}.npz", **arrays)

        # Save metadata to json (without large arrays)
        session_dict = session.to_dict()

        # Remove array data from json (they're in npz)
        if session_dict.get("current_result"):
            for key in ["wavelength_um", "reflectance", "absorptance", "transmittance"]:
                session_dict["current_result"][key] = []

        for k in session_dict["particle_materials"]:
            for key in ["wavelength_um", "n", "k"]:
                session_dict["particle_materials"][k][key] = []

        for k in session_dict["matrix_materials"]:
            for key in ["wavelength_um", "n", "k"]:
                session_dict["matrix_materials"][k][key] = []

        metadata = {
            "version": cls.VERSION,
            "saved_at": datetime.now().isoformat(),
            "format": "compact",
            "session": session_dict,
        }

        json_path = Path(f"{base}.json")
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return json_path

    @classmethod
    def load_compact(cls, filepath: str | Path) -> SimulationSession:
        """Load session from compact format.

        Args:
            filepath: Path to .json metadata file

        Returns:
            Restored SimulationSession
        """
        filepath = Path(filepath)
        base = filepath.with_suffix("")

        # Load metadata
        with open(filepath) as f:
            metadata = json.load(f)

        session_dict = metadata["session"]

        # Load arrays
        npz_path = Path(f"{base}.npz")
        with np.load(npz_path) as data:
            # Restore result arrays
            if session_dict.get("current_result"):
                session_dict["current_result"]["wavelength_um"] = data["wavelength_um"].tolist()
                session_dict["current_result"]["reflectance"] = data["reflectance"].tolist()
                session_dict["current_result"]["absorptance"] = data["absorptance"].tolist()
                session_dict["current_result"]["transmittance"] = data["transmittance"].tolist()

            # Restore material arrays
            for k in session_dict["particle_materials"]:
                session_dict["particle_materials"][k]["wavelength_um"] = data[f"particle_{k}_wavelength"].tolist()
                session_dict["particle_materials"][k]["n"] = data[f"particle_{k}_n"].tolist()
                session_dict["particle_materials"][k]["k"] = data[f"particle_{k}_k"].tolist()

            for k in session_dict["matrix_materials"]:
                session_dict["matrix_materials"][k]["wavelength_um"] = data[f"matrix_{k}_wavelength"].tolist()
                session_dict["matrix_materials"][k]["n"] = data[f"matrix_{k}_n"].tolist()
                session_dict["matrix_materials"][k]["k"] = data[f"matrix_{k}_k"].tolist()

        return SimulationSession.from_dict(session_dict)

    @classmethod
    def export_results_csv(
        cls,
        session: SimulationSession,
        filepath: str | Path,
    ) -> Path:
        """Export results to CSV.

        Args:
            session: Session with results
            filepath: Output CSV path

        Returns:
            Path to CSV file
        """
        filepath = Path(filepath)

        if session.current_result is None:
            raise ValueError("No results to export")

        result = session.current_result
        header = "wavelength_um,reflectance,absorptance,transmittance\n"

        with open(filepath, "w") as f:
            f.write(header)
            for i in range(len(result.wavelength_um)):
                f.write(
                    f"{result.wavelength_um[i]:.6f},"
                    f"{result.reflectance[i]:.6f},"
                    f"{result.absorptance[i]:.6f},"
                    f"{result.transmittance[i]:.6f}\n"
                )

        return filepath

    @classmethod
    def get_state_summary(cls, filepath: str | Path) -> dict:
        """Get summary of saved state without fully loading.

        Args:
            filepath: Path to state file (.json)

        Returns:
            Summary dict with key metadata
        """
        filepath = Path(filepath)

        with open(filepath) as f:
            data = json.load(f)

        session = data.get("session", {})

        return {
            "version": data.get("version"),
            "saved_at": data.get("saved_at"),
            "session_id": session.get("session_id"),
            "status": session.get("status"),
            "batches_completed": session.get("batches_completed"),
            "total_batches": session.get("total_batches"),
            "photons_completed": session.get("photons_completed"),
            "photons_target": session.get("photons_target"),
            "created_at": session.get("created_at"),
        }
