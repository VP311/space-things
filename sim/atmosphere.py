"""Simple exponential atmosphere model."""

from __future__ import annotations

import numpy as np

from sim.constants import RHO0, SCALE_HEIGHT


def density(z: float) -> float:
    """Atmospheric density as a function of altitude z (meters)."""
    z_clamped = float(np.maximum(z, 0.0))
    return float(RHO0 * np.exp(-z_clamped / SCALE_HEIGHT))
