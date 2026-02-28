"""Exponential atmosphere with optional per-episode profile perturbations and winds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.constants import RHO0, SCALE_HEIGHT


@dataclass(frozen=True)
class AtmosphereProfile:
    density_scale: float = 1.0
    scale_height_m: float = SCALE_HEIGHT
    wind_bias_x_mps: float = 0.0
    wind_bias_y_mps: float = 0.0
    wind_shear_x_mps: float = 0.0
    wind_shear_y_mps: float = 0.0
    gust_amp_x_mps: float = 0.0
    gust_amp_y_mps: float = 0.0
    gust_freq_x_hz: float = 0.08
    gust_freq_y_hz: float = 0.11
    gust_phase_x_rad: float = 0.0
    gust_phase_y_rad: float = 0.0


def density(z: float, profile: AtmosphereProfile | None = None) -> float:
    """Atmospheric density as a function of altitude z (meters)."""
    z_clamped = float(np.maximum(z, 0.0))
    if profile is None:
        return float(RHO0 * np.exp(-z_clamped / SCALE_HEIGHT))
    scale_height = float(np.maximum(profile.scale_height_m, 1.0))
    return float(profile.density_scale * RHO0 * np.exp(-z_clamped / scale_height))


def wind_velocity(z: float, t: float = 0.0, profile: AtmosphereProfile | None = None) -> np.ndarray:
    """Horizontal wind velocity in world frame at altitude z and time t."""
    if profile is None:
        return np.zeros(3, dtype=float)

    z_clamped = float(np.maximum(z, 0.0))
    z_ratio = np.clip(z_clamped / 60_000.0, 0.0, 1.0)
    shear_shape = float(np.sin(np.pi * z_ratio))
    gust_x = profile.gust_amp_x_mps * np.sin(2.0 * np.pi * profile.gust_freq_x_hz * t + profile.gust_phase_x_rad)
    gust_y = profile.gust_amp_y_mps * np.sin(2.0 * np.pi * profile.gust_freq_y_hz * t + profile.gust_phase_y_rad)

    wx = profile.wind_bias_x_mps + profile.wind_shear_x_mps * shear_shape + gust_x
    wy = profile.wind_bias_y_mps + profile.wind_shear_y_mps * shear_shape + gust_y
    return np.array([wx, wy, 0.0], dtype=float)
