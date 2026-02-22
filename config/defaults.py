"""Default simulator and RL configuration values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.vehicle import VehicleParams


@dataclass(frozen=True)
class EnvConfig:
    dt: float = 0.05
    t_final: float = 250.0
    q_limit: float = 35_000.0
    target_apogee: float = 120_000.0


def default_vehicle_params() -> VehicleParams:
    return VehicleParams(
        max_thrust=1.5e6,
        isp=300.0,
        dry_mass=2.0e4,
        prop_mass=1.0e4,
        area_ref=10.0,
        cd=0.5,
        gimbal_arm=2.0,
        I_body=np.array([8e5, 8e5, 1e5], dtype=float),
        max_gimbal=np.deg2rad(5.0),
        aero_damp=1e4,
    )
