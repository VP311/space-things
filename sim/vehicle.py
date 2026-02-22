"""Vehicle data structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RocketState:
    pos: np.ndarray
    vel: np.ndarray
    quat: np.ndarray
    omega: np.ndarray
    mass: float


@dataclass(frozen=True)
class Control:
    throttle: float
    gimbal_pitch: float
    gimbal_yaw: float


@dataclass(frozen=True)
class VehicleParams:
    max_thrust: float
    isp: float
    dry_mass: float
    prop_mass: float
    area_ref: float
    cd: float
    gimbal_arm: float
    I_body: np.ndarray
    max_gimbal: float
    aero_damp: float
