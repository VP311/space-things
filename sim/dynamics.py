"""Core rocket rigid-body dynamics."""

from __future__ import annotations

import numpy as np

from sim.atmosphere import AtmosphereProfile, density, wind_velocity
from sim.constants import G0
from sim.math3d import quat_mul, quat_normalize, quat_rotate
from sim.vehicle import Control, RocketState, VehicleParams


def gimbal_direction_body(gimbal_pitch: float, gimbal_yaw: float) -> np.ndarray:
    """Thrust unit vector in body frame after pitch (about +Y) then yaw (about +Z)."""
    cp = float(np.cos(gimbal_pitch))
    sp = float(np.sin(gimbal_pitch))
    cy = float(np.cos(gimbal_yaw))
    sy = float(np.sin(gimbal_yaw))

    # Rotate body +X by Ry(pitch) then Rz(yaw): d = Rz * Ry * [1, 0, 0]
    return np.array([cy * cp, sy * cp, -sp], dtype=float)


def aero_forces(
    state: RocketState,
    control: Control,
    params: VehicleParams,
    *,
    t: float = 0.0,
    atmosphere: AtmosphereProfile | None = None,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Return (thrust_world, drag_world, q_dyn, air_velocity_world)."""
    throttle = float(np.clip(control.throttle, 0.0, 1.0))
    gimbal_pitch = float(np.clip(control.gimbal_pitch, -params.max_gimbal, params.max_gimbal))
    gimbal_yaw = float(np.clip(control.gimbal_yaw, -params.max_gimbal, params.max_gimbal))
    thrust_mag = throttle * params.max_thrust

    thrust_dir_body = gimbal_direction_body(gimbal_pitch, gimbal_yaw)
    thrust_body = thrust_mag * thrust_dir_body
    thrust_world = quat_rotate(state.quat, thrust_body)

    rho = density(float(state.pos[2]), profile=atmosphere)
    w_world = wind_velocity(float(state.pos[2]), t=t, profile=atmosphere)
    v_air_world = state.vel - w_world
    v_air = float(np.linalg.norm(v_air_world))
    q_dyn = 0.5 * rho * v_air * v_air
    if v_air > 0.0:
        v_hat = v_air_world / v_air
        drag_world = -(q_dyn * params.cd * params.area_ref) * v_hat
    else:
        drag_world = np.zeros(3, dtype=float)
    return thrust_world, drag_world, float(q_dyn), v_air_world


def step(
    state: RocketState,
    control: Control,
    params: VehicleParams,
    dt: float,
    *,
    t: float = 0.0,
    atmosphere: AtmosphereProfile | None = None,
) -> RocketState:
    """Advance rocket state by one fixed step."""
    throttle = float(np.clip(control.throttle, 0.0, 1.0))
    gimbal_pitch = float(np.clip(control.gimbal_pitch, -params.max_gimbal, params.max_gimbal))
    gimbal_yaw = float(np.clip(control.gimbal_yaw, -params.max_gimbal, params.max_gimbal))
    mass = float(np.maximum(state.mass, params.dry_mass))

    thrust_world, drag_world, _, _ = aero_forces(
        state,
        control,
        params,
        t=t,
        atmosphere=atmosphere,
    )
    thrust_mag = throttle * params.max_thrust
    thrust_body = thrust_mag * gimbal_direction_body(gimbal_pitch, gimbal_yaw)

    gravity_world = np.array([0.0, 0.0, -G0], dtype=float)
    accel_world = gravity_world + (thrust_world + drag_world) / mass

    # Longitudinal axis is +X in this model; the engine is aft along -X.
    # With zero gimbal, thrust is colinear with this lever arm so net thrust torque is zero.
    r_body = np.array([-params.gimbal_arm, 0.0, 0.0], dtype=float)
    torque_body = np.cross(r_body, thrust_body) - params.aero_damp * state.omega
    alpha_body = torque_body / params.I_body

    vel = state.vel + accel_world * dt
    pos = state.pos + vel * dt
    omega = state.omega + alpha_body * dt

    q_omega = np.array([0.0, omega[0], omega[1], omega[2]], dtype=float)
    quat = state.quat + 0.5 * quat_mul(state.quat, q_omega) * dt
    quat = quat_normalize(quat)

    mdot = thrust_mag / (params.isp * G0) if params.isp > 0.0 else 0.0
    mass_next = mass - mdot * dt
    mass_next = float(np.maximum(mass_next, params.dry_mass))

    return RocketState(pos=pos, vel=vel, quat=quat, omega=omega, mass=mass_next)
