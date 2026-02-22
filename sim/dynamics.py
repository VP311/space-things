"""Core rocket rigid-body dynamics."""

from __future__ import annotations

import numpy as np

from sim.atmosphere import density
from sim.constants import G0
from sim.math3d import quat_mul, quat_normalize, quat_rotate
from sim.vehicle import Control, RocketState, VehicleParams


def _gimbal_direction_body(gimbal_pitch: float, gimbal_yaw: float) -> np.ndarray:
    """Thrust unit vector in body frame after pitch (about +Y) then yaw (about +Z)."""
    cp = float(np.cos(gimbal_pitch))
    sp = float(np.sin(gimbal_pitch))
    cy = float(np.cos(gimbal_yaw))
    sy = float(np.sin(gimbal_yaw))

    # Rotate body +X by Ry(pitch) then Rz(yaw): d = Rz * Ry * [1, 0, 0]
    return np.array([cy * cp, sy * cp, -sp], dtype=float)


def step(state: RocketState, control: Control, params: VehicleParams, dt: float) -> RocketState:
    """Advance rocket state by one fixed step."""
    throttle = float(np.clip(control.throttle, 0.0, 1.0))
    gimbal_pitch = float(np.clip(control.gimbal_pitch, -params.max_gimbal, params.max_gimbal))
    gimbal_yaw = float(np.clip(control.gimbal_yaw, -params.max_gimbal, params.max_gimbal))
    mass = float(np.maximum(state.mass, params.dry_mass))

    thrust_mag = throttle * params.max_thrust
    thrust_dir_body = _gimbal_direction_body(gimbal_pitch, gimbal_yaw)
    thrust_body = thrust_mag * thrust_dir_body
    thrust_world = quat_rotate(state.quat, thrust_body)

    rho = density(float(state.pos[2]))
    speed = float(np.linalg.norm(state.vel))
    if speed > 0.0:
        v_hat = state.vel / speed
        drag_mag = 0.5 * rho * speed * speed * params.cd * params.area_ref
        drag_world = -drag_mag * v_hat
    else:
        drag_world = np.zeros(3, dtype=float)

    gravity_world = np.array([0.0, 0.0, -G0], dtype=float)
    accel_world = gravity_world + (thrust_world + drag_world) / mass

    r_body = np.array([0.0, 0.0, -params.gimbal_arm], dtype=float)
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
