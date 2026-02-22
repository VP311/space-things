
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.atmosphere import density
from sim.dynamics import step
from sim.vehicle import Control, RocketState, VehicleParams


@dataclass(frozen=True)
class Telemetry:
    t: float
    pos: np.ndarray
    vel: np.ndarray
    quat: np.ndarray
    omega: np.ndarray
    mass: float
    q_dyn: float
    speed: float
    altitude: float
    control: Control


def compute_telemetry(
    state: RocketState,
    control: Control,
    params: VehicleParams,
    t: float = 0.0,
) -> Telemetry:
    del params  # Reserved for future telemetry fields.
    altitude = float(state.pos[2])
    speed = float(np.linalg.norm(state.vel))
    rho = density(altitude)
    q_dyn = 0.5 * rho * speed * speed
    return Telemetry(
        t=float(t),
        pos=state.pos.copy(),
        vel=state.vel.copy(),
        quat=state.quat.copy(),
        omega=state.omega.copy(),
        mass=float(state.mass),
        q_dyn=float(q_dyn),
        speed=speed,
        altitude=altitude,
        control=control,
    )


def simple_schedule(t: float, max_gimbal: float = np.deg2rad(5.0)) -> Control:
    if t < 2.0:
        throttle = 0.0
    elif t < 10.0:
        throttle = 1.0
    else:
        throttle = 0.85

    if t <= 10.0:
        gimbal_pitch = 0.0
    elif t >= 60.0:
        gimbal_pitch = np.deg2rad(-2.0)
    else:
        gimbal_pitch = np.deg2rad(-2.0) * ((t - 10.0) / 50.0)
    gimbal_yaw = 0.0

    gimbal_pitch = float(np.clip(gimbal_pitch, -max_gimbal, max_gimbal))
    gimbal_yaw = float(np.clip(gimbal_yaw, -max_gimbal, max_gimbal))
    return Control(throttle=float(throttle), gimbal_pitch=gimbal_pitch, gimbal_yaw=gimbal_yaw)


def run_episode(params: VehicleParams, dt: float = 0.05, t_final: float = 200.0) -> list[Telemetry]:
    state = RocketState(
        pos=np.array([0.0, 0.0, 0.0], dtype=float),
        vel=np.array([0.0, 0.0, 0.0], dtype=float),
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        omega=np.array([0.0, 0.0, 0.0], dtype=float),
        mass=float(params.dry_mass + params.prop_mass),
    )

    telemetry: list[Telemetry] = []
    t = 0.0
    while t < t_final:
        control = simple_schedule(t, params.max_gimbal)
        state = step(state, control, params, dt)
        t += dt

        if not np.all(np.isfinite(state.pos)):
            raise ValueError("NaN/Inf in state.pos")
        if not np.all(np.isfinite(state.vel)):
            raise ValueError("NaN/Inf in state.vel")
        if not np.all(np.isfinite(state.quat)):
            raise ValueError("NaN/Inf in state.quat")
        if not np.all(np.isfinite(state.omega)):
            raise ValueError("NaN/Inf in state.omega")
        if abs(float(np.linalg.norm(state.quat)) - 1.0) >= 1e-2:
            raise ValueError("Quaternion norm diverged from unit length")

        telemetry.append(compute_telemetry(state, control, params, t=t))

        if state.pos[2] < -1.0:
            break
        if state.mass <= params.dry_mass + 1e-6:
            break

    return telemetry


if __name__ == "__main__":
    demo_params = VehicleParams(
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
    samples = run_episode(demo_params, dt=0.05, t_final=200.0)
    if samples:
        final = samples[-1]
        max_q_dyn = max(s.q_dyn for s in samples)
        print(f"final altitude: {final.altitude:.3f} m")
        print(f"final speed: {final.speed:.3f} m/s")
        print(f"final mass: {final.mass:.3f} kg")
        print(f"max q_dyn: {max_q_dyn:.3f} Pa")
    else:
        print("No telemetry samples produced.")
