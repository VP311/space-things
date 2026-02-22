"""Gymnasium environment for rocket ascent with burn + coast episode logic."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.atmosphere import density
from sim.dynamics import step
from sim.runner import Telemetry, compute_telemetry
from sim.vehicle import Control, RocketState, VehicleParams


@dataclass(frozen=True)
class EpisodeScalars:
    altitude: float
    speed: float
    q_dyn: float
    vx: float
    vz: float


class RocketAscentEnv(gym.Env[np.ndarray, np.ndarray]):
    """Minimal RL environment for ascent control with max-Q awareness."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: VehicleParams,
        dt: float = 0.05,
        t_final: float = 250.0,
        q_limit: float = 35_000.0,
        target_apogee: float = 120_000.0,
        record: bool = False,
    ) -> None:
        super().__init__()
        self.params = params
        self.dt = float(dt)
        self.t_final = float(t_final)
        self.q_limit = float(q_limit)
        self.target_apogee = float(target_apogee)
        self.record = bool(record)

        self.action_space = spaces.Box(
            low=np.array([0.0, -self.params.max_gimbal], dtype=np.float32),
            high=np.array([1.0, self.params.max_gimbal], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )

        self.state: RocketState | None = None
        self.t = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.max_q_seen = 0.0
        self.max_alt_seen = 0.0
        self.burnout = False
        self.telemetry: list[Telemetry] | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        record = self.record
        if options and "record" in options:
            record = bool(options["record"])

        angle = -np.pi / 2.0
        quat_up = np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0], dtype=float)
        self.state = RocketState(
            pos=np.array([0.0, 0.0, 0.0], dtype=float),
            vel=np.array([0.0, 0.0, 0.0], dtype=float),
            quat=quat_up,
            omega=np.array([0.0, 0.0, 0.0], dtype=float),
            mass=float(self.params.dry_mass + self.params.prop_mass),
        )
        self.t = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.max_q_seen = 0.0
        self.max_alt_seen = 0.0
        self.burnout = False
        self.telemetry = [] if record else None

        scalars = self._scalars(self.state)
        obs = self._observation(scalars)
        info = self._info(scalars, terminated=False, truncated=False, apogee_reached=False, crash=False)
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        act = np.asarray(action, dtype=np.float32).reshape(2)
        throttle = float(np.clip(act[0], 0.0, 1.0))
        gimbal_pitch = float(np.clip(act[1], -self.params.max_gimbal, self.params.max_gimbal))
        if self.burnout:
            throttle = 0.0
        control = Control(throttle=throttle, gimbal_pitch=gimbal_pitch, gimbal_yaw=0.0)

        self.state = step(self.state, control, self.params, self.dt)
        self.t += self.dt
        self.burnout = bool(self.state.mass <= self.params.dry_mass + 1e-6)

        self._check_state(self.state)

        scalars = self._scalars(self.state)
        self.max_q_seen = max(self.max_q_seen, scalars.q_dyn)
        self.max_alt_seen = max(self.max_alt_seen, scalars.altitude)

        crash = bool(scalars.altitude < -1.0)
        apogee_reached = bool(self.burnout and scalars.vz <= 0.0 and scalars.altitude > 0.0)
        terminated = bool(crash or apogee_reached)
        truncated = bool(self.t >= self.t_final)

        q_excess = max(0.0, scalars.q_dyn - self.q_limit) / self.q_limit
        tilt_pen = abs(gimbal_pitch) / self.params.max_gimbal if self.params.max_gimbal > 0.0 else 0.0
        smooth_pen = float(np.linalg.norm(act - self.prev_action))
        reward = 0.02 * (scalars.vz / 2000.0)
        reward -= 2.0 * (q_excess**2)
        reward -= 0.01 * (tilt_pen**2)
        reward -= 0.02 * (smooth_pen**2)

        if terminated:
            if apogee_reached:
                apogee_err = abs(self.max_alt_seen - self.target_apogee) / self.target_apogee
                reward += 50.0 * float(np.exp(-5.0 * apogee_err))
            if crash:
                reward -= 50.0
            if self.max_q_seen > 1.5 * self.q_limit:
                reward -= 25.0

        self.prev_action = act

        if self.telemetry is not None:
            self.telemetry.append(compute_telemetry(self.state, control, self.params, t=self.t))

        obs = self._observation(scalars)
        info = self._info(
            scalars,
            terminated=terminated,
            truncated=truncated,
            apogee_reached=apogee_reached,
            crash=crash,
        )
        return obs, float(reward), terminated, truncated, info

    def _scalars(self, state: RocketState) -> EpisodeScalars:
        altitude = float(state.pos[2])
        speed = float(np.linalg.norm(state.vel))
        q_dyn = 0.5 * density(altitude) * speed * speed
        return EpisodeScalars(
            altitude=altitude,
            speed=speed,
            q_dyn=float(q_dyn),
            vx=float(state.vel[0]),
            vz=float(state.vel[2]),
        )

    def _observation(self, scalars: EpisodeScalars) -> np.ndarray:
        fuel_frac = (self.state.mass - self.params.dry_mass) / self.params.prop_mass
        time_frac = self.t / self.t_final if self.t_final > 0.0 else 0.0
        obs = np.array(
            [
                scalars.altitude / 120_000.0,
                scalars.vz / 2000.0,
                scalars.vx / 2000.0,
                scalars.speed / 2500.0,
                scalars.q_dyn / self.q_limit if self.q_limit > 0.0 else 0.0,
                fuel_frac,
                time_frac,
            ],
            dtype=np.float32,
        )
        return obs

    def _info(
        self,
        scalars: EpisodeScalars,
        *,
        terminated: bool,
        truncated: bool,
        apogee_reached: bool,
        crash: bool,
    ) -> dict:
        return {
            "t": self.t,
            "altitude": scalars.altitude,
            "speed": scalars.speed,
            "q_dyn": scalars.q_dyn,
            "max_q_seen": self.max_q_seen,
            "max_alt_seen": self.max_alt_seen,
            "burnout": self.burnout,
            "terminated": terminated,
            "truncated": truncated,
            "apogee_reached": apogee_reached,
            "crash": crash,
        }

    @staticmethod
    def _check_state(state: RocketState) -> None:
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
    env = RocketAscentEnv(demo_params, dt=0.05, t_final=20.0, record=False)
    obs, info = env.reset(seed=0)
    done = False
    truncated = False
    steps = 0
    while not (done or truncated):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        steps += 1
    print(f"steps: {steps}")
    print(f"terminated: {done}, truncated: {truncated}")
    print(f"final altitude: {info['altitude']:.3f} m")
    print(f"max altitude: {info['max_alt_seen']:.3f} m")
    print(f"max q_dyn: {info['max_q_seen']:.3f} Pa")
