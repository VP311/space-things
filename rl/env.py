"""Gymnasium environment for Falcon 9-inspired launch control."""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config.defaults import AtmosphereConfig, MissionConfig
from sim.atmosphere import AtmosphereProfile, density, wind_velocity
from sim.constants import G0
from sim.dynamics import aero_forces, step
from sim.math3d import quat_rotate
from sim.runner import Telemetry, compute_telemetry
from sim.vehicle import Control, RocketState, VehicleParams


@dataclass(frozen=True)
class EpisodeScalars:
    altitude: float
    speed: float
    q_dyn: float
    vx: float
    vy: float
    vz: float
    downrange: float
    flight_path_angle_deg: float
    wind_x: float
    wind_y: float
    g_load: float


def baseline_action(
    t: float,
    max_gimbal: float,
    *,
    altitude_m: float = 0.0,
    q_dyn_pa: float = 0.0,
    q_limit_pa: float = 40_000.0,
) -> np.ndarray:
    """Simple open-loop baseline with crude max-Q throttle bucket and gravity turn."""
    if t < 5.0:
        throttle = 1.0
    elif q_dyn_pa > q_limit_pa:
        throttle = 0.58
    elif t < 100.0:
        throttle = 0.95
    elif altitude_m < 70_000.0:
        throttle = 0.78
    else:
        throttle = 0.65

    if t < 12.0:
        gimbal_pitch = 0.0
    elif t < 80.0:
        gimbal_pitch = np.deg2rad(-1.0)
    elif t < 130.0:
        gimbal_pitch = np.deg2rad(-1.8)
    else:
        gimbal_pitch = np.deg2rad(-1.0)

    gimbal_pitch = float(np.clip(gimbal_pitch, -max_gimbal, max_gimbal))
    return np.array([throttle, gimbal_pitch], dtype=np.float32)


class RocketAscentEnv(gym.Env[np.ndarray, np.ndarray]):
    """Falcon 9-inspired ascent env with atmospheric randomization and load constraints."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        params: VehicleParams,
        mission: MissionConfig,
        atmosphere_cfg: AtmosphereConfig,
        dt: float = 0.05,
        t_final: float = 300.0,
        record: bool = False,
        min_throttle_liftoff: float = 0.65,
        liftoff_guard_time: float = 3.0,
        survival_reward: float = 0.01,
        attitude_hold_kp: float = 0.9,
        attitude_hold_kd: float = 0.25,
        attitude_hold_max_altitude_m: float = 25_000.0,
        ascent_guard_altitude_m: float = 8_000.0,
        ascent_guard_min_vz_mps: float = 350.0,
        min_throttle_ascent: float = 0.72,
        coast_terminate_vz_mps: float = 0.0,
        min_coast_time_s: float = 1.0,
    ) -> None:
        super().__init__()
        self.params = params
        self.mission = mission
        self.atmosphere_cfg = atmosphere_cfg
        self.dt = float(dt)
        self.t_final = float(t_final)
        self.record = bool(record)
        self.min_throttle_liftoff = float(min_throttle_liftoff)
        self.liftoff_guard_time = float(liftoff_guard_time)
        self.survival_reward = float(survival_reward)
        self.attitude_hold_kp = float(attitude_hold_kp)
        self.attitude_hold_kd = float(attitude_hold_kd)
        self.attitude_hold_max_altitude_m = float(attitude_hold_max_altitude_m)
        self.ascent_guard_altitude_m = float(ascent_guard_altitude_m)
        self.ascent_guard_min_vz_mps = float(ascent_guard_min_vz_mps)
        self.min_throttle_ascent = float(min_throttle_ascent)
        self.coast_terminate_vz_mps = float(coast_terminate_vz_mps)
        self.min_coast_time_s = float(min_coast_time_s)

        self.action_space = spaces.Box(
            low=np.array([0.0, -self.params.max_gimbal], dtype=np.float32),
            high=np.array([1.0, self.params.max_gimbal], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        self.state: RocketState | None = None
        self.atmosphere_profile = AtmosphereProfile()
        self.t = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.max_q_seen = 0.0
        self.max_g_seen = 0.0
        self.max_alt_seen = 0.0
        self.burnout = False
        self.burnout_scalars: EpisodeScalars | None = None
        self.burnout_time: float | None = None
        self.crash = False
        self.prev_altitude = 0.0
        self.last_g_load = 0.0
        self.telemetry: list[Telemetry] | None = None

    def sample_atmosphere_profile(self) -> AtmosphereProfile:
        rng = self.np_random
        cfg = self.atmosphere_cfg
        if not cfg.randomize:
            return AtmosphereProfile(
                density_scale=1.0,
                scale_height_m=8_500.0,
                gust_freq_x_hz=cfg.gust_freq_x_hz,
                gust_freq_y_hz=cfg.gust_freq_y_hz,
            )

        return AtmosphereProfile(
            density_scale=float(rng.uniform(cfg.density_scale_min, cfg.density_scale_max)),
            scale_height_m=float(rng.uniform(cfg.scale_height_min_m, cfg.scale_height_max_m)),
            wind_bias_x_mps=float(rng.uniform(cfg.wind_bias_x_min_mps, cfg.wind_bias_x_max_mps)),
            wind_bias_y_mps=float(rng.uniform(cfg.wind_bias_y_min_mps, cfg.wind_bias_y_max_mps)),
            wind_shear_x_mps=float(rng.uniform(cfg.wind_shear_x_min_mps, cfg.wind_shear_x_max_mps)),
            wind_shear_y_mps=float(rng.uniform(cfg.wind_shear_y_min_mps, cfg.wind_shear_y_max_mps)),
            gust_amp_x_mps=float(rng.uniform(cfg.gust_amp_x_min_mps, cfg.gust_amp_x_max_mps)),
            gust_amp_y_mps=float(rng.uniform(cfg.gust_amp_y_min_mps, cfg.gust_amp_y_max_mps)),
            gust_freq_x_hz=cfg.gust_freq_x_hz,
            gust_freq_y_hz=cfg.gust_freq_y_hz,
            gust_phase_x_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
            gust_phase_y_rad=float(rng.uniform(0.0, 2.0 * np.pi)),
        )

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

        self.atmosphere_profile = self.sample_atmosphere_profile()

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
        self.prev_action = np.array([self.min_throttle_liftoff, 0.0], dtype=np.float32)
        self.max_q_seen = 0.0
        self.max_g_seen = 0.0
        self.max_alt_seen = 0.0
        self.burnout = False
        self.burnout_scalars = None
        self.burnout_time = None
        self.crash = False
        self.prev_altitude = 0.0
        self.last_g_load = 0.0
        self.telemetry = [] if record else None

        scalars = self.scalars(self.state)
        obs = self.observation(scalars)
        info = self.info_dict(
            scalars=scalars,
            terminated=False,
            truncated=False,
            success=False,
            hard_violation=False,
        )
        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        act = np.asarray(action, dtype=np.float32).reshape(2)
        throttle = float(np.clip(act[0], 0.0, 1.0))
        if self.t < self.liftoff_guard_time:
            throttle = max(throttle, self.min_throttle_liftoff)
        elif (
            float(self.state.pos[2]) < self.ascent_guard_altitude_m
            and float(self.state.vel[2]) < self.ascent_guard_min_vz_mps
        ):
            throttle = max(throttle, self.min_throttle_ascent)

        gimbal_pitch_cmd = float(np.clip(act[1], -self.params.max_gimbal, self.params.max_gimbal))
        gimbal_pitch = gimbal_pitch_cmd
        if float(self.state.pos[2]) <= self.attitude_hold_max_altitude_m:
            thrust_axis_world = quat_rotate(self.state.quat, np.array([1.0, 0.0, 0.0], dtype=float))
            pitch_angle = float(np.arctan2(thrust_axis_world[0], max(thrust_axis_world[2], 1e-6)))
            pitch_rate = float(self.state.omega[1])
            hold_correction = -self.attitude_hold_kp * pitch_angle - self.attitude_hold_kd * pitch_rate
            gimbal_pitch = float(
                np.clip(gimbal_pitch_cmd + hold_correction, -self.params.max_gimbal, self.params.max_gimbal)
            )
        burnout = bool(self.state.mass <= self.params.dry_mass + 1e-6)
        if burnout:
            throttle = 0.0

        applied_action = np.array([throttle, gimbal_pitch], dtype=np.float32)
        control = Control(throttle=throttle, gimbal_pitch=gimbal_pitch, gimbal_yaw=0.0)

        thrust_world, drag_world, q_dyn_now, _ = aero_forces(
            self.state,
            control,
            self.params,
            t=self.t,
            atmosphere=self.atmosphere_profile,
        )
        mass = float(np.maximum(self.state.mass, self.params.dry_mass))
        non_gravity_accel = (thrust_world + drag_world) / mass
        g_load = float(np.linalg.norm(non_gravity_accel) / G0)

        self.state = step(
            self.state,
            control,
            self.params,
            self.dt,
            t=self.t,
            atmosphere=self.atmosphere_profile,
        )
        self.t += self.dt
        self.check_state(self.state)

        scalars = self.scalars(self.state, g_load=g_load)
        self.max_q_seen = max(self.max_q_seen, q_dyn_now, scalars.q_dyn)
        self.max_g_seen = max(self.max_g_seen, g_load)
        self.max_alt_seen = max(self.max_alt_seen, scalars.altitude)
        self.crash = bool(self.crash or scalars.altitude < -5.0)

        hard_violation = bool(
            self.max_q_seen > 1.5 * self.mission.max_q_pa
            or self.max_g_seen > 1.5 * self.mission.max_g_load
        )
        burnout_now = bool(self.state.mass <= self.params.dry_mass + 1e-6)
        if burnout_now and not self.burnout:
            self.burnout = True
            self.burnout_scalars = scalars
            self.burnout_time = self.t

        reached_space = bool(scalars.altitude >= self.mission.space_boundary_altitude_m)
        terminated = bool(self.crash or hard_violation or reached_space)
        truncated = bool(self.t >= self.t_final)

        q_excess = max(0.0, scalars.q_dyn - self.mission.max_q_pa) / max(self.mission.max_q_pa, 1.0)
        g_excess = max(0.0, g_load - self.mission.max_g_load) / max(self.mission.max_g_load, 1.0)
        tilt_pen = abs(gimbal_pitch) / self.params.max_gimbal if self.params.max_gimbal > 0.0 else 0.0
        smooth_pen = float(np.linalg.norm(applied_action - self.prev_action))
        fuel_used = (self.params.dry_mass + self.params.prop_mass - self.state.mass) / max(self.params.prop_mass, 1.0)

        space_progress = np.clip(scalars.altitude / max(self.mission.space_boundary_altitude_m, 1.0), 0.0, 1.5)
        vx_progress = np.clip(max(scalars.vx, 0.0) / max(self.mission.target_vx_mps, 1.0), 0.0, 1.5)
        vz_progress = np.clip(max(scalars.vz, 0.0) / max(self.mission.target_vz_mps, 1.0), 0.0, 1.5)
        alt_gain = max(0.0, scalars.altitude - self.prev_altitude)
        downrange_err = abs(scalars.downrange - self.mission.target_downrange_m) / max(
            self.mission.target_downrange_m, 1.0
        )
        vy_pen = abs(scalars.vy) / 400.0
        fpa_target = 0.5 * (self.mission.min_flight_path_angle_deg + self.mission.max_flight_path_angle_deg)
        fpa_err = abs(scalars.flight_path_angle_deg - fpa_target) / max(abs(fpa_target), 1.0)
        atmo_frac = np.clip(scalars.q_dyn / max(self.mission.max_q_pa, 1.0), 0.0, 1.0)
        phase_coast = 1.0 if self.burnout else 0.0

        powered_reward = (
            0.18 * (alt_gain / 100.0)
            + 0.35 * space_progress
            + 0.25 * vx_progress
            + 0.20 * vz_progress
            - (1.0 + 0.8 * atmo_frac) * (q_excess**2)
            - (1.2 + 0.8 * atmo_frac) * (g_excess**2)
            - 0.05 * (tilt_pen**2)
            - 0.06 * (smooth_pen**2)
            - 0.03 * (fuel_used**2)
            - 0.05 * (downrange_err**2)
            - 0.10 * (vy_pen**2)
        )
        coast_reward = (
            0.75 * space_progress
            + 0.20 * np.clip(max(scalars.vz, 0.0) / max(self.mission.target_vz_mps, 1.0), 0.0, 1.5)
            - 0.16 * (downrange_err**2)
            - 0.20 * (fpa_err**2)
            - 0.10 * (vy_pen**2)
            - 0.06 * (smooth_pen**2)
        )
        reward = self.survival_reward + (1.0 - phase_coast) * powered_reward + phase_coast * coast_reward

        terminal_scalars = scalars
        success = bool((terminated or truncated) and self.terminal_success(terminal_scalars))
        if terminated or truncated:
            terminal_penalty = self.terminal_error_cost(terminal_scalars)
            if success:
                reward += 300.0
            else:
                reward -= 80.0 * terminal_penalty
            if self.crash:
                reward -= 100.0
            if hard_violation:
                reward -= 60.0

        self.prev_action = applied_action
        self.prev_altitude = scalars.altitude
        self.last_g_load = g_load

        if self.telemetry is not None:
            self.telemetry.append(
                compute_telemetry(
                    self.state,
                    control,
                    self.params,
                    t=self.t,
                    atmosphere=self.atmosphere_profile,
                )
            )

        obs = self.observation(scalars)
        info = self.info_dict(
            scalars=scalars,
            terminated=terminated,
            truncated=truncated,
            success=success,
            hard_violation=hard_violation,
        )
        return obs, float(reward), terminated, truncated, info

    def scalars(self, state: RocketState, *, g_load: float | None = None) -> EpisodeScalars:
        altitude = float(state.pos[2])
        speed = float(np.linalg.norm(state.vel))
        w_world = wind_velocity(altitude, t=self.t, profile=self.atmosphere_profile)
        v_air = state.vel - w_world
        q_dyn = 0.5 * density(altitude, profile=self.atmosphere_profile) * float(np.dot(v_air, v_air))
        horizontal_speed = float(np.linalg.norm(state.vel[:2]))
        gamma = float(np.rad2deg(np.arctan2(state.vel[2], max(horizontal_speed, 1e-6))))

        return EpisodeScalars(
            altitude=altitude,
            speed=speed,
            q_dyn=float(q_dyn),
            vx=float(state.vel[0]),
            vy=float(state.vel[1]),
            vz=float(state.vel[2]),
            downrange=float(np.linalg.norm(state.pos[:2])),
            flight_path_angle_deg=gamma,
            wind_x=float(w_world[0]),
            wind_y=float(w_world[1]),
            g_load=float(self.last_g_load if g_load is None else g_load),
        )

    def observation(self, scalars: EpisodeScalars) -> np.ndarray:
        if self.state is None:
            raise RuntimeError("State unavailable before reset().")

        fuel_frac = (self.state.mass - self.params.dry_mass) / max(self.params.prop_mass, 1.0)
        time_frac = self.t / self.t_final if self.t_final > 0.0 else 0.0
        rho = density(scalars.altitude, profile=self.atmosphere_profile)
        rho0 = density(0.0, profile=self.atmosphere_profile)
        rho_ratio = np.clip(rho / max(rho0, 1e-9), 0.0, 1.0)
        crossrange_scale = max(self.mission.max_terminal_downrange_m, 1.0)
        fpa_norm = np.clip(scalars.flight_path_angle_deg / 90.0, -1.0, 1.0)
        max_gimbal = max(self.params.max_gimbal, 1e-6)
        thrust_axis_world = quat_rotate(self.state.quat, np.array([1.0, 0.0, 0.0], dtype=float))
        pitch_angle = float(np.arctan2(thrust_axis_world[0], max(thrust_axis_world[2], 1e-6)))
        engine_on = 0.0 if self.burnout else 1.0
        obs = np.array(
            [
                scalars.altitude / max(self.mission.space_boundary_altitude_m, 1.0),
                scalars.downrange / max(self.mission.target_downrange_m, 1.0),
                self.state.pos[1] / crossrange_scale,
                scalars.vz / 2500.0,
                scalars.vx / 3000.0,
                scalars.vy / 1000.0,
                scalars.speed / 3500.0,
                fpa_norm,
                scalars.q_dyn / max(self.mission.max_q_pa, 1.0),
                scalars.g_load / max(self.mission.max_g_load, 1.0),
                rho_ratio,
                scalars.wind_x / 80.0,
                scalars.wind_y / 80.0,
                pitch_angle / (0.5 * np.pi),
                self.state.omega[1] / 2.0,
                fuel_frac,
                self.prev_action[0],
                self.prev_action[1] / max_gimbal,
                time_frac,
                engine_on,
            ],
            dtype=np.float32,
        )
        return obs

    def terminal_success(self, scalars: EpisodeScalars) -> bool:
        mission = self.mission
        fuel_used = (self.params.dry_mass + self.params.prop_mass - self.state.mass) / max(self.params.prop_mass, 1.0)

        return bool(
            scalars.altitude >= mission.space_boundary_altitude_m
            and self.max_q_seen <= mission.max_q_pa
            and self.max_g_seen <= mission.max_g_load
            and fuel_used <= mission.max_fuel_fraction_used
            and not self.crash
        )

    def terminal_error_cost(self, scalars: EpisodeScalars) -> float:
        mission = self.mission
        space_shortfall = max(0.0, mission.space_boundary_altitude_m - scalars.altitude) / max(
            mission.space_boundary_altitude_m, 1.0
        )
        downrange_err = abs(scalars.downrange - mission.target_downrange_m) / max(mission.target_downrange_m, 1.0)
        descent_pen = max(0.0, -scalars.vz) / 500.0
        return float(0.7 * space_shortfall + 0.2 * descent_pen + 0.1 * downrange_err)

    def info_dict(
        self,
        *,
        scalars: EpisodeScalars,
        terminated: bool,
        truncated: bool,
        success: bool,
        hard_violation: bool,
    ) -> dict:
        fuel_used = (self.params.dry_mass + self.params.prop_mass - self.state.mass) / max(self.params.prop_mass, 1.0)
        return {
            "t": self.t,
            "terminated": terminated,
            "truncated": truncated,
            "success": success,
            "crash": self.crash,
            "hard_violation": hard_violation,
            "burnout": self.burnout,
            "phase": "coast" if self.burnout else "powered",
            "burnout_time_s": self.burnout_time,
            "altitude_m": scalars.altitude,
            "speed_mps": scalars.speed,
            "vx_mps": scalars.vx,
            "vy_mps": scalars.vy,
            "vz_mps": scalars.vz,
            "q_dyn_pa": scalars.q_dyn,
            "g_load": scalars.g_load,
            "flight_path_angle_deg": scalars.flight_path_angle_deg,
            "wind_x_mps": scalars.wind_x,
            "wind_y_mps": scalars.wind_y,
            "downrange_m": scalars.downrange,
            "max_altitude_m": self.max_alt_seen,
            "max_q_dyn": self.max_q_seen,
            "max_g_load": self.max_g_seen,
            "fuel_used_fraction": float(fuel_used),
            "mission": {
                "space_boundary_altitude_m": self.mission.space_boundary_altitude_m,
                "target_altitude_m": self.mission.target_altitude_m,
                "target_vx_mps": self.mission.target_vx_mps,
                "target_vz_mps": self.mission.target_vz_mps,
                "max_q_pa": self.mission.max_q_pa,
                "max_g_load": self.mission.max_g_load,
            },
            "atmosphere": {
                "density_scale": self.atmosphere_profile.density_scale,
                "scale_height_m": self.atmosphere_profile.scale_height_m,
                "wind_bias_x_mps": self.atmosphere_profile.wind_bias_x_mps,
                "wind_bias_y_mps": self.atmosphere_profile.wind_bias_y_mps,
                "wind_shear_x_mps": self.atmosphere_profile.wind_shear_x_mps,
                "wind_shear_y_mps": self.atmosphere_profile.wind_shear_y_mps,
            },
        }

    @staticmethod
    def check_state(state: RocketState) -> None:
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
