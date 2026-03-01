"""Gymnasium environment for Falcon 9-inspired launch control."""

from __future__ import annotations

from dataclasses import dataclass, replace

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
        survival_reward: float = 0.005,
        attitude_hold_kp: float = 0.9,
        attitude_hold_kd: float = 0.25,
        attitude_hold_max_altitude_m: float = 25_000.0,
        ascent_guard_altitude_m: float = 8_000.0,
        ascent_guard_min_vz_mps: float = 350.0,
        min_throttle_ascent: float = 0.72,
        coast_terminate_vz_mps: float = 0.0,
        min_coast_time_s: float = 1.0,
        hard_violation_steps: int = 6,
        start_state_randomization: bool = True,
        domain_randomization: bool = True,
        max_start_altitude_m: float = 500.0,
    ) -> None:
        super().__init__()
        self.base_params = params
        self.params = params
        self.mission = mission
        self.atmosphere_cfg = atmosphere_cfg
        self.dt = float(dt)
        self.dt_nominal = float(dt)
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
        self.hard_violation_steps = int(max(1, hard_violation_steps))
        self.start_state_randomization = bool(start_state_randomization)
        self.domain_randomization = bool(domain_randomization)
        self.max_start_altitude_m = float(max_start_altitude_m)
        self.curriculum_target_altitude_m = float(mission.target_altitude_m)
        self.curriculum_start_altitude_cap_m = float(max_start_altitude_m)

        self.action_space = spaces.Box(
            low=np.array([0.0, -self.params.max_gimbal], dtype=np.float32),
            high=np.array([1.0, self.params.max_gimbal], dtype=np.float32),
            dtype=np.float32,
        )
        # Fixed-scale normalized observation; include variables used by reward and success checks.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float32)

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
        self.apogee_reached = False
        self.termination_reason = "running"
        self.last_g_load = 0.0
        self.prev_vz = 0.0
        self.prev_potential = 0.0
        self.extreme_q_steps = 0
        self.extreme_g_steps = 0
        self.telemetry: list[Telemetry] | None = None
        self.reward_sums = self._empty_reward_sums()
        self.control_sum = np.zeros(2, dtype=np.float64)
        self.control_sumsq = np.zeros(2, dtype=np.float64)
        self.control_count = 0

    def _empty_reward_sums(self) -> dict[str, float]:
        return {
            "survival": 0.0,
            "progress": 0.0,
            "q_pen": 0.0,
            "g_pen": 0.0,
            "tilt_pen": 0.0,
            "smooth_pen": 0.0,
            "fuel_pen": 0.0,
            "terminal_bonus": 0.0,
            "terminal_penalty": 0.0,
        }

    def set_curriculum(self, target_altitude_m: float, start_altitude_cap_m: float | None = None) -> None:
        self.curriculum_target_altitude_m = float(max(200.0, target_altitude_m))
        if start_altitude_cap_m is not None:
            self.curriculum_start_altitude_cap_m = float(max(0.0, start_altitude_cap_m))

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

    def _sample_vehicle_params(self) -> tuple[VehicleParams, float]:
        if not self.domain_randomization:
            return self.base_params, self.dt_nominal

        rng = self.np_random
        mass_scale = float(rng.uniform(0.95, 1.05))
        thrust_scale = float(rng.uniform(0.94, 1.06))
        isp_scale = float(rng.uniform(0.97, 1.03))
        cd_scale = float(rng.uniform(0.90, 1.10))
        dt_scale = float(rng.uniform(0.95, 1.05))

        sampled = replace(
            self.base_params,
            max_thrust=self.base_params.max_thrust * thrust_scale,
            isp=self.base_params.isp * isp_scale,
            dry_mass=self.base_params.dry_mass * mass_scale,
            prop_mass=self.base_params.prop_mass * mass_scale,
            cd=self.base_params.cd * cd_scale,
        )
        return sampled, self.dt_nominal * dt_scale

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

        self.params, self.dt = self._sample_vehicle_params()
        self.atmosphere_profile = self.sample_atmosphere_profile()

        angle = -np.pi / 2.0
        quat_up = np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0], dtype=float)

        z0 = 0.0
        vx0 = 0.0
        vy0 = 0.0
        vz0 = 0.0
        if self.start_state_randomization:
            alt_cap = min(self.curriculum_start_altitude_cap_m, self.curriculum_target_altitude_m)
            alt_cap = max(0.0, alt_cap)
            z0 = float(self.np_random.uniform(0.0, alt_cap))
            blend = np.clip(z0 / max(self.curriculum_target_altitude_m, 1.0), 0.0, 1.0)
            vx0 = float(self.np_random.uniform(0.0, 220.0 * (0.3 + 0.7 * blend)))
            vy0 = float(self.np_random.uniform(-40.0, 40.0))
            vz0 = float(self.np_random.uniform(0.0, 320.0 * (0.2 + 0.8 * blend)))

        self.state = RocketState(
            pos=np.array([0.0, 0.0, z0], dtype=float),
            vel=np.array([vx0, vy0, vz0], dtype=float),
            quat=quat_up,
            omega=np.array([0.0, 0.0, 0.0], dtype=float),
            mass=float(self.params.dry_mass + self.params.prop_mass),
        )
        self.t = 0.0
        self.prev_action = np.array([self.min_throttle_liftoff, 0.0], dtype=np.float32)
        self.max_q_seen = 0.0
        self.max_g_seen = 0.0
        self.max_alt_seen = float(z0)
        self.burnout = False
        self.burnout_scalars = None
        self.burnout_time = None
        self.crash = False
        self.apogee_reached = False
        self.termination_reason = "running"
        self.last_g_load = 0.0
        self.prev_vz = float(vz0)
        self.extreme_q_steps = 0
        self.extreme_g_steps = 0
        self.telemetry = [] if record else None
        self.reward_sums = self._empty_reward_sums()
        self.control_sum = np.zeros(2, dtype=np.float64)
        self.control_sumsq = np.zeros(2, dtype=np.float64)
        self.control_count = 0

        scalars = self.scalars(self.state)
        self.prev_potential = self._progress_potential(scalars, coast_phase=False)

        obs = self.observation(scalars)
        info = self.info_dict(
            scalars=scalars,
            terminated=False,
            truncated=False,
            success=False,
            hard_violation=False,
            terminal_penalty=0.0,
            step_components=self._empty_reward_sums(),
        )
        return obs, info

    def _progress_potential(self, scalars: EpisodeScalars, *, coast_phase: bool) -> float:
        target_alt = max(self.curriculum_target_altitude_m, 1.0)
        target_vx = max(self.mission.target_vx_mps, 1.0)
        target_down = max(self.mission.target_downrange_m, 1.0)
        side_obj_gate = float(np.clip((scalars.altitude - 10_000.0) / 30_000.0, 0.0, 1.0))
        side_obj_w = 0.10 + 0.90 * side_obj_gate

        downrange_err = (scalars.downrange - self.mission.target_downrange_m) / target_down
        vy_pen = scalars.vy / 350.0
        fpa_target = 0.5 * (self.mission.min_flight_path_angle_deg + self.mission.max_flight_path_angle_deg)
        fpa_err = (scalars.flight_path_angle_deg - fpa_target) / 45.0

        alt_term = np.clip(scalars.altitude / target_alt, 0.0, 2.0)
        vx_term = np.clip(max(scalars.vx, 0.0) / target_vx, 0.0, 2.0)

        specific_energy = 0.5 * (scalars.speed**2) - G0 * scalars.altitude
        target_energy = 0.5 * (self.mission.target_vx_mps**2 + self.mission.target_vz_mps**2) - G0 * target_alt
        energy_err = abs(specific_energy - target_energy) / max(abs(target_energy), 1.0)

        energy_w = 0.25 if coast_phase else 0.08
        phi = (
            1.40 * alt_term
            + 0.60 * vx_term
            - 0.45 * side_obj_w * (downrange_err**2)
            - 0.20 * side_obj_w * (vy_pen**2)
            - 0.20 * side_obj_w * (fpa_err**2)
            - energy_w * energy_err
        )
        return float(phi)

    def _mission_stage(self) -> str:
        target_alt = float(self.curriculum_target_altitude_m)
        if target_alt < 20_000.0:
            return "low"
        if target_alt < 60_000.0:
            return "mid"
        return "high"

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self.state is None:
            raise RuntimeError("Call reset() before step().")

        prev_scalars = self.scalars(self.state)
        prev_mass = float(self.state.mass)
        prev_vz = float(self.state.vel[2])

        act = np.asarray(action, dtype=np.float32).reshape(2)
        throttle = float(np.clip(act[0], 0.0, 1.0))
        gimbal_pitch_cmd = float(np.clip(act[1], -self.params.max_gimbal, self.params.max_gimbal))

        if self.t < self.liftoff_guard_time:
            throttle = max(throttle, self.min_throttle_liftoff)
        elif (
            float(self.state.pos[2]) < self.ascent_guard_altitude_m
            and float(self.state.vel[2]) < self.ascent_guard_min_vz_mps
            and not self.burnout
        ):
            throttle = max(throttle, self.min_throttle_ascent)

        if self.burnout:
            throttle = 0.0
        elif bool(self.state.mass <= self.params.dry_mass + 1e-6):
            self.burnout = True
            self.burnout_scalars = prev_scalars
            self.burnout_time = self.t
            throttle = 0.0

        gimbal_pitch = gimbal_pitch_cmd
        if float(self.state.pos[2]) <= self.attitude_hold_max_altitude_m and not self.burnout:
            thrust_axis_world = quat_rotate(self.state.quat, np.array([1.0, 0.0, 0.0], dtype=float))
            pitch_angle = float(np.arctan2(thrust_axis_world[0], max(thrust_axis_world[2], 1e-6)))
            pitch_rate = float(self.state.omega[1])
            hold_correction = -self.attitude_hold_kp * pitch_angle - self.attitude_hold_kd * pitch_rate
            gimbal_pitch = float(
                np.clip(gimbal_pitch_cmd + hold_correction, -self.params.max_gimbal, self.params.max_gimbal)
            )

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

        burnout_now = bool(self.state.mass <= self.params.dry_mass + 1e-6)
        if burnout_now and not self.burnout:
            self.burnout = True
            self.burnout_scalars = scalars
            self.burnout_time = self.t

        q_excess = max(0.0, scalars.q_dyn - self.mission.max_q_pa) / max(self.mission.max_q_pa, 1.0)
        g_excess = max(0.0, g_load - self.mission.max_g_load) / max(self.mission.max_g_load, 1.0)
        smooth_pen = float(np.linalg.norm(applied_action - self.prev_action))
        tilt_pen = abs(gimbal_pitch) / max(self.params.max_gimbal, 1e-6)
        fuel_step = max(0.0, prev_mass - float(self.state.mass)) / max(self.params.prop_mass, 1.0)

        if scalars.q_dyn > 2.0 * self.mission.max_q_pa:
            self.extreme_q_steps += 1
        else:
            self.extreme_q_steps = 0
        if g_load > 2.0 * self.mission.max_g_load:
            self.extreme_g_steps += 1
        else:
            self.extreme_g_steps = 0
        hard_violation = bool(
            self.extreme_q_steps >= self.hard_violation_steps or self.extreme_g_steps >= self.hard_violation_steps
        )

        coast_time = self.t - (self.burnout_time if self.burnout_time is not None else self.t)
        apogee_reached = bool(
            self.burnout
            and coast_time >= self.min_coast_time_s
            and prev_vz > 0.0
            and scalars.vz <= self.coast_terminate_vz_mps
        )
        self.apogee_reached = bool(self.apogee_reached or apogee_reached)

        terminated = bool(self.crash or hard_violation or apogee_reached)
        truncated = bool(self.t >= self.t_final)
        if self.crash:
            self.termination_reason = "crash"
        elif hard_violation:
            self.termination_reason = "hard_violation"
        elif apogee_reached:
            self.termination_reason = "apogee"
        elif truncated:
            self.termination_reason = "t_final"
        else:
            self.termination_reason = "running"

        coast_phase = bool(self.burnout)
        phi_prev = self.prev_potential
        phi_now = self._progress_potential(scalars, coast_phase=coast_phase)
        progress_reward = 4.0 * (phi_now - phi_prev)

        q_pen = (1.7 if not coast_phase else 1.0) * (q_excess**2)
        g_pen = (1.9 if not coast_phase else 1.1) * (g_excess**2)
        tilt_cost = 0.05 * (tilt_pen**2) if not coast_phase else 0.0
        smooth_cost = 0.05 * (smooth_pen**2) if not coast_phase else 0.0
        fuel_cost = 0.06 * fuel_step if scalars.altitude < 40_000.0 else 0.14 * fuel_step

        reward = self.survival_reward + progress_reward - q_pen - g_pen - tilt_cost - smooth_cost - fuel_cost

        terminal_penalty = 0.0
        success = False
        if terminated or truncated:
            terminal_penalty = float(np.clip(self.terminal_error_cost(scalars), 0.0, 1.0))
            success = bool(self.terminal_success(scalars))
            if success:
                reward += 100.0
                self.reward_sums["terminal_bonus"] += 100.0
            else:
                fail_pen = 50.0 + 25.0 * terminal_penalty
                reward -= fail_pen
                self.reward_sums["terminal_penalty"] -= fail_pen
                if not self.burnout and self.max_alt_seen < 0.65 * self.curriculum_target_altitude_m:
                    stall_pen = 25.0
                    reward -= stall_pen
                    self.reward_sums["terminal_penalty"] -= stall_pen

        burnout_bonus = 0.0
        if self.burnout and self.burnout_scalars is not None and not (terminated or truncated):
            burnout_ratio = self.burnout_scalars.altitude / max(self.curriculum_target_altitude_m, 1.0)
            burnout_bonus = 0.015 * float(np.clip(burnout_ratio, 0.0, 1.0))
            reward += burnout_bonus

        step_components = {
            "survival": self.survival_reward,
            "progress": progress_reward,
            "q_pen": -q_pen,
            "g_pen": -g_pen,
            "tilt_pen": -tilt_cost,
            "smooth_pen": -smooth_cost,
            "fuel_pen": -fuel_cost,
            "terminal_bonus": burnout_bonus,
            "terminal_penalty": 0.0,
        }
        for key, value in step_components.items():
            self.reward_sums[key] += float(value)
        self.control_sum += applied_action.astype(np.float64)
        self.control_sumsq += np.square(applied_action.astype(np.float64))
        self.control_count += 1

        self.prev_action = applied_action
        self.prev_vz = scalars.vz
        self.prev_potential = phi_now
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
            terminal_penalty=terminal_penalty,
            step_components=step_components,
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
        thrust_axis_world = quat_rotate(self.state.quat, np.array([1.0, 0.0, 0.0], dtype=float))
        pitch_angle = float(np.arctan2(thrust_axis_world[0], max(thrust_axis_world[2], 1e-6)))
        omega_mag = float(np.linalg.norm(self.state.omega))
        fpa_rad = np.deg2rad(scalars.flight_path_angle_deg)

        obs = np.array(
            [
                scalars.altitude / max(self.curriculum_target_altitude_m, 1.0),
                scalars.vx / 3000.0,
                scalars.vz / 2500.0,
                scalars.speed / 3500.0,
                scalars.q_dyn / max(self.mission.max_q_pa, 1.0),
                scalars.g_load / max(self.mission.max_g_load, 1.0),
                fuel_frac,
                scalars.wind_x / 80.0,
                time_frac,
                scalars.downrange / max(self.mission.target_downrange_m, 1.0),
                scalars.vy / 1000.0,
                np.sin(fpa_rad),
                np.cos(fpa_rad),
                pitch_angle / (0.5 * np.pi),
                omega_mag / 3.0,
                self.prev_action[0],
                self.prev_action[1] / max(self.params.max_gimbal, 1e-6),
                1.0 if self.burnout else 0.0,
            ],
            dtype=np.float32,
        )
        return obs

    def terminal_success(self, scalars: EpisodeScalars) -> bool:
        mission = self.mission
        fuel_used = (self.params.dry_mass + self.params.prop_mass - self.state.mass) / max(self.params.prop_mass, 1.0)
        stage = self._mission_stage()
        if stage == "low":
            vx_ok = True
            vz_ok = abs(scalars.vz) <= 300.0
            fpa_ok = True
            downrange_ok = True
        elif stage == "mid":
            vx_ok = scalars.vx >= 0.5 * mission.min_terminal_vx_mps
            vz_ok = abs(scalars.vz) <= 220.0
            fpa_ok = (mission.min_flight_path_angle_deg - 5.0) <= scalars.flight_path_angle_deg <= (
                mission.max_flight_path_angle_deg + 5.0
            )
            downrange_ok = abs(scalars.downrange - mission.target_downrange_m) <= 1.5 * mission.max_terminal_downrange_m
        else:
            fpa_ok = mission.min_flight_path_angle_deg <= scalars.flight_path_angle_deg <= mission.max_flight_path_angle_deg
            downrange_ok = (
                abs(scalars.downrange - mission.target_downrange_m) <= mission.max_terminal_downrange_m
            )
            vx_ok = mission.min_terminal_vx_mps <= scalars.vx <= mission.max_terminal_vx_mps
            vz_ok = abs(scalars.vz) <= mission.max_terminal_abs_vz_mps

        return bool(
            scalars.altitude >= self.curriculum_target_altitude_m
            and vx_ok
            and vz_ok
            and fpa_ok
            and downrange_ok
            and self.max_q_seen <= mission.max_q_pa
            and self.max_g_seen <= mission.max_g_load
            and fuel_used <= mission.max_fuel_fraction_used
            and not self.crash
        )

    def terminal_error_cost(self, scalars: EpisodeScalars) -> float:
        mission = self.mission
        stage = self._mission_stage()
        alt_err = max(0.0, self.curriculum_target_altitude_m - scalars.altitude) / max(
            self.curriculum_target_altitude_m, 1.0
        )
        downrange_err = abs(scalars.downrange - mission.target_downrange_m) / max(mission.max_terminal_downrange_m, 1.0)
        vx_target = 0.5 * (mission.min_terminal_vx_mps + mission.max_terminal_vx_mps)
        vx_err = abs(scalars.vx - vx_target) / max(vx_target, 1.0)
        vz_err = abs(scalars.vz) / max(mission.max_terminal_abs_vz_mps, 1.0)
        fpa_target = 0.5 * (mission.min_flight_path_angle_deg + mission.max_flight_path_angle_deg)
        fpa_err = abs(scalars.flight_path_angle_deg - fpa_target) / max(abs(fpa_target), 1.0)
        fuel_used = (self.params.dry_mass + self.params.prop_mass - self.state.mass) / max(self.params.prop_mass, 1.0)
        fuel_err = max(0.0, fuel_used - mission.max_fuel_fraction_used)

        if stage == "low":
            return float(0.80 * alt_err + 0.15 * vz_err + 0.05 * fuel_err)
        if stage == "mid":
            return float(0.55 * alt_err + 0.12 * downrange_err + 0.10 * vx_err + 0.13 * vz_err + 0.07 * fpa_err + 0.03 * fuel_err)
        return float(
            0.35 * alt_err + 0.20 * downrange_err + 0.15 * vx_err + 0.15 * vz_err + 0.10 * fpa_err + 0.05 * fuel_err
        )

    def info_dict(
        self,
        *,
        scalars: EpisodeScalars,
        terminated: bool,
        truncated: bool,
        success: bool,
        hard_violation: bool,
        terminal_penalty: float,
        step_components: dict[str, float],
    ) -> dict:
        fuel_used = (self.params.dry_mass + self.params.prop_mass - self.state.mass) / max(self.params.prop_mass, 1.0)
        control_count = max(self.control_count, 1)
        ctrl_mean = self.control_sum / float(control_count)
        ctrl_var = np.maximum(self.control_sumsq / float(control_count) - np.square(ctrl_mean), 0.0)
        burnout_altitude = self.burnout_scalars.altitude if self.burnout_scalars is not None else np.nan
        apogee_altitude = self.max_alt_seen
        time_to_apogee = np.nan
        if self.apogee_reached and self.burnout_time is not None:
            time_to_apogee = max(0.0, self.t - self.burnout_time)
        return {
            "t": self.t,
            "dt": self.dt,
            "terminated": terminated,
            "truncated": truncated,
            "success": success,
            "termination_reason": self.termination_reason,
            "crash": self.crash,
            "hard_violation": hard_violation,
            "burnout": self.burnout,
            "phase": "coast" if self.burnout else "powered",
            "burnout_time_s": self.burnout_time,
            "altitude_at_burnout_m": burnout_altitude,
            "altitude_at_apogee_m": apogee_altitude,
            "time_to_apogee_s": time_to_apogee,
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
            "apogee_reached": self.apogee_reached,
            "terminal_penalty": terminal_penalty,
            "reward_components_step": dict(step_components),
            "reward_components_cumulative": dict(self.reward_sums),
            "mean_throttle": float(ctrl_mean[0]),
            "throttle_variance": float(ctrl_var[0]),
            "gimbal_variance": float(ctrl_var[1]),
            "mission": {
                "space_boundary_altitude_m": self.mission.space_boundary_altitude_m,
                "target_altitude_m": self.curriculum_target_altitude_m,
                "target_vx_mps": self.mission.target_vx_mps,
                "target_vz_mps": self.mission.target_vz_mps,
                "max_q_pa": self.mission.max_q_pa,
                "max_g_load": self.mission.max_g_load,
                "target_downrange_m": self.mission.target_downrange_m,
            },
            "vehicle": {
                "max_thrust": self.params.max_thrust,
                "isp": self.params.isp,
                "dry_mass": self.params.dry_mass,
                "prop_mass": self.params.prop_mass,
                "cd": self.params.cd,
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
