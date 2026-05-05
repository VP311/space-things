"""Gymnasium environment for Falcon 9-inspired launch control."""

from __future__ import annotations

from dataclasses import dataclass, replace

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from config.defaults import AtmosphereConfig, MissionConfig
from sim.atmosphere import AtmosphereProfile, density, wind_velocity
from sim.constants import G0, RHO0
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
    rho_ratio: float


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
    _FUEL_FRACTION_TOL = 1e-9

    def __init__(
        self,
        params: VehicleParams,
        mission: MissionConfig,
        atmosphere_cfg: AtmosphereConfig,
        curriculum_milestones_m: tuple[float, ...] | None = None,
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
        ascent_guard_min_vz_mps: float = 100.0,
        min_throttle_ascent: float = 0.72,
        coast_terminate_vz_mps: float = 0.0,
        min_coast_time_s: float = 1.0,
        hard_violation_steps: int = 6,
        start_state_randomization: bool = True,
        domain_randomization: bool = True,
        max_start_altitude_m: float = 500.0,
        action_repeat: int = 1,
        observation_noise_std: float = 0.0,
        action_lag_steps: int = 0,
    ) -> None:
        super().__init__()
        self.action_repeat = max(1, int(action_repeat))
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
        self.observation_noise_std = float(max(0.0, observation_noise_std))
        self.action_lag_steps = int(max(0, action_lag_steps))
        self.curriculum_target_altitude_m = float(mission.target_altitude_m)
        self.curriculum_start_altitude_cap_m = float(max_start_altitude_m)
        milestone_seed = (
            tuple(float(m) for m in curriculum_milestones_m)
            if curriculum_milestones_m is not None
            else (float(mission.target_altitude_m), float(mission.space_boundary_altitude_m))
        )
        self.curriculum_milestones_m = tuple(
            sorted(
                {
                    float(m) for m in milestone_seed
                }
                | {self.curriculum_target_altitude_m, float(mission.space_boundary_altitude_m)}
            )
        )

        self.action_space = spaces.Box(
            low=np.array([0.0, -self.params.max_gimbal], dtype=np.float32),
            high=np.array([1.0, self.params.max_gimbal], dtype=np.float32),
            dtype=np.float32,
        )
        # Fixed-scale normalized observation; include variables used by reward and success checks.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(20,), dtype=np.float32)

        self.state: RocketState | None = None
        self.atmosphere_profile = AtmosphereProfile()
        self.t = 0.0
        self.prev_action = np.zeros(2, dtype=np.float32)
        self._action_delay_buffer: list[np.ndarray] = []
        self.max_q_seen = 0.0
        self.max_g_seen = 0.0
        self.max_alt_seen = 0.0
        self.burnout = False
        self.burnout_scalars: EpisodeScalars | None = None
        self.burnout_time: float | None = None
        self.burnout_bonus_paid = False
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
            "burnout_bonus": 0.0,
            "success_bonus": 0.0,
            "q_pen": 0.0,
            "g_pen": 0.0,
            "tilt_pen": 0.0,
            "smooth_pen": 0.0,
            "fuel_pen": 0.0,
            "coasting_pen": 0.0,
            "omega_pen": 0.0,
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
        self._action_delay_buffer = [self.prev_action.copy() for _ in range(self.action_lag_steps)]
        self.max_q_seen = 0.0
        self.max_g_seen = 0.0
        self.max_alt_seen = float(z0)
        self.burnout = False
        self.burnout_scalars = None
        self.burnout_time = None
        self.burnout_bonus_paid = False
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

        obs = self._observe(scalars)
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

    def _curriculum_success_band(self) -> tuple[float, float | None]:
        target_alt = max(self.curriculum_target_altitude_m, 1.0)
        final_alt = max(self.mission.space_boundary_altitude_m, 1.0)
        if target_alt >= final_alt - 1.0:
            return target_alt, None
        return 0.95 * target_alt, None

    def _q_margin_pa(self) -> float:
        return float(self.mission.max_q_pa - self.max_q_seen)

    def _q_over_limit_fraction(self) -> float:
        return float(max(0.0, self.max_q_seen - self.mission.max_q_pa) / max(self.mission.max_q_pa, 1.0))

    def _terminal_q_failure_only(self, scalars: EpisodeScalars) -> bool:
        min_altitude_m, max_altitude_m = self._curriculum_success_band()
        fuel_used = (self.params.dry_mass + self.params.prop_mass - self.state.mass) / max(self.params.prop_mass, 1.0)
        altitude_ok = scalars.altitude >= min_altitude_m and (
            max_altitude_m is None or scalars.altitude <= max_altitude_m
        )
        return bool(
            altitude_ok
            and self.max_g_seen <= self.mission.max_g_load
            and fuel_used <= self.mission.max_fuel_fraction_used + self._FUEL_FRACTION_TOL
            and not self.crash
            and self.max_q_seen > self.mission.max_q_pa
        )

    def _overshoot_penalty(self, scalars: EpisodeScalars) -> float:
        _, max_altitude_m = self._curriculum_success_band()
        if max_altitude_m is None:
            return 0.0
        overshoot_fraction = max(0.0, scalars.altitude - max_altitude_m) / max(max_altitude_m, 1.0)
        return 8.0 * (overshoot_fraction**2)

    def _q_barrier_penalty(self, q_dyn_pa: float) -> float:
        q_limit = max(self.mission.max_q_pa, 1.0)
        q_soft_limit = 0.92 * q_limit
        if q_dyn_pa <= q_soft_limit:
            return 0.0
        if q_dyn_pa <= q_limit:
            frac = (q_dyn_pa - q_soft_limit) / max(q_limit - q_soft_limit, 1.0)
            return 0.8 * (frac**2)
        frac_over = (q_dyn_pa - q_limit) / q_limit
        return 0.8 + 30.0 * (frac_over**3)

    @staticmethod
    def _piecewise_linear_potential(x: float, knots: tuple[tuple[float, float], ...]) -> float:
        if x <= knots[0][0]:
            return float(knots[0][1])
        for (x0, y0), (x1, y1) in zip(knots[:-1], knots[1:]):
            if x <= x1:
                span = max(x1 - x0, 1e-9)
                t = (x - x0) / span
                return float(y0 + t * (y1 - y0))
        return float(knots[-1][1])

    def _effective_progress_altitude(self, scalars: EpisodeScalars, *, coast_phase: bool, cap_altitude_m: float) -> float:
        if coast_phase:
            return float(scalars.altitude)
        projected_apogee = scalars.altitude + (scalars.vz**2) / (2.0 * G0)
        return float(np.clip(projected_apogee, scalars.altitude, max(cap_altitude_m, scalars.altitude)))

    def _next_curriculum_milestone(self) -> float:
        current_target = float(self.curriculum_target_altitude_m)
        for milestone in self.curriculum_milestones_m:
            if milestone > current_target + 1e-6:
                return float(milestone)
        return float(max(current_target, self.mission.space_boundary_altitude_m))

    @staticmethod
    def _carry_ratio(effective_altitude_m: float, *, start_altitude_m: float, target_altitude_m: float) -> float:
        if target_altitude_m <= start_altitude_m + 1e-6:
            return 1.0
        return float(
            np.clip(
                (effective_altitude_m - start_altitude_m) / (target_altitude_m - start_altitude_m),
                0.0,
                1.0,
            )
        )

    def _mid_stage_carry_potential(self, effective_altitude_m: float) -> float:
        target_alt = max(self.curriculum_target_altitude_m, 1.0)
        next_milestone = max(self._next_curriculum_milestone(), target_alt)
        decay_start = max(1.10 * next_milestone, next_milestone + 1_000.0)
        decay_end = max(1.80 * next_milestone, decay_start + 1_000.0)
        knots = (
            (target_alt, 0.0),
            (next_milestone, 1.0),
            (decay_start, 0.85),
            (decay_end, 0.0),
        )
        return self._piecewise_linear_potential(effective_altitude_m, knots)

    def _progress_potential(self, scalars: EpisodeScalars, *, coast_phase: bool) -> float:
        # During powered flight: use projected apogee (alt + vz²/2g) so the policy
        # receives immediate reward for building vertical energy.
        # During coast: use actual altitude (projected apogee shrinks as vz→0).
        target_alt = max(self.curriculum_target_altitude_m, 1.0)
        if self.curriculum_target_altitude_m >= 88_000.0:
            target_alt = max(self.mission.space_boundary_altitude_m, target_alt)
        if coast_phase:
            effective_alt = scalars.altitude
        else:
            projected_apogee = scalars.altitude + (scalars.vz ** 2) / (2.0 * G0)
            effective_alt = float(np.clip(projected_apogee, scalars.altitude, target_alt * 1.5))
        base = float(np.clip(effective_alt / target_alt, 0.0, 1.0))
        bonus = 0.5 * float(np.clip((effective_alt - 0.8 * target_alt) / (0.2 * target_alt), 0.0, 1.0))
        return base + bonus

    def _mission_stage(self) -> str:
        target_alt = float(self.curriculum_target_altitude_m)
        if target_alt < 20_000.0:
            return "low"
        if target_alt < 60_000.0:
            return "mid"
        return "high"

    def _final_stage_active(self) -> bool:
        return bool(self.curriculum_target_altitude_m >= self.mission.space_boundary_altitude_m - 1.0)

    def _terminal_success_bonus(self, scalars: EpisodeScalars) -> float:
        if not self._final_stage_active():
            return 100.0
        overshoot_ratio = float(
            np.clip(
                (scalars.altitude - self.mission.space_boundary_altitude_m)
                / (0.10 * max(self.mission.space_boundary_altitude_m, 1.0)),
                0.0,
                1.0,
            )
        )
        return 175.0 + 25.0 * overshoot_ratio

    def _terminal_failure_penalty(self, scalars: EpisodeScalars, terminal_penalty: float) -> float:
        if self._final_stage_active() and scalars.altitude < self.mission.space_boundary_altitude_m:
            shortfall_ratio = float(
                np.clip(
                    (self.mission.space_boundary_altitude_m - scalars.altitude)
                    / (0.20 * max(self.mission.space_boundary_altitude_m, 1.0)),
                    0.0,
                    1.0,
                )
            )
            return 15.0 + 90.0 * shortfall_ratio
        return 15.0 + 30.0 * terminal_penalty

    def _burnout_energy_bonus(self) -> float:
        if not self.burnout or self.burnout_scalars is None or self.burnout_bonus_paid:
            return 0.0
        if self.max_q_seen > self.mission.max_q_pa or self.max_g_seen > self.mission.max_g_load:
            return 0.0

        self.burnout_bonus_paid = True

        target_alt = max(self.curriculum_target_altitude_m, 1.0)
        projected_apogee = self.burnout_scalars.altitude + (self.burnout_scalars.vz**2) / (2.0 * G0)

        if self.curriculum_target_altitude_m < 20_000.0:
            return 0.0
        if self.curriculum_target_altitude_m < 65_000.0:
            carry_potential = self._mid_stage_carry_potential(projected_apogee)
            return 10.0 * carry_potential

        final_target = max(self.mission.space_boundary_altitude_m, 1.0)
        hinge_start = 0.92 * final_target
        hinge_width = max(final_target - hinge_start, 1.0)
        boundary_ratio = float(np.clip((projected_apogee - hinge_start) / hinge_width, 0.0, 1.0))
        boundary_bonus = 120.0 * (boundary_ratio**2)

        overshoot_ratio = float(np.clip((projected_apogee - final_target) / (0.10 * final_target), 0.0, 1.0))
        overshoot_bonus = 25.0 * overshoot_ratio

        vz_ratio = float(np.clip((self.burnout_scalars.vz - 1_180.0) / 70.0, 0.0, 1.0))
        vz_bonus = 35.0 * vz_ratio * boundary_ratio
        return boundary_bonus + overshoot_bonus + vz_bonus

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Run action_repeat sub-steps with the same action, accumulating reward."""
        total_reward = 0.0
        obs: np.ndarray | None = None
        info: dict = {}
        terminated = False
        truncated = False
        for _ in range(self.action_repeat):
            obs, reward, terminated, truncated, info = self._single_step(action)
            total_reward += reward
            if terminated or truncated:
                break
        assert obs is not None
        return obs, total_reward, terminated, truncated, info

    def _single_step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
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
            pitch_angle = float(np.arctan2(thrust_axis_world[0], thrust_axis_world[2]))
            pitch_rate = float(self.state.omega[1])
            hold_correction = self.attitude_hold_kp * pitch_angle + self.attitude_hold_kd * pitch_rate
            gimbal_pitch = float(
                np.clip(gimbal_pitch_cmd + hold_correction, -self.params.max_gimbal, self.params.max_gimbal)
            )

        commanded_action = np.array([throttle, gimbal_pitch], dtype=np.float32)
        applied_action = self._apply_action_lag(commanded_action)
        throttle = float(applied_action[0])
        gimbal_pitch = float(applied_action[1])
        if self.burnout:
            throttle = 0.0
            applied_action = np.array([0.0, gimbal_pitch], dtype=np.float32)
            self._action_delay_buffer = [applied_action.copy() for _ in range(self.action_lag_steps)]
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

        q_pen = self._q_barrier_penalty(scalars.q_dyn)
        g_pen = (0.6 if not coast_phase else 0.4) * (g_excess**2)
        tilt_cost = 0.05 * (tilt_pen**2) if not coast_phase else 0.0
        smooth_cost = 0.015 * (smooth_pen**2) if not coast_phase else 0.0
        fuel_cost = 0.008 * fuel_step if scalars.altitude < 40_000.0 else 0.14 * fuel_step
        # Penalise low throttle during powered flight to discourage coasting with fuel remaining.
        coasting_pen = 0.02 * max(0.0, 0.5 - float(throttle)) if not coast_phase else 0.0
        # Penalise high angular rate during powered flight to discourage tumbling.
        omega_mag = float(np.linalg.norm(self.state.omega))
        omega_pen = 0.10 * (omega_mag / 3.0) ** 2 if not coast_phase else 0.0
        reward = (
            self.survival_reward
            + progress_reward
            - q_pen
            - g_pen
            - tilt_cost
            - smooth_cost
            - fuel_cost
            - coasting_pen
            - omega_pen
        )

        terminal_penalty = 0.0
        success_bonus = 0.0
        terminal_penalty_component = 0.0
        success = False
        if terminated or truncated:
            terminal_penalty = float(np.clip(self.terminal_error_cost(scalars), 0.0, 1.0))
            success = bool(self.terminal_success(scalars))
            if success:
                success_bonus = self._terminal_success_bonus(scalars)
                reward += success_bonus
            else:
                # Final-stage misses need a much steeper gradient than curriculum misses.
                fail_pen = self._terminal_failure_penalty(scalars, terminal_penalty)
                if self._terminal_q_failure_only(scalars):
                    fail_pen *= 1.75
                reward -= fail_pen
                terminal_penalty_component -= fail_pen
                if not self.burnout and self.max_alt_seen < 0.65 * self.curriculum_target_altitude_m:
                    stall_pen = 25.0
                    reward -= stall_pen
                    terminal_penalty_component -= stall_pen

        burnout_bonus = 0.0
        if not (terminated or truncated):
            burnout_bonus = self._burnout_energy_bonus()
        if burnout_bonus > 0.0:
            reward += burnout_bonus

        step_components = {
            "survival": self.survival_reward,
            "progress": progress_reward,
            "burnout_bonus": burnout_bonus,
            "success_bonus": success_bonus,
            "q_pen": -q_pen,
            "g_pen": -g_pen,
            "tilt_pen": -tilt_cost,
            "smooth_pen": -smooth_cost,
            "fuel_pen": -fuel_cost,
            "coasting_pen": -coasting_pen,
            "omega_pen": -omega_pen,
            "terminal_penalty": terminal_penalty_component,
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

        obs = self._observe(scalars)
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
        rho = density(altitude, profile=self.atmosphere_profile)
        q_dyn = 0.5 * rho * float(np.dot(v_air, v_air))
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
            rho_ratio=float(rho / RHO0),
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

        if self.burnout:
            projected_apogee = scalars.altitude
        else:
            projected_apogee = scalars.altitude + (scalars.vz ** 2) / (2.0 * G0)
        obs_target_alt = max(self.curriculum_target_altitude_m, 1.0)
        if self.curriculum_target_altitude_m >= 88_000.0:
            obs_target_alt = max(self.mission.space_boundary_altitude_m, obs_target_alt)
        apogee_ratio = float(np.clip(projected_apogee / obs_target_alt, 0.0, 1.5))

        obs = np.array(
            [
                scalars.altitude / obs_target_alt,
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
                scalars.rho_ratio,
                apogee_ratio,
            ],
            dtype=np.float32,
        )
        return obs

    def _observe(self, scalars: EpisodeScalars) -> np.ndarray:
        obs = self.observation(scalars)
        if self.observation_noise_std <= 0.0:
            return obs
        noise = self.np_random.normal(0.0, self.observation_noise_std, size=obs.shape).astype(np.float32)
        return (obs + noise).astype(np.float32)

    def _apply_action_lag(self, commanded_action: np.ndarray) -> np.ndarray:
        action = np.asarray(commanded_action, dtype=np.float32).reshape(2).copy()
        action[0] = float(np.clip(action[0], 0.0, 1.0))
        action[1] = float(np.clip(action[1], -self.params.max_gimbal, self.params.max_gimbal))
        if self.action_lag_steps <= 0:
            return action
        if len(self._action_delay_buffer) < self.action_lag_steps:
            self._action_delay_buffer.extend(action.copy() for _ in range(self.action_lag_steps - len(self._action_delay_buffer)))
        delayed = self._action_delay_buffer.pop(0)
        self._action_delay_buffer.append(action)
        delayed = np.asarray(delayed, dtype=np.float32).reshape(2).copy()
        delayed[0] = float(np.clip(delayed[0], 0.0, 1.0))
        delayed[1] = float(np.clip(delayed[1], -self.params.max_gimbal, self.params.max_gimbal))
        return delayed

    def terminal_success(self, scalars: EpisodeScalars) -> bool:
        # Success = reached target altitude without violating safety envelopes.
        # Trajectory-shape requirements (fpa, vx, downrange) are NOT checked here
        # because this env terminates at apogee where vz=0 → fpa=0° always, making
        # any fpa/vx criterion structurally impossible to satisfy.
        mission = self.mission
        fuel_used = (self.params.dry_mass + self.params.prop_mass - self.state.mass) / max(self.params.prop_mass, 1.0)
        min_altitude_m, max_altitude_m = self._curriculum_success_band()
        return bool(
            scalars.altitude >= min_altitude_m
            and (max_altitude_m is None or scalars.altitude <= max_altitude_m)
            and self.max_q_seen <= mission.max_q_pa
            and self.max_g_seen <= mission.max_g_load
            and fuel_used <= mission.max_fuel_fraction_used + self._FUEL_FRACTION_TOL
            and not self.crash
        )

    def terminal_error_cost(self, scalars: EpisodeScalars) -> float:
        mission = self.mission
        stage = self._mission_stage()
        min_altitude_m, max_altitude_m = self._curriculum_success_band()
        if scalars.altitude < min_altitude_m:
            alt_err = (min_altitude_m - scalars.altitude) / max(min_altitude_m, 1.0)
        elif max_altitude_m is not None and scalars.altitude > max_altitude_m:
            alt_err = (scalars.altitude - max_altitude_m) / max(max_altitude_m, 1.0)
        else:
            alt_err = 0.0
        downrange_err = abs(scalars.downrange - mission.target_downrange_m) / max(mission.max_terminal_downrange_m, 1.0)
        vx_target = 0.5 * (mission.min_terminal_vx_mps + mission.max_terminal_vx_mps)
        vx_err = abs(scalars.vx - vx_target) / max(vx_target, 1.0)
        vz_err = abs(scalars.vz) / max(mission.max_terminal_abs_vz_mps, 1.0)
        fpa_target = 0.5 * (mission.min_flight_path_angle_deg + mission.max_flight_path_angle_deg)
        fpa_err = abs(scalars.flight_path_angle_deg - fpa_target) / max(abs(fpa_target), 1.0)
        fuel_used = (self.params.dry_mass + self.params.prop_mass - self.state.mass) / max(self.params.prop_mass, 1.0)
        fuel_err = max(0.0, fuel_used - mission.max_fuel_fraction_used)
        q_err = self._q_over_limit_fraction()

        if stage == "low":
            return float(0.68 * alt_err + 0.18 * vz_err + 0.06 * q_err + 0.08 * fuel_err)
        if stage == "mid":
            return float(
                0.44 * alt_err
                + 0.10 * downrange_err
                + 0.08 * vx_err
                + 0.12 * vz_err
                + 0.06 * fpa_err
                + 0.15 * q_err
                + 0.05 * fuel_err
            )
        # At apogee termination vz=0 and fpa=0 structurally, so vx/vz/fpa/downrange
        # errors are constant noise — altitude must dominate the gradient.
        return float(
            0.85 * alt_err
            + 0.10 * q_err
            + 0.05 * fuel_err
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
        # vz_at_burnout determines apogee via vz²/2g; velocity_at_burnout is total speed magnitude
        velocity_at_burnout = self.burnout_scalars.speed if self.burnout_scalars is not None else np.nan
        vz_at_burnout = self.burnout_scalars.vz if self.burnout_scalars is not None else np.nan
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
            "velocity_at_burnout_mps": velocity_at_burnout,
            "vz_at_burnout_mps": vz_at_burnout,
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
            "q_margin_pa": self._q_margin_pa(),
            "q_over_limit_fraction": self._q_over_limit_fraction(),
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
                "area_ref": self.params.area_ref,
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
            "stress": {
                "observation_noise_std": self.observation_noise_std,
                "action_lag_steps": self.action_lag_steps,
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
