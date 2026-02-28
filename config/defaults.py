"""Default simulator and RL configuration values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sim.vehicle import VehicleParams


@dataclass(frozen=True)
class EnvConfig:
    dt: float = 0.05
    t_final: float = 300.0
    train_total_timesteps: int = 1_000_000


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 1_000_000
    n_envs: int = 4
    n_steps: int = 1024
    batch_size: int = 256
    gamma: float = 0.99
    gae_lambda: float = 0.95
    learning_rate: float = 3e-4
    ent_coef: float = 0.0
    clip_range: float = 0.2
    net_arch: tuple[int, int] = (256, 128)
    seed: int = 0
    eval_freq_steps: int = 100_000
    eval_episodes: int = 20
    checkpoint_freq_steps: int = 100_000


@dataclass(frozen=True)
class MissionConfig:
    # Falcon 9-like ascent target: breach atmosphere before considering in-space navigation tasks.
    space_boundary_altitude_m: float = 100_000.0
    min_terminal_altitude_m: float = 1_200.0
    max_terminal_altitude_m: float = 6_000.0
    min_terminal_vz_mps: float = 120.0
    max_terminal_vz_mps: float = 900.0
    min_terminal_vx_mps: float = 20.0
    max_terminal_vx_mps: float = 700.0
    min_flight_path_angle_deg: float = 8.0
    max_flight_path_angle_deg: float = 80.0
    max_q_pa: float = 40_000.0
    max_g_load: float = 5.5
    max_terminal_downrange_m: float = 10_000.0
    target_altitude_m: float = 2_500.0
    target_vx_mps: float = 250.0
    target_vz_mps: float = 450.0
    target_downrange_m: float = 1_500.0
    max_fuel_fraction_used: float = 1.0


@dataclass(frozen=True)
class AtmosphereConfig:
    randomize: bool = True
    density_scale_min: float = 0.90
    density_scale_max: float = 1.10
    scale_height_min_m: float = 7_500.0
    scale_height_max_m: float = 9_500.0
    wind_bias_x_min_mps: float = -25.0
    wind_bias_x_max_mps: float = 25.0
    wind_bias_y_min_mps: float = -15.0
    wind_bias_y_max_mps: float = 15.0
    wind_shear_x_min_mps: float = -45.0
    wind_shear_x_max_mps: float = 45.0
    wind_shear_y_min_mps: float = -25.0
    wind_shear_y_max_mps: float = 25.0
    gust_amp_x_min_mps: float = 0.0
    gust_amp_x_max_mps: float = 12.0
    gust_amp_y_min_mps: float = 0.0
    gust_amp_y_max_mps: float = 8.0
    gust_freq_x_hz: float = 0.08
    gust_freq_y_hz: float = 0.11


def default_vehicle_params() -> VehicleParams:
    return VehicleParams(
        max_thrust=1.5e6,
        isp=300.0,
        dry_mass=2.0e4,
        prop_mass=1.0e4,
        area_ref=10.0,
        cd=0.5,
        gimbal_arm=1.2,
        I_body=np.array([8e5, 8e5, 1e5], dtype=float),
        max_gimbal=np.deg2rad(5.0),
        aero_damp=1e4,
    )
