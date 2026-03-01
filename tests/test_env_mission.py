from __future__ import annotations

import unittest

import numpy as np

from config.defaults import AtmosphereConfig, MissionConfig, default_vehicle_params
from rl.env import RocketAscentEnv
from sim.atmosphere import AtmosphereProfile
from sim.dynamics import aero_forces
from sim.vehicle import Control, RocketState


class TestEnvironmentMission(unittest.TestCase):
    def setUp(self) -> None:
        self.params = default_vehicle_params()

    def test_wind_relative_airspeed_reduces_drag(self) -> None:
        state = RocketState(
            pos=np.array([0.0, 0.0, 1000.0], dtype=float),
            vel=np.array([120.0, 0.0, 0.0], dtype=float),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            omega=np.array([0.0, 0.0, 0.0], dtype=float),
            mass=self.params.dry_mass + self.params.prop_mass,
        )
        control = Control(throttle=0.0, gimbal_pitch=0.0, gimbal_yaw=0.0)

        no_wind = AtmosphereProfile()
        strong_tailwind = AtmosphereProfile(wind_bias_x_mps=120.0)

        _, drag_no_wind, q_no_wind, _ = aero_forces(state, control, self.params, atmosphere=no_wind)
        _, drag_tailwind, q_tailwind, _ = aero_forces(state, control, self.params, atmosphere=strong_tailwind)

        self.assertGreater(q_no_wind, 0.0)
        self.assertLess(q_tailwind, 1e-6)
        self.assertLess(np.linalg.norm(drag_tailwind), np.linalg.norm(drag_no_wind))

    def test_atmosphere_randomization_reproducible_with_seed(self) -> None:
        mission = MissionConfig()
        atmo_cfg = AtmosphereConfig(randomize=True)

        env_a = RocketAscentEnv(self.params, mission=mission, atmosphere_cfg=atmo_cfg)
        env_b = RocketAscentEnv(self.params, mission=mission, atmosphere_cfg=atmo_cfg)

        env_a.reset(seed=123)
        env_b.reset(seed=123)

        self.assertAlmostEqual(env_a.atmosphere_profile.density_scale, env_b.atmosphere_profile.density_scale)
        self.assertAlmostEqual(env_a.atmosphere_profile.scale_height_m, env_b.atmosphere_profile.scale_height_m)
        self.assertAlmostEqual(env_a.atmosphere_profile.wind_bias_x_mps, env_b.atmosphere_profile.wind_bias_x_mps)
        self.assertAlmostEqual(env_a.atmosphere_profile.wind_bias_y_mps, env_b.atmosphere_profile.wind_bias_y_mps)

    def test_terminal_success_requires_space_boundary_and_constraints(self) -> None:
        mission = MissionConfig()
        env = RocketAscentEnv(
            self.params,
            mission=mission,
            atmosphere_cfg=AtmosphereConfig(randomize=False),
        )
        _, _ = env.reset(seed=7)

        env.state = RocketState(
            pos=np.array([mission.target_downrange_m, 0.0, mission.target_altitude_m + 100.0], dtype=float),
            vel=np.array([mission.target_vx_mps, 0.0, 40.0], dtype=float),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            omega=np.array([0.0, 0.0, 0.0], dtype=float),
            mass=self.params.dry_mass + 0.2 * self.params.prop_mass,
        )
        env.max_q_seen = mission.max_q_pa * 0.9
        env.max_g_seen = mission.max_g_load * 0.9

        scalars = env.scalars(env.state, g_load=1.0)
        self.assertTrue(env.terminal_success(scalars))

        env.max_q_seen = mission.max_q_pa * 1.05
        self.assertFalse(env.terminal_success(scalars))

        env.max_q_seen = mission.max_q_pa * 0.9
        env.state = RocketState(
            pos=np.array([mission.target_downrange_m, 0.0, mission.target_altitude_m - 1.0], dtype=float),
            vel=np.array([mission.target_vx_mps, 0.0, 40.0], dtype=float),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            omega=np.array([0.0, 0.0, 0.0], dtype=float),
            mass=self.params.dry_mass + 0.2 * self.params.prop_mass,
        )
        self.assertFalse(env.terminal_success(env.scalars(env.state, g_load=1.0)))

    def test_observation_vector_expanded(self) -> None:
        env = RocketAscentEnv(
            self.params,
            mission=MissionConfig(),
            atmosphere_cfg=AtmosphereConfig(randomize=False),
        )
        obs, _ = env.reset(seed=11)
        self.assertEqual(obs.shape, (18,))
        self.assertEqual(env.observation_space.shape, (18,))

    def test_burnout_transitions_to_coast_instead_of_immediate_terminate(self) -> None:
        mission = MissionConfig()
        env = RocketAscentEnv(
            self.params,
            mission=mission,
            atmosphere_cfg=AtmosphereConfig(randomize=False),
            min_coast_time_s=0.5,
        )
        _, _ = env.reset(seed=5)
        env.state = RocketState(
            pos=np.array([0.0, 0.0, 1_500.0], dtype=float),
            vel=np.array([150.0, 0.0, 40.0], dtype=float),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            omega=np.array([0.0, 0.0, 0.0], dtype=float),
            mass=self.params.dry_mass,
        )
        env.max_q_seen = 0.0
        env.max_g_seen = 0.0
        _, _, terminated, truncated, info = env.step(np.array([0.0, 0.0], dtype=np.float32))

        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["burnout"])
        self.assertEqual(info["phase"], "coast")


if __name__ == "__main__":
    unittest.main()
