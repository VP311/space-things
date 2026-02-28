from __future__ import annotations

import unittest

import numpy as np

from sim.dynamics import step
from sim.vehicle import Control, RocketState, VehicleParams


class TestDynamics(unittest.TestCase):
    def setUp(self) -> None:
        self.params = VehicleParams(
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
        self.dt = 0.05

    def test_step_outputs_finite_and_normalized_quaternion(self) -> None:
        state = RocketState(
            pos=np.array([0.0, 0.0, 0.0], dtype=float),
            vel=np.array([0.0, 0.0, 0.0], dtype=float),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            omega=np.array([0.0, 0.0, 0.0], dtype=float),
            mass=3.0e4,
        )
        control = Control(throttle=0.7, gimbal_pitch=0.01, gimbal_yaw=-0.01)

        next_state = step(state, control, self.params, self.dt)

        self.assertTrue(np.all(np.isfinite(next_state.pos)))
        self.assertTrue(np.all(np.isfinite(next_state.vel)))
        self.assertTrue(np.all(np.isfinite(next_state.quat)))
        self.assertTrue(np.all(np.isfinite(next_state.omega)))
        self.assertTrue(np.isfinite(next_state.mass))
        self.assertAlmostEqual(float(np.linalg.norm(next_state.quat)), 1.0, places=10)

    def test_drag_opposes_velocity(self) -> None:
        state = RocketState(
            pos=np.array([0.0, 0.0, 1000.0], dtype=float),
            vel=np.array([100.0, 0.0, 0.0], dtype=float),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            omega=np.array([0.0, 0.0, 0.0], dtype=float),
            mass=3.0e4,
        )
        control = Control(throttle=0.0, gimbal_pitch=0.0, gimbal_yaw=0.0)

        next_state = step(state, control, self.params, self.dt)

        self.assertLess(next_state.vel[0], state.vel[0])

    def test_mass_decreases_with_thrust(self) -> None:
        state = RocketState(
            pos=np.array([0.0, 0.0, 0.0], dtype=float),
            vel=np.array([0.0, 0.0, 0.0], dtype=float),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            omega=np.array([0.0, 0.0, 0.0], dtype=float),
            mass=3.0e4,
        )
        control = Control(throttle=1.0, gimbal_pitch=0.0, gimbal_yaw=0.0)

        next_state = step(state, control, self.params, self.dt)

        self.assertLess(next_state.mass, state.mass)
        self.assertGreaterEqual(next_state.mass, self.params.dry_mass)

    def test_upward_acceleration_when_thrust_points_up(self) -> None:
        # -90 deg rotation about body/world +Y maps body +X to world +Z.
        angle = -np.pi / 2.0
        quat = np.array([np.cos(angle / 2.0), 0.0, np.sin(angle / 2.0), 0.0], dtype=float)
        state = RocketState(
            pos=np.array([0.0, 0.0, 0.0], dtype=float),
            vel=np.array([0.0, 0.0, 0.0], dtype=float),
            quat=quat,
            omega=np.array([0.0, 0.0, 0.0], dtype=float),
            mass=3.0e4,
        )
        control = Control(throttle=1.0, gimbal_pitch=0.0, gimbal_yaw=0.0)

        next_state = step(state, control, self.params, self.dt)

        self.assertGreater(next_state.vel[2], 0.0)

    def test_zero_gimbal_has_no_spurious_torque(self) -> None:
        state = RocketState(
            pos=np.array([0.0, 0.0, 0.0], dtype=float),
            vel=np.array([0.0, 0.0, 0.0], dtype=float),
            quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            omega=np.array([0.0, 0.0, 0.0], dtype=float),
            mass=3.0e4,
        )
        control = Control(throttle=1.0, gimbal_pitch=0.0, gimbal_yaw=0.0)

        next_state = step(state, control, self.params, self.dt)

        self.assertTrue(np.allclose(next_state.omega, np.zeros(3), atol=1e-9))


if __name__ == "__main__":
    unittest.main()
