from __future__ import annotations

import json
import inspect
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import rl.train as train_module
from config.defaults import MissionConfig, PPOConfig, default_vehicle_params
from rl.env import EpisodeScalars
from rl.policy_eval import build_eval_env, normalise_obs, run_episode
from rl.train import CurriculumManager, EpisodeSummaryCallback, FixedFinalEvalCallback, VecNormalizeCheckpointCallback
from sim.constants import G0


class DummyVecTrainingEnv:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, float]]] = []

    def env_method(self, name: str, **kwargs: float) -> None:
        self.calls.append((name, kwargs))


class DummySyncEnv:
    def __init__(self, target_altitude_m: float) -> None:
        self.curriculum_target_altitude_m = float(target_altitude_m)
        self.start_altitude_cap_m = 0.0

    def set_curriculum(self, target_altitude_m: float, start_altitude_cap_m: float | None = None) -> None:
        self.curriculum_target_altitude_m = float(target_altitude_m)
        self.start_altitude_cap_m = float(start_altitude_cap_m or 0.0)


class DummyLogger:
    def __init__(self) -> None:
        self.records: list[tuple[str, float]] = []

    def record(self, key: str, value: float) -> None:
        self.records.append((key, value))


class DummyModel:
    def __init__(self, training_env: object | None = None) -> None:
        self.saved_paths: list[str] = []
        self.logger = DummyLogger()
        self._training_env = training_env

    def save(self, path: str) -> None:
        self.saved_paths.append(path)

    def get_env(self) -> object | None:
        return self._training_env


class DummyVecNormalizeSave:
    def __init__(self) -> None:
        self.saved_paths: list[str] = []

    def save(self, path: str) -> None:
        self.saved_paths.append(path)

    def close(self) -> None:
        pass


class RecordingModel:
    def __init__(self, action: np.ndarray | None = None) -> None:
        self.action = np.array([0.0, 0.0], dtype=np.float32) if action is None else action
        self.seen_obs: list[np.ndarray] = []

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        _ = deterministic
        self.seen_obs.append(np.asarray(obs, dtype=np.float32).copy())
        return self.action, None


class FakeVecNorm:
    class _ObsRms:
        def __init__(self, mean: np.ndarray, var: np.ndarray) -> None:
            self.mean = mean
            self.var = var

    def __init__(self, mean: np.ndarray, var: np.ndarray, epsilon: float = 0.0, clip_obs: float = 10.0) -> None:
        self.obs_rms = self._ObsRms(mean=mean, var=var)
        self.epsilon = epsilon
        self.clip_obs = clip_obs


class TestCurriculumManager(unittest.TestCase):
    def test_high_stage_requires_success_and_altitude_ratio(self) -> None:
        manager = CurriculumManager(
            current_target_m=85_000.0,
            target_milestones_m=(3_000.0, 65_000.0, 85_000.0, 92_000.0, 96_000.0, 100_000.0),
            promotion_window_episodes=50,
            start_altitude_fraction=0.0,
            low_stage_success_threshold=0.45,
            high_stage_threshold_m=65_000.0,
            high_stage_success_threshold=0.70,
            high_stage_altitude_ratio_threshold=0.98,
        )

        for idx in range(49):
            promoted, target = manager.record(success=idx < 34, altitude_ratio=1.02, q_ok=True)
            self.assertFalse(promoted)
            self.assertEqual(target, 85_000.0)

        promoted, target = manager.record(success=False, altitude_ratio=1.02, q_ok=True)
        self.assertFalse(promoted)
        self.assertEqual(target, 85_000.0)
        self.assertAlmostEqual(manager.last_decision["rolling_q_ok"], 1.0)
        self.assertEqual(manager.last_decision["reason"], "success_below_threshold")

        promoted_any = False
        target = manager.current_target_m
        for _ in range(50):
            promoted, target = manager.record(success=True, altitude_ratio=0.98, q_ok=True)
            promoted_any = promoted_any or promoted
        self.assertTrue(promoted_any)
        self.assertEqual(target, 92_000.0)


class TestEpisodeSummaryCallback(unittest.TestCase):
    def test_promotion_updates_training_and_progress_env_only(self) -> None:
        curriculum = CurriculumManager(
            current_target_m=85_000.0,
            target_milestones_m=(3_000.0, 65_000.0, 85_000.0, 92_000.0, 96_000.0, 100_000.0),
            promotion_window_episodes=50,
            start_altitude_fraction=0.0,
            low_stage_success_threshold=0.45,
            high_stage_threshold_m=65_000.0,
            high_stage_success_threshold=0.70,
            high_stage_altitude_ratio_threshold=0.98,
        )
        progress_env = DummySyncEnv(85_000.0)
        fixed_final_env = DummySyncEnv(100_000.0)
        training_env = DummyVecTrainingEnv()

        with tempfile.TemporaryDirectory() as tmpdir:
            callback = EpisodeSummaryCallback(
                Path(tmpdir) / "episodes.jsonl",
                curriculum=curriculum,
                curriculum_sync_envs=[progress_env],
            )
            callback.init_callback(DummyModel(training_env))
            callback._on_training_start()
            try:
                callback.locals = {
                    "dones": [True],
                    "infos": [
                        {
                            "t": 1.0,
                            "max_altitude_m": 85_000.0 * 0.99,
                            "altitude_at_burnout_m": 24_000.0,
                            "velocity_at_burnout_mps": 1_200.0,
                            "vz_at_burnout_mps": 1_180.0,
                            "altitude_at_apogee_m": 90_000.0,
                            "time_to_apogee_s": 100.0,
                            "max_q_dyn": 60_000.0,
                            "q_margin_pa": 10_000.0,
                            "q_over_limit_fraction": 0.0,
                            "max_g_load": 6.0,
                            "fuel_used_fraction": 1.0,
                            "success": True,
                            "termination_reason": "apogee",
                            "mean_throttle": 0.25,
                            "throttle_variance": 0.1,
                            "gimbal_variance": 0.01,
                            "mission": {"target_altitude_m": 85_000.0},
                            "reward_components_cumulative": {},
                        }
                    ],
                }
                for _ in range(50):
                    callback._on_step()
            finally:
                callback._on_training_end()

            rows = [json.loads(line) for line in (Path(tmpdir) / "episodes.jsonl").read_text().splitlines() if line.strip()]

        self.assertTrue(training_env.calls)
        self.assertEqual(training_env.calls[-1][0], "set_curriculum")
        self.assertEqual(training_env.calls[-1][1]["target_altitude_m"], 92_000.0)
        self.assertEqual(progress_env.curriculum_target_altitude_m, 92_000.0)
        self.assertEqual(fixed_final_env.curriculum_target_altitude_m, 100_000.0)
        self.assertIn("curriculum_window", rows[-1])
        self.assertAlmostEqual(rows[-1]["q_margin_pa"], 10_000.0)
        self.assertAlmostEqual(rows[-1]["curriculum_window"]["rolling_q_ok"], 1.0)


class TestBenchmarkCallback(unittest.TestCase):
    def test_best_final_selection_uses_fixed_final_metrics(self) -> None:
        class ScriptedCallback(FixedFinalEvalCallback):
            def __init__(self) -> None:
                self._tmpdir = tempfile.TemporaryDirectory()
                base = Path(self._tmpdir.name)
                super().__init__(
                    progress_env=DummySyncEnv(85_000.0),  # type: ignore[arg-type]
                    fixed_final_env=DummySyncEnv(100_000.0),  # type: ignore[arg-type]
                    eval_freq_steps=100,
                    eval_episodes=2,
                    final_eval_seed_start=2000,
                    official_eval_freq_steps=500,
                    official_eval_episodes=2,
                    official_eval_seed_start=3000,
                    official_success_threshold=0.8,
                    official_success_streak=2,
                    best_model_path=base / "best_final_model.zip",
                    best_vecnorm_path=base / "best_final_vecnormalize.pkl",
                    progress_latest_path=base / "progress_latest.json",
                    fixed_final_latest_path=base / "fixed_latest.json",
                    official_latest_path=base / "official_latest.json",
                    progress_history_path=base / "progress_history.jsonl",
                    fixed_final_history_path=base / "fixed_history.jsonl",
                    official_history_path=base / "official_history.jsonl",
                )
                self.results = iter(
                    [
                        {"success_rate": 0.0, "q_ok_rate": 1.0, "avg_max_altitude_m": 90_000.0, "avg_burnout_vz_mps": 1_150.0},
                        {"success_rate": 0.2, "q_ok_rate": 1.0, "avg_max_altitude_m": 80_000.0, "avg_burnout_vz_mps": 1_160.0},
                        {"success_rate": 0.0, "q_ok_rate": 1.0, "avg_max_altitude_m": 95_000.0, "avg_burnout_vz_mps": 1_180.0},
                        {"success_rate": 0.1, "q_ok_rate": 1.0, "avg_max_altitude_m": 99_000.0, "avg_burnout_vz_mps": 1_190.0},
                    ]
                )

            def _evaluate_raw_env(self, env: DummySyncEnv, *, n_episodes: int, seed_start: int) -> dict[str, float]:
                _ = env, n_episodes, seed_start
                return dict(next(self.results))

        callback = ScriptedCallback()
        training_env = DummyVecNormalizeSave()
        callback.init_callback(DummyModel(training_env))
        callback._on_training_start()
        callback.num_timesteps = 100
        self.assertTrue(callback._on_step())
        callback.num_timesteps = 200
        self.assertTrue(callback._on_step())

        self.assertEqual(len(callback.model.saved_paths), 1)
        self.assertEqual(len(training_env.saved_paths), 1)
        self.assertAlmostEqual(callback._best_success_rate, 0.2)
        callback._tmpdir.cleanup()

    def test_fixed_final_regression_guard_stops_training(self) -> None:
        class ScriptedRegressionCallback(FixedFinalEvalCallback):
            def __init__(self) -> None:
                self._tmpdir = tempfile.TemporaryDirectory()
                base = Path(self._tmpdir.name)
                super().__init__(
                    progress_env=DummySyncEnv(85_000.0),  # type: ignore[arg-type]
                    fixed_final_env=DummySyncEnv(100_000.0),  # type: ignore[arg-type]
                    eval_freq_steps=100,
                    eval_episodes=2,
                    final_eval_seed_start=2000,
                    official_eval_freq_steps=0,
                    official_eval_episodes=2,
                    official_eval_seed_start=3000,
                    official_success_threshold=0.8,
                    official_success_streak=2,
                    best_model_path=base / "best_final_model.zip",
                    best_vecnorm_path=base / "best_final_vecnormalize.pkl",
                    progress_latest_path=base / "progress_latest.json",
                    fixed_final_latest_path=base / "fixed_latest.json",
                    official_latest_path=base / "official_latest.json",
                    progress_history_path=base / "progress_history.jsonl",
                    fixed_final_history_path=base / "fixed_history.jsonl",
                    official_history_path=base / "official_history.jsonl",
                    guardrail_min_success_rate=0.5,
                    guardrail_min_q_ok_rate=0.95,
                )
                self.results = iter(
                    [
                        {"success_rate": 0.9, "q_ok_rate": 1.0, "avg_max_altitude_m": 99_000.0, "avg_burnout_vz_mps": 1_200.0},
                        {"success_rate": 0.6, "q_ok_rate": 1.0, "avg_max_altitude_m": 101_000.0, "avg_burnout_vz_mps": 1_220.0},
                        {"success_rate": 0.8, "q_ok_rate": 1.0, "avg_max_altitude_m": 98_000.0, "avg_burnout_vz_mps": 1_180.0},
                        {"success_rate": 0.2, "q_ok_rate": 0.9, "avg_max_altitude_m": 90_000.0, "avg_burnout_vz_mps": 1_100.0},
                    ]
                )

            def _evaluate_raw_env(self, env: DummySyncEnv, *, n_episodes: int, seed_start: int) -> dict[str, float]:
                _ = env, n_episodes, seed_start
                return dict(next(self.results))

        callback = ScriptedRegressionCallback()
        try:
            callback.init_callback(DummyModel(DummyVecNormalizeSave()))
            callback._on_training_start()
            callback.num_timesteps = 100
            self.assertTrue(callback._on_step())
            callback.num_timesteps = 200
            self.assertFalse(callback._on_step())
        finally:
            callback._tmpdir.cleanup()


class TestTrainFineTuneControls(unittest.TestCase):
    def test_train_exposes_runtime_lr_overrides(self) -> None:
        signature = inspect.signature(train_module.train)
        self.assertIn("lr_start", signature.parameters)
        self.assertIn("lr_end", signature.parameters)
        self.assertIn("anchor_ref_model_path", signature.parameters)
        self.assertIn("anchor_coef", signature.parameters)

    def test_cli_accepts_lr_override_flags(self) -> None:
        with patch(
            "sys.argv",
            [
                "train.py",
                "--lr-start",
                "3e-5",
                "--lr-end",
                "1e-6",
                "--ent-coef-end",
                "0",
                "--anchor-ref-model",
                "ref.zip",
                "--anchor-coef",
                "0.2",
            ],
        ):
            args = train_module._parse_args()

        self.assertAlmostEqual(args.lr_start, 3e-5)
        self.assertAlmostEqual(args.lr_end, 1e-6)
        self.assertAlmostEqual(args.ent_coef_end, 0.0)
        self.assertEqual(args.anchor_ref_model, "ref.zip")
        self.assertAlmostEqual(args.anchor_coef, 0.2)

    def test_warm_start_applies_runtime_ppo_overrides(self) -> None:
        class LoadedModel:
            def __init__(self) -> None:
                self.saved_paths: list[str] = []
                self.learn_calls: list[dict[str, object]] = []

            def learn(self, *, total_timesteps: int, callback: object) -> None:
                self.learn_calls.append({"total_timesteps": total_timesteps, "callback": callback})

            def save(self, path: str) -> None:
                self.saved_paths.append(path)

        model = LoadedModel()
        vec_env = DummyVecNormalizeSave()
        raw_env = object()
        with tempfile.TemporaryDirectory() as tmpdir:
            out_dir = Path(tmpdir) / "artifacts"
            with (
                patch.object(train_module, "make_vec_env", return_value=raw_env),
                patch.object(train_module.VecNormalize, "load", return_value=vec_env),
                patch.object(train_module.PPO, "load", return_value=model),
            ):
                train_module.train(
                    total_timesteps=10,
                    n_envs=1,
                    initial_target_altitude_m=100_000.0,
                    curriculum_enabled=False,
                    ent_coef=0.0,
                    ent_coef_end=0.0,
                    lr_start=2e-5,
                    lr_end=2e-6,
                    clip_range=0.05,
                    target_kl=0.01,
                    eval_freq_steps=0,
                    official_eval_freq_steps=0,
                    checkpoint_freq_steps=10,
                    load_model_path="model.zip",
                    load_vecnorm_path="vec.pkl",
                    artifacts_dir=str(out_dir),
                    use_lstm=False,
                )

            run_config = json.loads((out_dir / "train_run_config.json").read_text())

        self.assertEqual(model.learn_calls[0]["total_timesteps"], 10)
        self.assertAlmostEqual(model.lr_schedule(1.0), 2e-5)
        self.assertAlmostEqual(model.lr_schedule(0.0), 2e-6)
        self.assertAlmostEqual(model.clip_range(1.0), 0.05)
        self.assertAlmostEqual(model.target_kl, 0.01)
        self.assertAlmostEqual(run_config["ppo"]["lr_start"], 2e-5)
        self.assertAlmostEqual(run_config["ppo"]["lr_end"], 2e-6)
        self.assertAlmostEqual(run_config["ppo"]["ent_coef_start"], 0.0)
        self.assertAlmostEqual(run_config["ppo"]["ent_coef_end"], 0.0)
        self.assertAlmostEqual(run_config["ppo"]["clip_range"], 0.05)
        self.assertAlmostEqual(run_config["ppo"]["target_kl"], 0.01)
        self.assertIsNone(run_config["ppo"]["anchor_ref_model_path"])
        self.assertAlmostEqual(run_config["ppo"]["anchor_coef"], 0.0)


class TestVecNormalizeCheckpoints(unittest.TestCase):
    def test_checkpoint_callback_saves_matching_vecnormalize_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            training_env = DummyVecNormalizeSave()
            callback = VecNormalizeCheckpointCallback(save_freq=1, save_path=Path(tmpdir), name_prefix="ppo_rocket")
            callback.init_callback(DummyModel(training_env))
            callback._on_training_start()
            callback.n_calls = 1
            callback.num_timesteps = 123_456

            self.assertTrue(callback._on_step())

        self.assertEqual(callback.model.saved_paths, [str(Path(tmpdir) / "ppo_rocket_123456_steps")])
        self.assertEqual(training_env.saved_paths, [str(Path(tmpdir) / "vecnormalize_123456_steps.pkl")])


class TestPolicyEvalHelpers(unittest.TestCase):
    def test_run_episode_applies_vecnorm_and_respects_target_altitude(self) -> None:
        target_altitude_m = 12_345.0
        env = build_eval_env(target_altitude_m, start_state_randomization=False)
        env.t_final = env.dt
        expected_env = build_eval_env(target_altitude_m, start_state_randomization=False)
        expected_env.t_final = expected_env.dt
        raw_obs, _ = expected_env.reset(seed=7)

        vecnorm = FakeVecNorm(
            mean=np.full(raw_obs.shape, 0.5, dtype=np.float32),
            var=np.full(raw_obs.shape, 4.0, dtype=np.float32),
        )
        model = RecordingModel()
        row = run_episode(env, mode="trained", seed=7, model=model, vecnorm=vecnorm)

        self.assertEqual(row["target_altitude_m"], target_altitude_m)
        self.assertIn("q_margin_pa", row)
        self.assertIn("q_over_limit_fraction", row)
        self.assertTrue(model.seen_obs)
        np.testing.assert_allclose(model.seen_obs[0], normalise_obs(raw_obs, vecnorm), atol=1e-6)

    def test_run_eval_honors_seed_start_and_skip_baseline(self) -> None:
        from rl import eval as eval_module

        def fake_evaluate_policy(**kwargs: object) -> dict[str, object]:
            return {
                "mode": kwargs["mode"],
                "n_episodes": kwargs["n_episodes"],
                "target_altitude_m": kwargs["target_altitude_m"],
                "success_rate": 1.0,
                "q_ok_rate": 1.0,
                "g_ok_rate": 1.0,
                "avg_max_altitude_m": 101_000.0,
                "avg_max_q": 50_000.0,
                "avg_burnout_vz_mps": 1_300.0,
                "episodes": [{"seed": kwargs["seed_start"]}],
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.zip"
            vecnorm_path = Path(tmpdir) / "vec.pkl"
            out_path = Path(tmpdir) / "eval.json"
            model_path.write_bytes(b"model")
            vecnorm_path.write_bytes(b"vec")
            with patch.object(eval_module.PPO, "load", return_value=object()), patch.object(
                eval_module, "load_vecnormalize_for_eval", return_value=object()
            ), patch.object(eval_module, "evaluate_policy", side_effect=fake_evaluate_policy) as eval_mock:
                result = eval_module.run_eval(
                    model_path=str(model_path),
                    vecnorm_path=str(vecnorm_path),
                    out_path=str(out_path),
                    n_episodes=3,
                    seed_start=12_000,
                    skip_baseline=True,
                    deterministic=False,
                )

        self.assertNotIn("baseline", result)
        self.assertEqual(result["eval"]["seed_start"], 12_000)
        self.assertEqual(result["eval"]["seed_end"], 12_002)
        self.assertFalse(result["eval"]["deterministic"])
        self.assertEqual(result["trained"]["episodes"][0]["seed"], 12_000)
        self.assertEqual(eval_mock.call_args.kwargs["seed_start"], 12_000)


class TestEnvRewardAndSuccessBands(unittest.TestCase):
    def _make_scalars(
        self,
        altitude_m: float,
        q_dyn_pa: float = 20_000.0,
        *,
        vz_mps: float = 0.0,
        speed_mps: float = 500.0,
    ) -> EpisodeScalars:
        return EpisodeScalars(
            altitude=float(altitude_m),
            speed=float(max(speed_mps, abs(vz_mps))),
            q_dyn=float(q_dyn_pa),
            vx=0.0,
            vy=0.0,
            vz=float(vz_mps),
            downrange=0.0,
            flight_path_angle_deg=0.0,
            wind_x=0.0,
            wind_y=0.0,
            g_load=4.0,
            rho_ratio=1.0,
        )

    def _burnout_scalars_for_projected_apogee(
        self,
        projected_apogee_m: float,
        *,
        vz_mps: float | None = None,
    ) -> EpisodeScalars:
        if vz_mps is None:
            burnout_altitude_m = 25_000.0
            vz_mps = float(np.sqrt(max(0.0, 2.0 * G0 * (projected_apogee_m - burnout_altitude_m))))
        else:
            burnout_altitude_m = projected_apogee_m - (vz_mps**2) / (2.0 * G0)
        return self._make_scalars(burnout_altitude_m, vz_mps=vz_mps, speed_mps=vz_mps)

    def _burnout_bonus(
        self,
        projected_apogee_m: float,
        *,
        vz_mps: float | None = None,
        max_q_seen: float = 60_000.0,
        max_g_seen: float = 6.0,
    ) -> float:
        env = build_eval_env(100_000.0, start_state_randomization=False)
        env.reset(seed=0)
        env.burnout = True
        env.burnout_scalars = self._burnout_scalars_for_projected_apogee(projected_apogee_m, vz_mps=vz_mps)
        env.max_q_seen = max_q_seen
        env.max_g_seen = max_g_seen
        return env._burnout_energy_bonus()

    def test_low_stage_compliant_overshoot_counts_as_success(self) -> None:
        env = build_eval_env(10_000.0, start_state_randomization=False)
        env.reset(seed=0)
        env.max_q_seen = 60_000.0
        env.max_g_seen = 6.0
        env.crash = False

        self.assertTrue(env.terminal_success(self._make_scalars(13_000.0)))
        self.assertTrue(env.terminal_success(self._make_scalars(12_000.0)))

    def test_success_band_has_no_upper_ceiling(self) -> None:
        # Overshoot is fine — policy should not be penalized for flying higher than target.
        env = build_eval_env(20_000.0, start_state_randomization=False)
        env.reset(seed=0)
        env.max_q_seen = 60_000.0
        env.max_g_seen = 6.0
        env.crash = False

        self.assertTrue(env.terminal_success(self._make_scalars(20_000.0)))
        self.assertTrue(env.terminal_success(self._make_scalars(80_000.0)))

    def test_final_stage_remains_one_sided(self) -> None:
        env = build_eval_env(100_000.0, start_state_randomization=False)
        env.reset(seed=0)
        env.max_q_seen = 60_000.0
        env.max_g_seen = 6.0
        env.crash = False

        self.assertTrue(env.terminal_success(self._make_scalars(102_000.0)))

    def test_q_barrier_penalty_is_zero_below_soft_limit_and_steeper_over_limit(self) -> None:
        env = build_eval_env(20_000.0, start_state_randomization=False)
        soft_limit = 0.85 * env.mission.max_q_pa

        self.assertEqual(env._q_barrier_penalty(soft_limit - 1.0), 0.0)
        near_limit_pen = env._q_barrier_penalty(0.95 * env.mission.max_q_pa)
        over_limit_pen = env._q_barrier_penalty(1.10 * env.mission.max_q_pa)
        self.assertGreater(near_limit_pen, 0.0)
        self.assertGreater(over_limit_pen, near_limit_pen)

    def test_high_stage_burnout_bonus_is_zero_below_final_hinge(self) -> None:
        self.assertEqual(self._burnout_bonus(91_000.0), 0.0)

    def test_high_stage_burnout_bonus_strongly_prefers_100km_over_95km(self) -> None:
        bonus_95km = self._burnout_bonus(95_000.0)
        bonus_100km = self._burnout_bonus(100_000.0)

        self.assertGreater(bonus_100km, 4.0 * bonus_95km)

    def test_high_stage_burnout_bonus_rewards_1200_mps_burnout_vz(self) -> None:
        bonus_1180 = self._burnout_bonus(100_000.0, vz_mps=1_180.0)
        bonus_1200 = self._burnout_bonus(100_000.0, vz_mps=1_200.0)

        self.assertGreater(bonus_1200, bonus_1180)

    def test_high_stage_burnout_bonus_is_suppressed_by_q_violation(self) -> None:
        compliant_bonus = self._burnout_bonus(100_000.0, max_q_seen=60_000.0)
        violating_bonus = self._burnout_bonus(100_000.0, max_q_seen=80_000.0)

        self.assertGreater(compliant_bonus, 0.0)
        self.assertEqual(violating_bonus, 0.0)

    def test_final_stage_failure_penalty_scales_over_last_20km(self) -> None:
        env = build_eval_env(100_000.0, start_state_randomization=False)
        env.reset(seed=0)

        penalty_95km = env._terminal_failure_penalty(self._make_scalars(95_000.0), terminal_penalty=0.0)
        penalty_85km = env._terminal_failure_penalty(self._make_scalars(85_000.0), terminal_penalty=0.0)

        self.assertGreater(penalty_85km, penalty_95km)
        self.assertAlmostEqual(penalty_85km, 82.5)

    def test_low_stage_progress_potential_saturates_after_target_hit(self) -> None:
        env = build_eval_env(10_000.0, start_state_randomization=False)
        low = env._progress_potential(self._make_scalars(9_800.0), coast_phase=True)
        hit = env._progress_potential(self._make_scalars(10_500.0), coast_phase=True)
        overshoot = env._progress_potential(self._make_scalars(25_000.0), coast_phase=True)

        self.assertLess(low, hit)
        self.assertAlmostEqual(hit, overshoot)

    def test_progress_potential_is_monotonic_in_altitude(self) -> None:
        env = build_eval_env(100_000.0, start_state_randomization=False)
        low = env._progress_potential(self._make_scalars(50_000.0), coast_phase=True)
        mid = env._progress_potential(self._make_scalars(80_000.0), coast_phase=True)
        high = env._progress_potential(self._make_scalars(100_000.0), coast_phase=True)

        self.assertGreater(mid, low)
        self.assertGreater(high, mid)

    def test_progress_potential_bonus_kicks_in_above_80_pct(self) -> None:
        env = build_eval_env(100_000.0, start_state_randomization=False)
        at_79 = env._progress_potential(self._make_scalars(79_000.0), coast_phase=True)
        at_81 = env._progress_potential(self._make_scalars(81_000.0), coast_phase=True)
        # Marginal gain from 79→81km should be larger than 2% because bonus kicks in at 80%
        marginal = at_81 - at_79
        linear_marginal = 2_000.0 / 100_000.0  # what pure linear would give
        self.assertGreater(marginal, linear_marginal)

    def test_burnout_bonus_is_zero_below_high_stage(self) -> None:
        env = build_eval_env(10_000.0, start_state_randomization=False)
        env.reset(seed=0)
        env.burnout = True
        env.burnout_scalars = self._make_scalars(15_000.0)
        bonus = env._burnout_energy_bonus()

        self.assertEqual(bonus, 0.0)

    def test_mid_stage_burnout_bonus_is_positive_when_compliant(self) -> None:
        env = build_eval_env(20_000.0, start_state_randomization=False)
        env.reset(seed=0)
        env.burnout = True
        env.burnout_scalars = EpisodeScalars(
            altitude=10_000.0,
            speed=550.0,
            q_dyn=25_000.0,
            vx=0.0,
            vy=0.0,
            vz=540.0,
            downrange=0.0,
            flight_path_angle_deg=0.0,
            wind_x=0.0,
            wind_y=0.0,
            g_load=4.0,
            rho_ratio=1.0,
        )
        env.max_q_seen = 50_000.0
        env.max_g_seen = 6.0

        self.assertGreater(env._burnout_energy_bonus(), 0.0)

    def test_mid_stage_burnout_bonus_is_suppressed_by_q_violation(self) -> None:
        env = build_eval_env(20_000.0, start_state_randomization=False)
        env.reset(seed=0)
        env.burnout = True
        env.burnout_scalars = EpisodeScalars(
            altitude=10_000.0,
            speed=550.0,
            q_dyn=25_000.0,
            vx=0.0,
            vy=0.0,
            vz=540.0,
            downrange=0.0,
            flight_path_angle_deg=0.0,
            wind_x=0.0,
            wind_y=0.0,
            g_load=4.0,
            rho_ratio=1.0,
        )
        env.max_q_seen = 80_000.0
        env.max_g_seen = 6.0

        self.assertEqual(env._burnout_energy_bonus(), 0.0)

    def test_reward_component_keys_include_success_and_burnout_bonus(self) -> None:
        env = build_eval_env(20_000.0, start_state_randomization=False)
        keys = env._empty_reward_sums().keys()

        self.assertIn("success_bonus", keys)
        self.assertIn("burnout_bonus", keys)


class TestDefaults(unittest.TestCase):
    def test_default_ent_coef_schedule(self) -> None:
        self.assertAlmostEqual(PPOConfig().ent_coef_start, 1e-3)
        self.assertAlmostEqual(PPOConfig().ent_coef_end, 1e-4)

    def test_default_vehicle_uses_solvable_propellant_mass(self) -> None:
        self.assertEqual(default_vehicle_params().prop_mass, 25_000.0)


class TestEvalReplayWiring(unittest.TestCase):
    def test_run_eval_uses_explicit_target_and_vecnorm(self) -> None:
        from rl import eval as eval_module

        fake_vecnorm = object()

        def fake_evaluate_policy(*, mode: str, target_altitude_m: float, vecnorm=None, **_: object) -> dict[str, object]:
            if mode == "baseline":
                self.assertEqual(target_altitude_m, 43_210.0)
                self.assertIsNone(vecnorm)
            else:
                self.assertEqual(target_altitude_m, 43_210.0)
                self.assertIs(vecnorm, fake_vecnorm)
            return {
                "success_rate": 0.0,
                "q_ok_rate": 0.0,
                "avg_max_altitude_m": 0.0,
                "avg_burnout_vz_mps": 0.0,
                "avg_max_q": 0.0,
                "avg_max_g_load": 0.0,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "eval.json"
            model_path = Path(tmpdir) / "model.zip"
            model_path.write_bytes(b"")
            with patch.object(eval_module, "resolve_model_path", return_value=str(model_path)), patch.object(
                eval_module, "resolve_vecnorm_path", return_value="vecnorm.pkl"
            ), patch.object(eval_module, "load_vecnormalize_for_eval", return_value=fake_vecnorm), patch.object(
                eval_module, "evaluate_policy", side_effect=fake_evaluate_policy
            ), patch.object(eval_module.PPO, "load", return_value=object()):
                results = eval_module.run_eval(
                    model_path="model.zip",
                    vecnorm_path="vecnorm.pkl",
                    out_path=str(out_path),
                    n_episodes=1,
                    target_altitude_m=43_210.0,
                )
            self.assertIn("trained", results)
            self.assertTrue(out_path.exists())

    def test_replay_uses_target_altitude_and_vecnorm(self) -> None:
        from rl import replay as replay_module

        with tempfile.TemporaryDirectory() as tmpdir:
            telemetry_path = Path(tmpdir) / "telemetry.npz"
            metrics_path = Path(tmpdir) / "metrics.json"
            env = build_eval_env(54_321.0, record=True, start_state_randomization=False)
            env.t_final = env.dt
            vecnorm = FakeVecNorm(mean=np.zeros(20, dtype=np.float32), var=np.ones(20, dtype=np.float32))
            model = RecordingModel()

            with patch.object(replay_module, "build_eval_env", return_value=env) as build_env_mock, patch.object(
                replay_module, "resolve_model_path", return_value="model.zip"
            ), patch.object(replay_module, "resolve_vecnorm_path", return_value="vecnorm.pkl"), patch.object(
                replay_module, "load_vecnormalize_for_eval", return_value=vecnorm
            ), patch.object(replay_module.PPO, "load", return_value=model):
                replay_module.replay(
                    model_path="model.zip",
                    vecnorm_path="vecnorm.pkl",
                    telemetry_path=str(telemetry_path),
                    metrics_path=str(metrics_path),
                    target_altitude_m=54_321.0,
                )

            build_env_mock.assert_called_once()
            self.assertTrue(model.seen_obs)
            metrics = json.loads(metrics_path.read_text())
            self.assertEqual(metrics["target_altitude_m"], 54_321.0)


class TestExpertTrajectorySearch(unittest.TestCase):
    def test_cem_parameter_decoding_produces_bounded_schedule(self) -> None:
        from rl import expert_search

        max_gimbal = 0.1
        params = np.linspace(-3.0, 3.0, 2 * len(expert_search.DEFAULT_KNOT_TIMES_S), dtype=np.float32)
        _, throttle, gimbal = expert_search.decode_schedule(params, max_gimbal=max_gimbal)

        self.assertTrue(np.all(throttle >= 0.0))
        self.assertTrue(np.all(throttle <= 1.0))
        self.assertTrue(np.all(gimbal >= -max_gimbal))
        self.assertTrue(np.all(gimbal <= max_gimbal))

    def test_expert_rollout_records_observation_action_pairs(self) -> None:
        from rl import expert_search

        env = expert_search.build_search_env(deterministic=True, record=False)
        env.t_final = env.dt
        with patch.object(expert_search, "build_search_env", return_value=env):
            rollout = expert_search.rollout_schedule(expert_search.initial_mean(), seed=0, deterministic=True)

        self.assertEqual(rollout.observations.shape[0], rollout.actions.shape[0])
        self.assertEqual(rollout.observations.shape[1], 20)
        self.assertEqual(rollout.actions.shape[1], 2)
        self.assertIn("fitness", rollout.metrics)

    def test_fitness_ranks_compliant_success_above_shortfall_and_noncompliant(self) -> None:
        from rl import expert_search

        success = {
            "success": True,
            "max_altitude_m": 101_000.0,
            "projected_burnout_apogee_m": 101_000.0,
            "vz_at_burnout_mps": 1_220.0,
            "max_q_dyn": 60_000.0,
            "max_g_load": 6.0,
            "q_ok": True,
            "g_ok": True,
        }
        shortfall = dict(success, success=False, max_altitude_m=92_000.0, projected_burnout_apogee_m=95_000.0)
        noncompliant = dict(success, success=False, max_q_dyn=95_000.0, q_ok=False)

        self.assertGreater(expert_search.score_metrics(success), expert_search.score_metrics(shortfall))
        self.assertGreater(expert_search.score_metrics(success), expert_search.score_metrics(noncompliant))


class TestBehaviorClone(unittest.TestCase):
    def test_dataset_loader_validates_required_shapes(self) -> None:
        from rl.behavior_clone import load_expert_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trajectories.npz"
            np.savez_compressed(
                path,
                observations=np.zeros((4, 20), dtype=np.float32),
                actions=np.zeros((4, 2), dtype=np.float32),
                episode_ids=np.zeros(4, dtype=np.int32),
            )
            dataset = load_expert_dataset(path)
            self.assertEqual(dataset["observations"].shape, (4, 20))
            self.assertEqual(dataset["actions"].shape, (4, 2))

            bad_path = Path(tmpdir) / "bad.npz"
            np.savez_compressed(bad_path, observations=np.zeros((4, 19), dtype=np.float32), actions=np.zeros((4, 2), dtype=np.float32))
            with self.assertRaises(ValueError):
                load_expert_dataset(bad_path)

    def test_dataset_loader_accepts_sample_weights(self) -> None:
        from rl.behavior_clone import apply_early_flight_weights, load_expert_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "weighted.npz"
            observations = np.zeros((2, 20), dtype=np.float32)
            observations[1, 0] = 0.2
            np.savez_compressed(
                path,
                observations=observations,
                actions=np.zeros((2, 2), dtype=np.float32),
                episode_ids=np.zeros(2, dtype=np.int32),
                sample_weights=np.array([2.0, 3.0], dtype=np.float32),
            )
            dataset = load_expert_dataset(path)
            weights = apply_early_flight_weights(
                dataset["observations"],
                dataset["sample_weights"],
                early_weight=4.0,
                early_altitude_m=8_000.0,
            )

        np.testing.assert_allclose(dataset["sample_weights"], np.array([2.0, 3.0], dtype=np.float32))
        self.assertGreater(weights[0], dataset["sample_weights"][0])
        self.assertEqual(weights[1], dataset["sample_weights"][1])

    def test_dagger_dataset_combines_expert_and_relabels(self) -> None:
        from rl import dagger as dagger_module

        expert = {
            "observations": np.zeros((2, 20), dtype=np.float32),
            "actions": np.ones((2, 2), dtype=np.float32),
            "episode_ids": np.zeros(2, dtype=np.int32),
            "sample_weights": np.ones(2, dtype=np.float32),
        }
        rollout_obs = [
            np.full((3, 20), 0.5, dtype=np.float32),
        ]
        rollout_actions = [
            np.full((3, 2), 0.25, dtype=np.float32),
        ]
        rollout_ids = [
            np.zeros(3, dtype=np.int32),
        ]
        metrics = [{"success": False, "max_altitude_m": 1.0}]

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = dagger_module._save_combined_dataset(
                expert=expert,
                dagger_obs=rollout_obs,
                dagger_actions=rollout_actions,
                dagger_episode_ids=rollout_ids,
                metrics=metrics,
                out_dir=Path(tmpdir),
                expert_dataset_path="expert.npz",
                model_path="model.zip",
                vecnorm_path="vec.pkl",
                episodes=1,
                dagger_weight=3.0,
                expert_weight=1.0,
            )
            with np.load(summary["dataset_path"], allow_pickle=False) as data:
                observations = data["observations"]
                actions = data["actions"]
                weights = data["sample_weights"]

        self.assertEqual(observations.shape, (5, 20))
        self.assertEqual(actions.shape, (5, 2))
        np.testing.assert_allclose(weights, np.array([1.0, 1.0, 3.0, 3.0, 3.0], dtype=np.float32))


class TestScriptedExpertExport(unittest.TestCase):
    def test_qbucket_export_saves_only_successful_matching_observation_action_pairs(self) -> None:
        from rl.behavior_clone import load_expert_dataset
        from rl.scripted_expert import export_scripted_experts

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = export_scripted_experts(
                out_dir=tmpdir,
                seed=11,
                q_caps_pa=(45_000.0,),
                buckets=(0.4,),
                until_alt_m=60_000.0,
            )
            dataset = load_expert_dataset(Path(tmpdir) / "trajectories.npz")
            with np.load(Path(tmpdir) / "trajectories.npz", allow_pickle=False) as data:
                metrics = [json.loads(item) for item in data["metrics_json"]]

        self.assertEqual(summary["n_successful"], 1)
        self.assertEqual(dataset["observations"].shape[0], dataset["actions"].shape[0])
        self.assertEqual(dataset["observations"].shape[1], 20)
        self.assertEqual(dataset["actions"].shape[1], 2)
        self.assertTrue(metrics[0]["success"])
        self.assertTrue(metrics[0]["q_ok"])
        self.assertTrue(metrics[0]["g_ok"])


class TestRobustnessStressEval(unittest.TestCase):
    def test_stress_case_decoding_applies_vehicle_and_env_overrides(self) -> None:
        from rl.stress_eval import StressCase, build_stress_env

        base = default_vehicle_params()
        case = StressCase(
            name="test",
            description="test",
            vehicle_scales={"cd": 1.2, "isp": 0.85, "dry_mass": 1.1, "max_thrust": 0.9},
            atmosphere_scales={"wind": 2.0},
            observation_noise_std=0.03,
            action_lag_steps=2,
            domain_randomization=False,
        )

        env = build_stress_env(case)

        self.assertAlmostEqual(env.base_params.cd, base.cd * 1.2)
        self.assertAlmostEqual(env.base_params.isp, base.isp * 0.85)
        self.assertAlmostEqual(env.base_params.dry_mass, base.dry_mass * 1.1)
        self.assertAlmostEqual(env.base_params.max_thrust, base.max_thrust * 0.9)
        self.assertAlmostEqual(env.atmosphere_cfg.wind_bias_x_max_mps, 50.0)
        self.assertAlmostEqual(env.observation_noise_std, 0.03)
        self.assertEqual(env.action_lag_steps, 2)
        self.assertFalse(env.domain_randomization)

    def test_stress_eval_writes_config_seed_range_and_case_metrics(self) -> None:
        import rl.stress_eval as stress_eval

        class DummyModel:
            pass

        class DummyVecnorm:
            def close(self) -> None:
                pass

        def fake_evaluate_case(case, **kwargs):
            return {
                "case": {"name": case.name},
                "trained": {
                    "n_episodes": kwargs["n_episodes"],
                    "success_rate": 1.0,
                    "q_ok_rate": 1.0,
                    "g_ok_rate": 1.0,
                    "episodes": [{"seed": kwargs["seed_start"], "termination_reason": "apogee"}],
                    "wilson": {"score": 1.0},
                    "termination_counts": {"apogee": 1},
                },
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            model = Path(tmpdir) / "model.zip"
            vec = Path(tmpdir) / "vec.pkl"
            model.write_bytes(b"model")
            vec.write_bytes(b"vec")
            with (
                patch.object(stress_eval.PPO, "load", return_value=DummyModel()),
                patch.object(stress_eval, "load_vecnormalize_for_eval", return_value=DummyVecnorm()),
                patch.object(stress_eval, "evaluate_case", side_effect=fake_evaluate_case),
            ):
                result = stress_eval.run_stress_eval(
                    model_path=str(model),
                    vecnorm_path=str(vec),
                    out_dir=tmpdir,
                    n_episodes=3,
                    seed_start=123,
                    case_names="nominal_distribution,cd_1.2x",
                    include_scripted=False,
                )

            saved = json.loads((Path(tmpdir) / "stress_eval.json").read_text())

        self.assertEqual(result["eval"]["seed_start"], 123)
        self.assertEqual(result["eval"]["seed_end"], 125)
        self.assertEqual(len(saved["cases"]), 2)
        self.assertEqual(saved["cases"][0]["trained"]["n_episodes"], 3)

    def test_stress_failure_dagger_preserves_dataset_shapes(self) -> None:
        import rl.dagger as dagger

        class DummyModel:
            def predict(self, obs, deterministic=True):
                _ = obs, deterministic
                return np.array([0.0, 0.0], dtype=np.float32), None

        class DummyVecnorm:
            obs_rms = type("Rms", (), {"mean": np.zeros(20), "var": np.ones(20)})()
            epsilon = 1e-8
            clip_obs = 10.0

            def close(self) -> None:
                pass

        class DummyEnv:
            def __init__(self) -> None:
                self.mission = MissionConfig()
                self.max_g_seen = 0.0
                self.steps = 0

            def reset(self, seed=None):
                self.steps = 0
                return np.zeros(20, dtype=np.float32), {"q_dyn_pa": 0.0, "altitude_m": 0.0}

            def step(self, action):
                self.steps += 1
                done = self.steps >= 2
                info = {
                    "success": False,
                    "termination_reason": "apogee",
                    "max_altitude_m": 80_000.0,
                    "altitude_at_burnout_m": 20_000.0,
                    "velocity_at_burnout_mps": 1_000.0,
                    "vz_at_burnout_mps": 1_000.0,
                    "max_q_dyn": 60_000.0,
                    "max_g_load": 6.0,
                    "fuel_used_fraction": 1.0,
                    "mission": {"max_q_pa": 70_000.0, "max_g_load": 8.5},
                }
                return np.ones(20, dtype=np.float32) * self.steps, 0.0, done, False, info

            def close(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            expert_path = base / "expert.npz"
            np.savez_compressed(
                expert_path,
                observations=np.zeros((2, 20), dtype=np.float32),
                actions=np.zeros((2, 2), dtype=np.float32),
                episode_ids=np.zeros(2, dtype=np.int32),
            )
            stress_path = base / "stress.json"
            stress_path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "case": {
                                    "name": "case",
                                    "description": "case",
                                    "vehicle_scales": {},
                                    "atmosphere_scales": {},
                                },
                                "trained": {
                                    "episodes": [
                                        {"seed": 99, "success": False, "q_margin_pa": 1_000.0},
                                    ]
                                },
                            }
                        ]
                    }
                )
            )
            with (
                patch.object(dagger.PPO, "load", return_value=DummyModel()),
                patch.object(dagger, "load_vecnormalize_for_eval", return_value=DummyVecnorm()),
                patch.object(dagger, "build_stress_env", return_value=DummyEnv()),
            ):
                summary = dagger.collect_stress_failure_dagger_dataset(
                    stress_eval_path=stress_path,
                    expert_dataset_path=expert_path,
                    model_path="model.zip",
                    vecnorm_path="vec.pkl",
                    out_dir=base / "out",
                    max_failure_episodes=1,
                )
            data = np.load(base / "out" / "trajectories.npz")

        self.assertEqual(summary["n_dagger_samples"], 2)
        self.assertEqual(data["observations"].shape[1], 20)
        self.assertEqual(data["actions"].shape[1], 2)
        self.assertEqual(data["observations"].shape[0], data["actions"].shape[0])

    def test_stress_feasibility_summary_ranks_success_above_shortfall(self) -> None:
        from rl.stress_feasibility import _controller_rank, summarise_controller

        success = summarise_controller(
            "success",
            [
                {
                    "success": True,
                    "q_ok": True,
                    "g_ok": True,
                    "max_altitude_m": 101_000.0,
                    "projected_burnout_apogee_m": 101_000.0,
                    "vz_at_burnout_mps": 1_250.0,
                    "max_q_dyn": 60_000.0,
                    "max_g_load": 7.0,
                }
            ],
            {"controller": "test"},
        )
        shortfall = summarise_controller(
            "shortfall",
            [
                {
                    "success": False,
                    "q_ok": True,
                    "g_ok": True,
                    "max_altitude_m": 99_000.0,
                    "projected_burnout_apogee_m": 99_000.0,
                    "vz_at_burnout_mps": 1_200.0,
                    "max_q_dyn": 50_000.0,
                    "max_g_load": 7.0,
                }
            ],
            {"controller": "test"},
        )

        self.assertGreater(_controller_rank(success), _controller_rank(shortfall))


class TestCheckpointSelector(unittest.TestCase):
    def test_select_checkpoint_prefers_official_success_with_matching_vecnorm(self) -> None:
        from rl.select_checkpoint import select_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            eval_logs = base / "eval_logs"
            checkpoints = base / "checkpoints"
            eval_logs.mkdir()
            checkpoints.mkdir()
            (eval_logs / "fixed_final_eval_history.jsonl").write_text(
                json.dumps({"num_timesteps": 100, "success_rate": 0.2, "q_ok_rate": 1.0, "g_ok_rate": 1.0, "avg_burnout_vz_mps": 1190.0, "avg_max_altitude_m": 95_000.0})
                + "\n"
                + json.dumps({"num_timesteps": 200, "success_rate": 0.1, "q_ok_rate": 1.0, "g_ok_rate": 1.0, "avg_burnout_vz_mps": 1210.0, "avg_max_altitude_m": 98_000.0})
                + "\n"
            )
            (eval_logs / "official_fixed_final_eval_history.jsonl").write_text(
                json.dumps({"num_timesteps": 200, "success_rate": 0.3, "q_ok_rate": 1.0, "g_ok_rate": 1.0, "avg_burnout_vz_mps": 1200.0, "avg_max_altitude_m": 99_000.0}) + "\n"
            )
            for step in (100, 200):
                (checkpoints / f"ppo_rocket_{step}_steps.zip").write_bytes(b"model")
                (checkpoints / f"vecnormalize_{step}_steps.pkl").write_bytes(b"vec")

            result = select_checkpoint(base)

        self.assertEqual(result["selected"]["num_timesteps"], 200)

    def test_select_checkpoint_uses_wilson_score_over_raw_altitude(self) -> None:
        from rl.select_checkpoint import select_checkpoint

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            eval_logs = base / "eval_logs"
            checkpoints = base / "checkpoints"
            eval_logs.mkdir()
            checkpoints.mkdir()
            (eval_logs / "fixed_final_eval_history.jsonl").write_text(
                json.dumps(
                    {
                        "num_timesteps": 100,
                        "n_episodes": 100,
                        "success_rate": 0.90,
                        "q_ok_rate": 1.0,
                        "g_ok_rate": 1.0,
                        "avg_q_margin_pa": 8_000.0,
                        "avg_burnout_vz_mps": 1_350.0,
                        "avg_max_altitude_m": 120_000.0,
                    }
                )
                + "\n"
                + json.dumps(
                    {
                        "num_timesteps": 200,
                        "n_episodes": 100,
                        "success_rate": 0.95,
                        "q_ok_rate": 0.60,
                        "g_ok_rate": 1.0,
                        "avg_q_margin_pa": -1_000.0,
                        "avg_burnout_vz_mps": 1_500.0,
                        "avg_max_altitude_m": 150_000.0,
                    }
                )
                + "\n"
            )
            (eval_logs / "official_fixed_final_eval_history.jsonl").write_text("")
            for step in (100, 200):
                (checkpoints / f"ppo_rocket_{step}_steps.zip").write_bytes(b"model")
                (checkpoints / f"vecnormalize_{step}_steps.pkl").write_bytes(b"vec")

            result = select_checkpoint(base)

        self.assertEqual(result["selected"]["num_timesteps"], 100)
        self.assertIn("fixed_wilson", result["selected"])
        self.assertGreater(
            result["selected"]["fixed_wilson"]["score"],
            result["candidates"][1]["fixed_wilson"]["score"],
        )


class TestPromotionManifest(unittest.TestCase):
    def test_promotion_manifest_validates_paths_and_counts(self) -> None:
        from rl.promote_policy import build_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            model = base / "model.zip"
            vecnorm = base / "vec.pkl"
            eval_path = base / "eval.json"
            out_path = base / "manifest.json"
            model.write_bytes(b"model")
            vecnorm.write_bytes(b"vec")
            episodes = [
                {"seed": 10, "success": True, "max_q_dyn": 50_000.0, "max_g_load": 7.0},
                {"seed": 11, "success": False, "max_q_dyn": 75_000.0, "max_g_load": 7.0},
            ]
            eval_path.write_text(
                json.dumps(
                    {
                        "eval": {"seed_start": 10, "seed_end": 11, "deterministic": True},
                        "trained": {
                            "target_altitude_m": 100_000.0,
                            "avg_max_altitude_m": 120_000.0,
                            "avg_max_q": 62_000.0,
                            "avg_q_margin_pa": 8_000.0,
                            "avg_max_g_load": 7.1,
                            "avg_burnout_vz_mps": 1_400.0,
                            "episodes": episodes,
                        },
                    }
                )
            )

            manifest = build_manifest(
                model_path=str(model),
                vecnorm_path=str(vecnorm),
                eval_path=str(eval_path),
                out_path=str(out_path),
                policy_label="test_policy",
            )
            self.assertTrue(out_path.exists())

        self.assertEqual(manifest["policy_label"], "test_policy")
        self.assertEqual(manifest["counts"]["success"]["count"], 1)
        self.assertEqual(manifest["counts"]["q_ok"]["count"], 1)
        self.assertEqual(manifest["eval"]["seed_start"], 10)
        self.assertIn("vehicle", manifest["config"])


class TestFeasibilityHarness(unittest.TestCase):
    def test_feedback_parameter_decoding_is_bounded(self) -> None:
        from rl.feasibility import decode_feedback_params

        params = decode_feedback_params(np.array([-9, -9, 9, 9, 9, -9, 9, 9], dtype=np.float32))

        self.assertGreaterEqual(params["q_cap_pa"], 45_000.0)
        self.assertLessEqual(params["q_cap_pa"], 72_000.0)
        self.assertGreaterEqual(params["floor"], 0.35)
        self.assertLessEqual(params["floor"], 0.78)
        self.assertGreaterEqual(params["high"], 0.82)
        self.assertLessEqual(params["high"], 1.0)
        self.assertGreaterEqual(params["gimbal_norm"], 0.0)
        self.assertLessEqual(params["gimbal_norm"], 0.9)

    def test_q_feedback_throttle_reduces_above_cap_and_respects_floor(self) -> None:
        from rl.feasibility import q_feedback_throttle

        below = q_feedback_throttle(q_dyn_pa=40_000.0, q_cap_pa=60_000.0, floor=0.5, high=1.0, gain=3.0)
        above = q_feedback_throttle(q_dyn_pa=72_000.0, q_cap_pa=60_000.0, floor=0.5, high=1.0, gain=3.0)
        very_high = q_feedback_throttle(q_dyn_pa=120_000.0, q_cap_pa=60_000.0, floor=0.5, high=1.0, gain=3.0)

        self.assertEqual(below, 1.0)
        self.assertLess(above, below)
        self.assertEqual(very_high, 0.5)

    def test_feasibility_ranking_prefers_success_then_100km_reach_then_shortfall(self) -> None:
        from rl.feasibility import rank_metrics

        compliant_success = {
            "success": True,
            "q_ok": True,
            "g_ok": True,
            "max_altitude_m": 101_000.0,
            "projected_burnout_apogee_m": 101_000.0,
            "vz_at_burnout_mps": 1_250.0,
        }
        q_violating_100km = dict(
            compliant_success,
            success=False,
            q_ok=False,
            max_altitude_m=110_000.0,
            projected_burnout_apogee_m=110_000.0,
        )
        compliant_shortfall = dict(
            compliant_success,
            success=False,
            max_altitude_m=90_000.0,
            projected_burnout_apogee_m=92_000.0,
        )

        self.assertGreater(rank_metrics(compliant_success), rank_metrics(q_violating_100km))
        self.assertGreater(rank_metrics(q_violating_100km), rank_metrics(compliant_shortfall))

    def test_feasibility_smoke_writes_outputs(self) -> None:
        from rl.feasibility import run_feasibility

        with tempfile.TemporaryDirectory() as tmpdir:
            summary = run_feasibility(
                out_dir=tmpdir,
                seed=1,
                cem_population=4,
                cem_generations=1,
                cem_restarts=1,
            )
            self.assertTrue((Path(tmpdir) / "results.jsonl").exists())
            self.assertTrue((Path(tmpdir) / "summary.json").exists())
            self.assertTrue((Path(tmpdir) / "trajectories.npz").exists())
            self.assertIn("best_overall", summary)
