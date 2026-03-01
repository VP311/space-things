"""Train PPO on the rocket ascent environment with curriculum and diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, sync_envs_normalization

from config.defaults import (
    AtmosphereConfig,
    CurriculumConfig,
    EnvConfig,
    MissionConfig,
    PPOConfig,
    default_vehicle_params,
)
from rl.env import RocketAscentEnv


@dataclass
class CurriculumManager:
    current_target_m: float
    max_target_m: float
    growth_factor: float
    success_threshold: float
    altitude_ratio_threshold: float
    window_episodes: int
    start_altitude_fraction: float

    def __post_init__(self) -> None:
        self.success_history: deque[float] = deque(maxlen=self.window_episodes)
        self.altitude_ratio_history: deque[float] = deque(maxlen=self.window_episodes)
        self._last_promoted_to = self.current_target_m

    def record(self, success: bool, altitude_ratio: float) -> tuple[bool, float]:
        self.success_history.append(1.0 if success else 0.0)
        self.altitude_ratio_history.append(float(np.clip(altitude_ratio, 0.0, 2.0)))
        if len(self.success_history) < self.window_episodes:
            return False, self.current_target_m

        rolling_success = sum(self.success_history) / len(self.success_history)
        rolling_alt_ratio = sum(self.altitude_ratio_history) / len(self.altitude_ratio_history)
        if rolling_success < self.success_threshold and rolling_alt_ratio < self.altitude_ratio_threshold:
            return False, self.current_target_m

        next_target = min(self.current_target_m * self.growth_factor, self.max_target_m)
        promoted = next_target > self.current_target_m + 1e-6
        self.current_target_m = next_target
        self.success_history.clear()
        self.altitude_ratio_history.clear()
        self._last_promoted_to = next_target
        return promoted, self.current_target_m

    def start_altitude_cap(self) -> float:
        return max(0.0, self.current_target_m * self.start_altitude_fraction)


class EpisodeSummaryCallback(BaseCallback):
    def __init__(
        self,
        out_path: Path,
        curriculum: CurriculumManager | None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.out_path = out_path
        self.curriculum = curriculum
        self._fh: Any | None = None
        self._episode_idx = 0

    def _on_training_start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.out_path.open("w", encoding="utf-8")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if self._fh is None:
            return True

        for done, info in zip(dones, infos):
            if not done:
                continue
            self._episode_idx += 1

            reward_sums = dict(info.get("reward_components_cumulative", {}))
            row = {
                "episode": self._episode_idx,
                "t": float(info.get("t", 0.0)),
                "max_altitude_m": float(info.get("max_altitude_m", 0.0)),
                "altitude_at_burnout_m": float(info.get("altitude_at_burnout_m", float("nan"))),
                "altitude_at_apogee_m": float(info.get("altitude_at_apogee_m", 0.0)),
                "time_to_apogee_s": float(info.get("time_to_apogee_s", float("nan"))),
                "max_q_dyn": float(info.get("max_q_dyn", 0.0)),
                "max_g_load": float(info.get("max_g_load", 0.0)),
                "fuel_used_fraction": float(info.get("fuel_used_fraction", 0.0)),
                "success": bool(info.get("success", False)),
                "termination_reason": str(info.get("termination_reason", "unknown")),
                "mean_throttle": float(info.get("mean_throttle", 0.0)),
                "throttle_variance": float(info.get("throttle_variance", 0.0)),
                "gimbal_variance": float(info.get("gimbal_variance", 0.0)),
                "curriculum_target_altitude_m": float(info.get("mission", {}).get("target_altitude_m", 0.0)),
                "altitude_ratio_to_target": float(info.get("max_altitude_m", 0.0))
                / max(float(info.get("mission", {}).get("target_altitude_m", 1.0)), 1.0),
                "reward_components": reward_sums,
            }
            self._fh.write(json.dumps(row) + "\n")

            if self.curriculum is not None:
                target_now = float(info.get("mission", {}).get("target_altitude_m", 1.0))
                max_alt = float(info.get("max_altitude_m", 0.0))
                altitude_ratio = max_alt / max(target_now, 1.0)
                promoted, target = self.curriculum.record(bool(info.get("success", False)), altitude_ratio)
                if promoted:
                    cap = self.curriculum.start_altitude_cap()
                    self.training_env.env_method(
                        "set_curriculum",
                        target_altitude_m=float(target),
                        start_altitude_cap_m=float(cap),
                    )
                    if self.verbose > 0:
                        print(
                            "curriculum promote: "
                            f"target_altitude_m={target:.1f} start_altitude_cap_m={cap:.1f}"
                        )

        self._fh.flush()
        return True

    def _on_training_end(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class NormalizedEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            sync_envs_normalization(self.training_env, self.eval_env)
        return super()._on_step()


def _build_env(
    env_cfg: EnvConfig,
    mission: MissionConfig,
    atmosphere_cfg: AtmosphereConfig,
    *,
    initial_target_altitude_m: float,
    curriculum_cfg: CurriculumConfig,
) -> RocketAscentEnv:
    env = RocketAscentEnv(
        params=default_vehicle_params(),
        mission=mission,
        atmosphere_cfg=atmosphere_cfg,
        dt=env_cfg.dt,
        t_final=env_cfg.t_final,
        record=False,
        max_start_altitude_m=initial_target_altitude_m * curriculum_cfg.start_altitude_fraction,
    )
    env.set_curriculum(
        target_altitude_m=initial_target_altitude_m,
        start_altitude_cap_m=initial_target_altitude_m * curriculum_cfg.start_altitude_fraction,
    )
    return env


def train(
    total_timesteps: int | None = None,
    n_envs: int | None = None,
    initial_target_altitude_m: float | None = None,
    curriculum_enabled: bool | None = None,
) -> Path:
    env_cfg = EnvConfig()
    mission = MissionConfig()
    atmosphere_cfg = AtmosphereConfig()
    ppo_cfg = PPOConfig()
    curriculum_cfg = CurriculumConfig()

    run_steps = int(total_timesteps if total_timesteps is not None else ppo_cfg.total_timesteps)
    num_envs = int(n_envs if n_envs is not None else ppo_cfg.n_envs)
    start_target = float(
        initial_target_altitude_m if initial_target_altitude_m is not None else curriculum_cfg.initial_target_altitude_m
    )
    use_curriculum = curriculum_cfg.enabled if curriculum_enabled is None else bool(curriculum_enabled)

    train_env = make_vec_env(
        lambda: _build_env(
            env_cfg,
            mission,
            atmosphere_cfg,
            initial_target_altitude_m=start_target,
            curriculum_cfg=curriculum_cfg,
        ),
        n_envs=num_envs,
        seed=ppo_cfg.seed,
    )
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    eval_env = make_vec_env(
        lambda: _build_env(
            env_cfg,
            mission,
            atmosphere_cfg,
            initial_target_altitude_m=start_target,
            curriculum_cfg=curriculum_cfg,
        ),
        n_envs=1,
        seed=ppo_cfg.seed + 10_000,
    )
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False, clip_obs=10.0)

    artifacts_dir = Path("artifacts")
    checkpoints_dir = artifacts_dir / "checkpoints"
    best_model_dir = artifacts_dir / "best_model"
    eval_logs_dir = artifacts_dir / "eval_logs"
    tensorboard_dir = artifacts_dir / "tb"
    summaries_path = artifacts_dir / "episode_summaries.jsonl"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)
    eval_logs_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log: str | None = str(tensorboard_dir)
    try:
        import tensorboard  # type: ignore  # noqa: F401
    except Exception:
        tensorboard_log = None
        print("tensorboard not installed; training will continue without tensorboard logs.")

    checkpoint_freq = max(ppo_cfg.checkpoint_freq_steps // num_envs, 1)
    eval_freq = max(ppo_cfg.eval_freq_steps // num_envs, 1)

    curriculum_mgr: CurriculumManager | None = None
    if use_curriculum:
        curriculum_mgr = CurriculumManager(
            current_target_m=start_target,
            max_target_m=curriculum_cfg.max_target_altitude_m,
            growth_factor=curriculum_cfg.growth_factor,
            success_threshold=curriculum_cfg.success_threshold,
            altitude_ratio_threshold=curriculum_cfg.altitude_ratio_threshold,
            window_episodes=curriculum_cfg.window_episodes,
            start_altitude_fraction=curriculum_cfg.start_altitude_fraction,
        )

    callbacks = CallbackList(
        [
            EpisodeSummaryCallback(summaries_path, curriculum=curriculum_mgr, verbose=1),
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(checkpoints_dir),
                name_prefix="ppo_rocket",
            ),
            NormalizedEvalCallback(
                eval_env,
                best_model_save_path=str(best_model_dir),
                log_path=str(eval_logs_dir),
                eval_freq=eval_freq,
                n_eval_episodes=ppo_cfg.eval_episodes,
                deterministic=True,
                render=False,
            ),
        ]
    )

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        n_steps=ppo_cfg.n_steps,
        batch_size=ppo_cfg.batch_size,
        gamma=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        learning_rate=ppo_cfg.learning_rate,
        ent_coef=ppo_cfg.ent_coef,
        clip_range=ppo_cfg.clip_range,
        policy_kwargs={"net_arch": list(ppo_cfg.net_arch)},
        target_kl=ppo_cfg.target_kl,
        tensorboard_log=tensorboard_log,
        seed=ppo_cfg.seed,
        verbose=1,
    )

    run_config = {
        "total_timesteps": run_steps,
        "n_envs": num_envs,
        "initial_target_altitude_m": start_target,
        "curriculum_enabled": use_curriculum,
        "ppo": {
            "n_steps": ppo_cfg.n_steps,
            "batch_size": ppo_cfg.batch_size,
            "gamma": ppo_cfg.gamma,
            "gae_lambda": ppo_cfg.gae_lambda,
            "learning_rate": ppo_cfg.learning_rate,
            "ent_coef": ppo_cfg.ent_coef,
            "clip_range": ppo_cfg.clip_range,
            "target_kl": ppo_cfg.target_kl,
            "net_arch": list(ppo_cfg.net_arch),
            "seed": ppo_cfg.seed,
            "checkpoint_freq_steps": ppo_cfg.checkpoint_freq_steps,
            "eval_freq_steps": ppo_cfg.eval_freq_steps,
            "eval_episodes": ppo_cfg.eval_episodes,
        },
        "curriculum": curriculum_cfg.__dict__,
        "env": {
            "dt": env_cfg.dt,
            "t_final": env_cfg.t_final,
            "mission": mission.__dict__,
            "atmosphere": atmosphere_cfg.__dict__,
        },
    }
    (artifacts_dir / "train_run_config.json").write_text(json.dumps(run_config, indent=2))

    model.learn(total_timesteps=run_steps, callback=callbacks)

    save_base = artifacts_dir / "ppo_rocket"
    model.save(str(save_base))
    train_env.save(str(artifacts_dir / "vecnormalize.pkl"))
    model_path = save_base.with_suffix(".zip")
    print(f"saved model: {model_path}")
    print(f"best model dir: {best_model_dir}")
    print(f"episode summaries: {summaries_path}")

    train_env.close()
    eval_env.close()
    return model_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO rocket guidance policy")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total training timesteps")
    parser.add_argument("--n-envs", type=int, default=None, help="Override number of vectorized environments")
    parser.add_argument(
        "--initial-target-altitude-m",
        type=float,
        default=None,
        help="Override initial curriculum target altitude",
    )
    parser.add_argument(
        "--disable-curriculum",
        action="store_true",
        help="Disable curriculum promotion and keep fixed target altitude",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        initial_target_altitude_m=args.initial_target_altitude_m,
        curriculum_enabled=False if args.disable_curriculum else None,
    )
