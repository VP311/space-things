"""Train PPO on the rocket ascent environment with long-run callbacks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

from config.defaults import AtmosphereConfig, EnvConfig, MissionConfig, PPOConfig, default_vehicle_params
from rl.env import RocketAscentEnv


def _build_env(
    env_cfg: EnvConfig,
    mission: MissionConfig,
    atmosphere_cfg: AtmosphereConfig,
) -> RocketAscentEnv:
    return RocketAscentEnv(
        params=default_vehicle_params(),
        mission=mission,
        atmosphere_cfg=atmosphere_cfg,
        dt=env_cfg.dt,
        t_final=env_cfg.t_final,
        record=False,
    )


def train(
    total_timesteps: int | None = None,
    n_envs: int | None = None,
) -> Path:
    env_cfg = EnvConfig()
    mission = MissionConfig()
    atmosphere_cfg = AtmosphereConfig()
    ppo_cfg = PPOConfig()

    run_steps = int(total_timesteps if total_timesteps is not None else ppo_cfg.total_timesteps)
    num_envs = int(n_envs if n_envs is not None else ppo_cfg.n_envs)

    train_env = make_vec_env(
        lambda: _build_env(env_cfg, mission, atmosphere_cfg),
        n_envs=num_envs,
        seed=ppo_cfg.seed,
    )
    eval_env = Monitor(_build_env(env_cfg, mission, atmosphere_cfg))

    artifacts_dir = Path("artifacts")
    checkpoints_dir = artifacts_dir / "checkpoints"
    best_model_dir = artifacts_dir / "best_model"
    eval_logs_dir = artifacts_dir / "eval_logs"
    tensorboard_dir = artifacts_dir / "tb"
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

    # SB3 callback frequencies are counted in vectorized env calls, not raw transitions.
    checkpoint_freq = max(ppo_cfg.checkpoint_freq_steps // num_envs, 1)
    eval_freq = max(ppo_cfg.eval_freq_steps // num_envs, 1)

    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=str(checkpoints_dir),
                name_prefix="ppo_rocket",
            ),
            EvalCallback(
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
        tensorboard_log=tensorboard_log,
        seed=ppo_cfg.seed,
        verbose=1,
    )

    run_config = {
        "total_timesteps": run_steps,
        "n_envs": num_envs,
        "ppo": {
            "n_steps": ppo_cfg.n_steps,
            "batch_size": ppo_cfg.batch_size,
            "gamma": ppo_cfg.gamma,
            "gae_lambda": ppo_cfg.gae_lambda,
            "learning_rate": ppo_cfg.learning_rate,
            "ent_coef": ppo_cfg.ent_coef,
            "clip_range": ppo_cfg.clip_range,
            "net_arch": list(ppo_cfg.net_arch),
            "seed": ppo_cfg.seed,
            "checkpoint_freq_steps": ppo_cfg.checkpoint_freq_steps,
            "eval_freq_steps": ppo_cfg.eval_freq_steps,
            "eval_episodes": ppo_cfg.eval_episodes,
        },
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
    model_path = save_base.with_suffix(".zip")
    print(f"saved model: {model_path}")
    print(f"best model dir: {best_model_dir}")

    train_env.close()
    eval_env.close()
    return model_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO rocket guidance policy")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total training timesteps")
    parser.add_argument("--n-envs", type=int, default=None, help="Override number of vectorized environments")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(total_timesteps=args.total_timesteps, n_envs=args.n_envs)
