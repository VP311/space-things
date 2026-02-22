"""Train PPO on the rocket ascent environment."""

from __future__ import annotations

from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from config.defaults import EnvConfig, default_vehicle_params
from rl.env import RocketAscentEnv


def train(total_timesteps: int = 10_000) -> Path:
    params = default_vehicle_params()
    cfg = EnvConfig()
    env = Monitor(
        RocketAscentEnv(
            params=params,
            dt=cfg.dt,
            t_final=cfg.t_final,
            q_limit=cfg.q_limit,
            target_apogee=cfg.target_apogee,
            record=False,
        )
    )

    model = PPO(
        policy="MlpPolicy",
        env=env,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        ent_coef=0.0,
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps)

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_base = artifacts_dir / "ppo_rocket"
    model.save(str(save_base))
    model_path = save_base.with_suffix(".zip")
    print(f"saved model: {model_path}")
    return model_path


if __name__ == "__main__":
    train(total_timesteps=10_000)
