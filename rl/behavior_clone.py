"""Behavior cloning from expert ascent trajectories into the PPO MlpPolicy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config.defaults import PPOConfig
from rl.policy_eval import build_eval_env


def load_expert_dataset(path: str | Path) -> dict[str, np.ndarray]:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)
    with np.load(dataset_path, allow_pickle=False) as data:
        required = {"observations", "actions"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"dataset missing arrays: {sorted(missing)}")
        observations = np.asarray(data["observations"], dtype=np.float32)
        actions = np.asarray(data["actions"], dtype=np.float32)
        episode_ids = np.asarray(data["episode_ids"], dtype=np.int32) if "episode_ids" in data.files else np.zeros(len(actions), dtype=np.int32)
        sample_weights = np.asarray(data["sample_weights"], dtype=np.float32) if "sample_weights" in data.files else np.ones(len(actions), dtype=np.float32)

    if observations.ndim != 2 or observations.shape[1] != 20:
        raise ValueError(f"observations must have shape [N, 20], got {observations.shape}")
    if actions.ndim != 2 or actions.shape[1] != 2:
        raise ValueError(f"actions must have shape [N, 2], got {actions.shape}")
    if observations.shape[0] != actions.shape[0]:
        raise ValueError("observations and actions must have matching first dimension")
    if observations.shape[0] == 0:
        raise ValueError("dataset is empty")
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError("dataset contains non-finite observations or actions")
    if sample_weights.ndim != 1 or sample_weights.shape[0] != observations.shape[0]:
        raise ValueError(f"sample_weights must have shape [N], got {sample_weights.shape}")
    if not np.isfinite(sample_weights).all() or np.any(sample_weights <= 0.0):
        raise ValueError("sample_weights must be finite and positive")
    return {"observations": observations, "actions": actions, "episode_ids": episode_ids, "sample_weights": sample_weights}


def build_bc_env(target_altitude_m: float = 100_000.0) -> VecNormalize:
    raw_env = DummyVecEnv([lambda: build_eval_env(target_altitude_m, start_state_randomization=False)])
    return VecNormalize(raw_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)


def fit_vecnormalize_stats(vecnorm: VecNormalize, observations: np.ndarray) -> None:
    obs = np.asarray(observations, dtype=np.float64)
    vecnorm.obs_rms.mean = np.mean(obs, axis=0)
    vecnorm.obs_rms.var = np.maximum(np.var(obs, axis=0), 1e-8)
    vecnorm.obs_rms.count = float(obs.shape[0])


def normalise_observations(vecnorm: VecNormalize, observations: np.ndarray) -> np.ndarray:
    obs = np.asarray(observations, dtype=np.float32)
    rms = vecnorm.obs_rms
    normed = (obs - rms.mean.astype(np.float32)) / np.sqrt(rms.var.astype(np.float32) + vecnorm.epsilon)
    return np.clip(normed, -vecnorm.clip_obs, vecnorm.clip_obs).astype(np.float32)


def apply_early_flight_weights(
    observations: np.ndarray,
    base_weights: np.ndarray,
    *,
    target_altitude_m: float = 100_000.0,
    early_altitude_m: float = 8_000.0,
    early_weight: float = 4.0,
) -> np.ndarray:
    weights = np.asarray(base_weights, dtype=np.float32).copy()
    if early_weight <= 0.0:
        return weights
    altitude_m = np.asarray(observations[:, 0], dtype=np.float32) * float(target_altitude_m)
    early_frac = np.clip((float(early_altitude_m) - altitude_m) / max(float(early_altitude_m), 1.0), 0.0, 1.0)
    return (weights * (1.0 + float(early_weight) * early_frac)).astype(np.float32)


def _make_model(vecnorm: VecNormalize, *, seed: int, learning_rate: float) -> PPO:
    cfg = PPOConfig()
    return PPO(
        policy="MlpPolicy",
        env=vecnorm,
        n_steps=min(cfg.n_steps, 512),
        batch_size=min(cfg.batch_size, 256),
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        learning_rate=float(learning_rate),
        ent_coef=0.0,
        clip_range=cfg.clip_range,
        n_epochs=1,
        vf_coef=cfg.vf_coef,
        policy_kwargs={"net_arch": list(cfg.net_arch)},
        target_kl=cfg.target_kl,
        seed=int(seed),
        verbose=0,
    )


def behavior_clone(
    *,
    dataset_path: str | Path = "artifacts/archived/artifacts_expert/trajectories.npz",
    out_dir: str | Path = "artifacts/living/artifacts_bc",
    epochs: int = 25,
    batch_size: int = 512,
    learning_rate: float = 3e-4,
    early_weight: float = 0.0,
    early_altitude_m: float = 8_000.0,
    seed: int = 0,
) -> dict[str, Any]:
    dataset = load_expert_dataset(dataset_path)
    observations = dataset["observations"]
    actions = dataset["actions"]
    sample_weights = apply_early_flight_weights(
        observations,
        dataset["sample_weights"],
        early_weight=early_weight,
        early_altitude_m=early_altitude_m,
    )

    vecnorm = build_bc_env(100_000.0)
    fit_vecnormalize_stats(vecnorm, observations)
    obs_norm = normalise_observations(vecnorm, observations)

    model = _make_model(vecnorm, seed=seed, learning_rate=learning_rate)
    rng = np.random.default_rng(seed)
    optimizer = torch.optim.Adam(model.policy.parameters(), lr=float(learning_rate))
    device = model.policy.device
    obs_tensor_all = torch.as_tensor(obs_norm, dtype=torch.float32, device=device)
    action_tensor_all = torch.as_tensor(actions, dtype=torch.float32, device=device)
    weight_tensor_all = torch.as_tensor(sample_weights, dtype=torch.float32, device=device)

    losses: list[float] = []
    n = observations.shape[0]
    for _ in range(int(epochs)):
        order = rng.permutation(n)
        for start in range(0, n, int(batch_size)):
            idx = order[start : start + int(batch_size)]
            obs_batch = obs_tensor_all[idx]
            action_batch = action_tensor_all[idx]
            weight_batch = weight_tensor_all[idx]
            dist = model.policy.get_distribution(obs_batch)
            pred = dist.mode()
            per_sample_loss = F.mse_loss(pred, action_batch, reduction="none").mean(dim=1)
            loss = torch.mean(per_sample_loss * weight_batch) / torch.clamp(torch.mean(weight_batch), min=1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.policy.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model_path_base = out / "bc_model"
    vecnorm_path = out / "vecnormalize.pkl"
    model.save(str(model_path_base))
    vecnorm.save(str(vecnorm_path))

    metrics = {
        "dataset_path": str(dataset_path),
        "n_samples": int(n),
        "n_episodes": int(len(np.unique(dataset["episode_ids"]))),
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "learning_rate": float(learning_rate),
        "early_weight": float(early_weight),
        "early_altitude_m": float(early_altitude_m),
        "mean_sample_weight": float(np.mean(sample_weights)),
        "max_sample_weight": float(np.max(sample_weights)),
        "initial_loss": float(losses[0]) if losses else None,
        "final_loss": float(losses[-1]) if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "model_path": str(model_path_base.with_suffix(".zip")),
        "vecnormalize_path": str(vecnorm_path),
    }
    (out / "bc_metrics.json").write_text(json.dumps(metrics, indent=2))
    vecnorm.close()
    print(json.dumps(metrics, indent=2))
    return metrics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Behavior-clone PPO MlpPolicy from expert trajectories")
    parser.add_argument("--dataset", type=str, default="artifacts/archived/artifacts_expert/trajectories.npz")
    parser.add_argument("--out-dir", type=str, default="artifacts/living/artifacts_bc")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--early-weight", type=float, default=0.0)
    parser.add_argument("--early-altitude-m", type=float, default=8_000.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    behavior_clone(
        dataset_path=args.dataset,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        early_weight=args.early_weight,
        early_altitude_m=args.early_altitude_m,
        seed=args.seed,
    )
