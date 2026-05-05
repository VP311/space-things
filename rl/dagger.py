"""DAgger-style relabeling for scripted q-bucket ascent experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO

from rl.behavior_clone import load_expert_dataset
from rl.feasibility import fixed_action_controller, projected_burnout_apogee
from rl.policy_eval import build_eval_env, load_vecnormalize_for_eval, normalise_obs
from rl.stress_eval import StressCase, build_stress_env


DEFAULT_QBUCKET_PARAMS = (
    {"q_cap_pa": 45_000.0, "bucket": 0.4, "high": 1.0, "until_alt_m": 60_000.0},
    {"q_cap_pa": 50_000.0, "bucket": 0.5, "high": 1.0, "until_alt_m": 60_000.0},
    {"q_cap_pa": 50_000.0, "bucket": 0.6, "high": 1.0, "until_alt_m": 60_000.0},
    {"q_cap_pa": 55_000.0, "bucket": 0.5, "high": 1.0, "until_alt_m": 60_000.0},
    {"q_cap_pa": 60_000.0, "bucket": 0.5, "high": 1.0, "until_alt_m": 60_000.0},
)


def _episode_metrics(info: dict[str, Any], *, seed: int, episode: int, steps: int, label_params: dict[str, float]) -> dict[str, Any]:
    max_q = float(info.get("max_q_dyn", 0.0))
    max_g = float(info.get("max_g_load", 0.0))
    metrics: dict[str, Any] = {
        "seed": int(seed),
        "episode": int(episode),
        "steps": int(steps),
        "success": bool(info.get("success", False)),
        "termination_reason": str(info.get("termination_reason", "unknown")),
        "max_altitude_m": float(info.get("max_altitude_m", 0.0)),
        "altitude_at_burnout_m": float(info.get("altitude_at_burnout_m", float("nan"))),
        "velocity_at_burnout_mps": float(info.get("velocity_at_burnout_mps", float("nan"))),
        "vz_at_burnout_mps": float(info.get("vz_at_burnout_mps", float("nan"))),
        "max_q_dyn": max_q,
        "max_g_load": max_g,
        "q_ok": bool(max_q <= float(info.get("mission", {}).get("max_q_pa", 70_000.0))),
        "g_ok": bool(max_g <= float(info.get("mission", {}).get("max_g_load", 8.5))),
        "fuel_used_fraction": float(info.get("fuel_used_fraction", 0.0)),
        "label_params": label_params,
    }
    metrics["projected_burnout_apogee_m"] = projected_burnout_apogee(metrics)
    return metrics


def _save_combined_dataset(
    *,
    expert: dict[str, np.ndarray],
    dagger_obs: list[np.ndarray],
    dagger_actions: list[np.ndarray],
    dagger_episode_ids: list[np.ndarray],
    metrics: list[dict[str, Any]],
    out_dir: Path,
    expert_dataset_path: str | Path,
    model_path: str | Path,
    vecnorm_path: str | Path,
    episodes: int,
    dagger_weight: float,
    expert_weight: float,
) -> dict[str, Any]:
    dagger_observations = np.concatenate(dagger_obs, axis=0).astype(np.float32)
    dagger_labels = np.concatenate(dagger_actions, axis=0).astype(np.float32)
    dagger_ids = np.concatenate(dagger_episode_ids, axis=0).astype(np.int32)
    expert_ids = expert["episode_ids"].astype(np.int32)
    expert_id_offset = int(dagger_ids.max() + 1) if dagger_ids.size else 0

    observations = np.concatenate([expert["observations"], dagger_observations], axis=0).astype(np.float32)
    actions = np.concatenate([expert["actions"], dagger_labels], axis=0).astype(np.float32)
    episode_ids = np.concatenate([expert_ids + expert_id_offset, dagger_ids], axis=0).astype(np.int32)
    sample_weights = np.concatenate(
        [
            np.full(len(expert["actions"]), float(expert_weight), dtype=np.float32),
            np.full(len(dagger_labels), float(dagger_weight), dtype=np.float32),
        ],
        axis=0,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "trajectories.npz"
    metrics_json = np.asarray([json.dumps(row, sort_keys=True) for row in metrics])
    np.savez_compressed(
        dataset_path,
        observations=observations,
        actions=actions,
        episode_ids=episode_ids,
        sample_weights=sample_weights,
        metrics_json=metrics_json,
    )

    summary = {
        "expert_dataset_path": str(expert_dataset_path),
        "model_path": str(model_path),
        "vecnorm_path": str(vecnorm_path),
        "episodes": int(episodes),
        "n_expert_samples": int(len(expert["actions"])),
        "n_dagger_samples": int(len(dagger_labels)),
        "n_total_samples": int(len(actions)),
        "dagger_weight": float(dagger_weight),
        "expert_weight": float(expert_weight),
        "policy_success_rate_during_collection": float(np.mean([row["success"] for row in metrics])) if metrics else 0.0,
        "avg_policy_max_altitude_m": float(np.mean([row["max_altitude_m"] for row in metrics])) if metrics else 0.0,
        "dataset_path": str(dataset_path),
    }
    (out_dir / "dagger_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "dagger_rollouts.jsonl").write_text("\n".join(json.dumps(row) for row in metrics) + "\n")
    return summary


def collect_dagger_dataset(
    *,
    expert_dataset_path: str | Path,
    model_path: str | Path,
    vecnorm_path: str | Path,
    out_dir: str | Path = "artifacts/living/artifacts_dagger_solvable",
    episodes: int = 40,
    seed: int = 37,
    target_altitude_m: float = 100_000.0,
    dagger_weight: float = 3.0,
    expert_weight: float = 1.0,
    max_steps: int | None = None,
) -> dict[str, Any]:
    expert = load_expert_dataset(expert_dataset_path)
    model = PPO.load(str(model_path))
    vecnorm = load_vecnormalize_for_eval(str(vecnorm_path), target_altitude_m=target_altitude_m)

    dagger_obs: list[np.ndarray] = []
    dagger_actions: list[np.ndarray] = []
    dagger_episode_ids: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []

    for episode in range(int(episodes)):
        label_params = dict(DEFAULT_QBUCKET_PARAMS[episode % len(DEFAULT_QBUCKET_PARAMS)])
        label_controller = fixed_action_controller("q_bucket", label_params)
        env = build_eval_env(target_altitude_m)
        obs, info = env.reset(seed=seed + episode)
        terminated = False
        truncated = False
        step_count = 0
        ep_obs: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []
        while not (terminated or truncated):
            label_action = label_controller(env, info)
            ep_obs.append(np.asarray(obs, dtype=np.float32).copy())
            ep_actions.append(np.asarray(label_action, dtype=np.float32).copy())

            policy_action, _ = model.predict(normalise_obs(obs, vecnorm), deterministic=True)
            obs, _, terminated, truncated, info = env.step(policy_action)
            step_count += 1
            if max_steps is not None and step_count >= int(max_steps):
                break
        dagger_obs.append(np.asarray(ep_obs, dtype=np.float32))
        dagger_actions.append(np.asarray(ep_actions, dtype=np.float32))
        dagger_episode_ids.append(np.full(len(ep_actions), episode, dtype=np.int32))
        metrics.append(_episode_metrics(info, seed=seed + episode, episode=episode, steps=step_count, label_params=label_params))
        env.close()
        print(
            f"dagger episode {episode + 1}/{episodes}: "
            f"policy_success={int(metrics[-1]['success'])} "
            f"alt={metrics[-1]['max_altitude_m']:.1f} "
            f"q={metrics[-1]['max_q_dyn']:.1f} "
            f"steps={step_count}"
        )

    vecnorm.close()

    out = Path(out_dir)
    summary = _save_combined_dataset(
        expert=expert,
        dagger_obs=dagger_obs,
        dagger_actions=dagger_actions,
        dagger_episode_ids=dagger_episode_ids,
        metrics=metrics,
        out_dir=out,
        expert_dataset_path=expert_dataset_path,
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        episodes=episodes,
        dagger_weight=dagger_weight,
        expert_weight=expert_weight,
    )
    print(json.dumps(summary, indent=2))
    return summary


def _stress_case_from_json(row: dict[str, Any]) -> StressCase:
    case = row.get("case", row)
    return StressCase(
        name=str(case["name"]),
        description=str(case.get("description", "")),
        vehicle_scales={str(k): float(v) for k, v in dict(case.get("vehicle_scales", {})).items()},
        atmosphere_scales={str(k): float(v) for k, v in dict(case.get("atmosphere_scales", {})).items()},
        observation_noise_std=float(case.get("observation_noise_std", 0.0)),
        action_lag_steps=int(case.get("action_lag_steps", 0)),
        domain_randomization=bool(case.get("domain_randomization", False)),
        atmosphere_randomization=bool(case.get("atmosphere_randomization", True)),
    )


def _select_stress_failures(
    stress_eval: dict[str, Any],
    *,
    min_q_margin_pa: float,
    max_episodes: int | None,
) -> list[tuple[StressCase, dict[str, Any]]]:
    selected: list[tuple[StressCase, dict[str, Any]]] = []
    for case_row in stress_eval.get("cases", []):
        case = _stress_case_from_json(case_row)
        for episode in case_row.get("trained", {}).get("episodes", []):
            failed = not bool(episode.get("success", False))
            q_low_margin = float(episode.get("q_margin_pa", 0.0)) < float(min_q_margin_pa)
            q_bad = float(episode.get("max_q_dyn", 0.0)) > 70_000.0
            g_bad = float(episode.get("max_g_load", 0.0)) > 8.5
            if failed or q_low_margin or q_bad or g_bad:
                selected.append((case, episode))
                if max_episodes is not None and len(selected) >= int(max_episodes):
                    return selected
    return selected


def collect_stress_failure_dagger_dataset(
    *,
    stress_eval_path: str | Path,
    expert_dataset_path: str | Path,
    model_path: str | Path,
    vecnorm_path: str | Path,
    out_dir: str | Path = "artifacts/living/artifacts_dagger_robust",
    target_altitude_m: float = 100_000.0,
    dagger_weight: float = 4.0,
    expert_weight: float = 1.0,
    min_q_margin_pa: float = 5_000.0,
    max_failure_episodes: int | None = 100,
    max_steps: int | None = None,
) -> dict[str, Any]:
    expert = load_expert_dataset(expert_dataset_path)
    stress_eval = json.loads(Path(stress_eval_path).read_text())
    selected = _select_stress_failures(
        stress_eval,
        min_q_margin_pa=min_q_margin_pa,
        max_episodes=max_failure_episodes,
    )
    if not selected:
        raise ValueError("stress eval contained no failed or low-margin trained episodes to relabel")

    model = PPO.load(str(model_path))
    vecnorm = load_vecnormalize_for_eval(str(vecnorm_path), target_altitude_m=target_altitude_m)
    label_controller = fixed_action_controller("q_bucket", dict(DEFAULT_QBUCKET_PARAMS[2]))

    dagger_obs: list[np.ndarray] = []
    dagger_actions: list[np.ndarray] = []
    dagger_episode_ids: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []

    for episode_idx, (case, source_episode) in enumerate(selected):
        env = build_stress_env(case, target_altitude_m=target_altitude_m)
        seed = int(source_episode["seed"])
        obs, info = env.reset(seed=seed)
        terminated = False
        truncated = False
        step_count = 0
        ep_obs: list[np.ndarray] = []
        ep_actions: list[np.ndarray] = []
        while not (terminated or truncated):
            label_action = label_controller(env, info)
            ep_obs.append(np.asarray(obs, dtype=np.float32).copy())
            ep_actions.append(np.asarray(label_action, dtype=np.float32).copy())

            policy_action, _ = model.predict(normalise_obs(obs, vecnorm), deterministic=True)
            obs, _, terminated, truncated, info = env.step(policy_action)
            step_count += 1
            if max_steps is not None and step_count >= int(max_steps):
                break

        dagger_obs.append(np.asarray(ep_obs, dtype=np.float32))
        dagger_actions.append(np.asarray(ep_actions, dtype=np.float32))
        dagger_episode_ids.append(np.full(len(ep_actions), episode_idx, dtype=np.int32))
        metric = _episode_metrics(
            info,
            seed=seed,
            episode=episode_idx,
            steps=step_count,
            label_params=dict(DEFAULT_QBUCKET_PARAMS[2]),
        )
        metric["stress_case"] = case.name
        metric["source_episode"] = source_episode
        metrics.append(metric)
        env.close()
        print(
            f"stress dagger {episode_idx + 1}/{len(selected)} case={case.name}: "
            f"policy_success={int(metric['success'])} alt={metric['max_altitude_m']:.1f} "
            f"q={metric['max_q_dyn']:.1f} steps={step_count}"
        )

    vecnorm.close()

    summary = _save_combined_dataset(
        expert=expert,
        dagger_obs=dagger_obs,
        dagger_actions=dagger_actions,
        dagger_episode_ids=dagger_episode_ids,
        metrics=metrics,
        out_dir=Path(out_dir),
        expert_dataset_path=expert_dataset_path,
        model_path=model_path,
        vecnorm_path=vecnorm_path,
        episodes=len(selected),
        dagger_weight=dagger_weight,
        expert_weight=expert_weight,
    )
    summary["stress_eval_path"] = str(stress_eval_path)
    summary["min_q_margin_pa"] = float(min_q_margin_pa)
    summary["max_failure_episodes"] = None if max_failure_episodes is None else int(max_failure_episodes)
    (Path(out_dir) / "dagger_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect DAgger relabeled states from a BC policy")
    parser.add_argument("--expert-dataset", type=str, default="artifacts/archived/artifacts_expert_solvable/trajectories.npz")
    parser.add_argument("--model-path", type=str, default="artifacts/archived/artifacts_bc_solvable/bc_model.zip")
    parser.add_argument("--vecnorm-path", type=str, default="artifacts/archived/artifacts_bc_solvable/vecnormalize.pkl")
    parser.add_argument("--out-dir", type=str, default="artifacts/living/artifacts_dagger_solvable")
    parser.add_argument("--episodes", type=int, default=40)
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--dagger-weight", type=float, default=3.0)
    parser.add_argument("--expert-weight", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--stress-eval-json", type=str, default=None, help="Collect DAgger data from failed/low-margin stress eval episodes")
    parser.add_argument("--min-q-margin-pa", type=float, default=5_000.0, help="Stress episodes below this q margin are relabeled")
    parser.add_argument("--max-failure-episodes", type=int, default=100, help="Maximum stress failure/low-margin episodes to relabel")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.stress_eval_json is not None:
        collect_stress_failure_dagger_dataset(
            stress_eval_path=args.stress_eval_json,
            expert_dataset_path=args.expert_dataset,
            model_path=args.model_path,
            vecnorm_path=args.vecnorm_path,
            out_dir=args.out_dir,
            dagger_weight=args.dagger_weight,
            expert_weight=args.expert_weight,
            min_q_margin_pa=args.min_q_margin_pa,
            max_failure_episodes=args.max_failure_episodes,
            max_steps=args.max_steps,
        )
    else:
        collect_dagger_dataset(
            expert_dataset_path=args.expert_dataset,
            model_path=args.model_path,
            vecnorm_path=args.vecnorm_path,
            out_dir=args.out_dir,
            episodes=args.episodes,
            seed=args.seed,
            dagger_weight=args.dagger_weight,
            expert_weight=args.expert_weight,
            max_steps=args.max_steps,
        )
