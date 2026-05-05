"""Open-loop expert trajectory search for 100 km compliant ascents."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from config.defaults import AtmosphereConfig, CurriculumConfig, EnvConfig, MissionConfig, default_vehicle_params
from rl.env import RocketAscentEnv
from sim.constants import G0


DEFAULT_TARGET_ALTITUDE_M = 100_000.0
DEFAULT_KNOT_TIMES_S = np.array([0.0, 6.0, 14.0, 28.0, 48.0, 72.0, 100.0, 130.0, 165.0], dtype=np.float32)


@dataclass(frozen=True)
class ExpertRollout:
    observations: np.ndarray
    actions: np.ndarray
    metrics: dict[str, Any]


def initial_mean(n_knots: int = len(DEFAULT_KNOT_TIMES_S)) -> np.ndarray:
    throttle = np.linspace(1.0, 0.70, n_knots, dtype=np.float32)
    gimbal_norm = np.array([0.0, 0.0, -0.12, -0.25, -0.38, -0.42, -0.32, -0.22, -0.12], dtype=np.float32)
    if n_knots != len(gimbal_norm):
        x_old = np.linspace(0.0, 1.0, len(gimbal_norm))
        x_new = np.linspace(0.0, 1.0, n_knots)
        gimbal_norm = np.interp(x_new, x_old, gimbal_norm).astype(np.float32)
    return np.concatenate([throttle, gimbal_norm]).astype(np.float32)


def initial_sigma(n_knots: int = len(DEFAULT_KNOT_TIMES_S)) -> np.ndarray:
    throttle_sigma = np.full(n_knots, 0.18, dtype=np.float32)
    gimbal_sigma = np.full(n_knots, 0.30, dtype=np.float32)
    return np.concatenate([throttle_sigma, gimbal_sigma]).astype(np.float32)


def decode_schedule(
    params: np.ndarray,
    *,
    max_gimbal: float,
    knot_times_s: np.ndarray = DEFAULT_KNOT_TIMES_S,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    params = np.asarray(params, dtype=np.float32).reshape(-1)
    n_knots = len(knot_times_s)
    expected = 2 * n_knots
    if params.shape[0] != expected:
        raise ValueError(f"expected {expected} parameters for {n_knots} knots, got {params.shape[0]}")
    throttle = np.clip(params[:n_knots], 0.0, 1.0).astype(np.float32)
    gimbal = (np.clip(params[n_knots:], -1.0, 1.0) * float(max_gimbal)).astype(np.float32)
    return np.asarray(knot_times_s, dtype=np.float32), throttle, gimbal


def action_at_time(
    params: np.ndarray,
    t_s: float,
    *,
    max_gimbal: float,
    knot_times_s: np.ndarray = DEFAULT_KNOT_TIMES_S,
) -> np.ndarray:
    times, throttle, gimbal = decode_schedule(params, max_gimbal=max_gimbal, knot_times_s=knot_times_s)
    action = np.array(
        [
            np.interp(float(t_s), times, throttle),
            np.interp(float(t_s), times, gimbal),
        ],
        dtype=np.float32,
    )
    return action


def build_search_env(*, deterministic: bool = True, record: bool = False) -> RocketAscentEnv:
    env_cfg = EnvConfig()
    mission = MissionConfig()
    curriculum_cfg = CurriculumConfig()
    atmosphere_cfg = AtmosphereConfig(randomize=not deterministic)
    env = RocketAscentEnv(
        params=default_vehicle_params(),
        mission=mission,
        atmosphere_cfg=atmosphere_cfg,
        curriculum_milestones_m=curriculum_cfg.target_milestones_m,
        dt=env_cfg.dt,
        t_final=env_cfg.t_final,
        record=record,
        start_state_randomization=not deterministic,
        domain_randomization=not deterministic,
        action_repeat=env_cfg.action_repeat,
    )
    env.set_curriculum(target_altitude_m=DEFAULT_TARGET_ALTITUDE_M, start_altitude_cap_m=0.0)
    return env


def _projected_burnout_apogee(metrics: dict[str, Any]) -> float:
    burnout_alt = float(metrics.get("altitude_at_burnout_m", float("nan")))
    burnout_vz = float(metrics.get("vz_at_burnout_mps", float("nan")))
    if not np.isfinite(burnout_alt) or not np.isfinite(burnout_vz):
        return float(metrics.get("max_altitude_m", 0.0))
    return float(burnout_alt + (burnout_vz**2) / (2.0 * G0))


def score_metrics(metrics: dict[str, Any]) -> float:
    target = DEFAULT_TARGET_ALTITUDE_M
    max_q = float(metrics.get("max_q_dyn", 0.0))
    max_g = float(metrics.get("max_g_load", 0.0))
    max_alt = float(metrics.get("max_altitude_m", 0.0))
    burnout_vz = float(metrics.get("vz_at_burnout_mps", 0.0))
    projected_apogee = float(metrics.get("projected_burnout_apogee_m", _projected_burnout_apogee(metrics)))
    success = bool(metrics.get("success", False))
    q_ok = bool(metrics.get("q_ok", max_q <= MissionConfig().max_q_pa))
    g_ok = bool(metrics.get("g_ok", max_g <= MissionConfig().max_g_load))

    q_limit = MissionConfig().max_q_pa
    g_limit = MissionConfig().max_g_load
    q_margin = (q_limit - max_q) / max(q_limit, 1.0)
    g_margin = (g_limit - max_g) / max(g_limit, 1.0)
    q_penalty = 500_000.0 * max(0.0, -q_margin) ** 2
    g_penalty = 500_000.0 * max(0.0, -g_margin) ** 2

    compliance_bonus = 80_000.0 if q_ok and g_ok else 0.0
    success_bonus = 1_000_000.0 if success and q_ok and g_ok else 0.0
    apogee_score = 35_000.0 * np.clip(projected_apogee / target, 0.0, 1.25)
    altitude_score = 25_000.0 * np.clip(max_alt / target, 0.0, 1.25)
    burnout_vz_score = 8_000.0 * np.clip(burnout_vz / 1_250.0, 0.0, 1.25)
    margin_score = 1_500.0 * np.clip(q_margin, -1.0, 1.0) + 1_500.0 * np.clip(g_margin, -1.0, 1.0)
    return float(success_bonus + compliance_bonus + apogee_score + altitude_score + burnout_vz_score + margin_score - q_penalty - g_penalty)


def rollout_schedule(
    params: np.ndarray,
    *,
    seed: int,
    deterministic: bool = True,
    record: bool = True,
) -> ExpertRollout:
    env = build_search_env(deterministic=deterministic, record=record)
    obs, _ = env.reset(seed=int(seed))
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    terminated = False
    truncated = False
    info: dict[str, Any] = {}

    while not (terminated or truncated):
        action = action_at_time(params, env.t, max_gimbal=env.params.max_gimbal)
        observations.append(np.asarray(obs, dtype=np.float32).copy())
        actions.append(action.copy())
        obs, _, terminated, truncated, info = env.step(action)

    max_q = float(info.get("max_q_dyn", 0.0))
    max_g = float(info.get("max_g_load", 0.0))
    metrics: dict[str, Any] = {
        "seed": int(seed),
        "success": bool(info.get("success", False)),
        "termination_reason": str(info.get("termination_reason", "unknown")),
        "max_altitude_m": float(info.get("max_altitude_m", 0.0)),
        "altitude_at_apogee_m": float(info.get("altitude_at_apogee_m", info.get("max_altitude_m", 0.0))),
        "altitude_at_burnout_m": float(info.get("altitude_at_burnout_m", float("nan"))),
        "velocity_at_burnout_mps": float(info.get("velocity_at_burnout_mps", float("nan"))),
        "vz_at_burnout_mps": float(info.get("vz_at_burnout_mps", float("nan"))),
        "max_q_dyn": max_q,
        "max_g_load": max_g,
        "q_ok": bool(max_q <= env.mission.max_q_pa),
        "g_ok": bool(max_g <= env.mission.max_g_load),
        "steps": len(actions),
        "smoothness": float(np.mean(np.linalg.norm(np.diff(np.asarray(actions), axis=0), axis=1))) if len(actions) > 1 else 0.0,
    }
    metrics["projected_burnout_apogee_m"] = _projected_burnout_apogee(metrics)
    metrics["fitness"] = score_metrics(metrics)
    env.close()
    return ExpertRollout(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        metrics=metrics,
    )


def save_trajectories(rollouts: list[ExpertRollout], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rollouts:
        raise ValueError("no trajectories to save")

    observations = np.concatenate([rollout.observations for rollout in rollouts], axis=0).astype(np.float32)
    actions = np.concatenate([rollout.actions for rollout in rollouts], axis=0).astype(np.float32)
    episode_ids = np.concatenate(
        [np.full(len(rollout.actions), idx, dtype=np.int32) for idx, rollout in enumerate(rollouts)],
        axis=0,
    )
    episode_starts = np.cumsum([0] + [len(rollout.actions) for rollout in rollouts[:-1]]).astype(np.int32)
    metrics_json = np.asarray([json.dumps(rollout.metrics, sort_keys=True) for rollout in rollouts])
    np.savez_compressed(
        out_path,
        observations=observations,
        actions=actions,
        episode_ids=episode_ids,
        episode_starts=episode_starts,
        metrics_json=metrics_json,
    )


def cem_search(
    *,
    population: int = 128,
    elites: int = 16,
    generations: int = 60,
    seed: int = 0,
    out_dir: str | Path = "artifacts/living/artifacts_expert",
    deterministic: bool = True,
    save_min_altitude_m: float = 98_000.0,
) -> dict[str, Any]:
    if population < 2:
        raise ValueError("population must be at least 2")
    if elites < 1 or elites > population:
        raise ValueError("elites must be in [1, population]")

    rng = np.random.default_rng(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results_path = out / "search_results.jsonl"
    results_path.write_text("")

    mean = initial_mean()
    sigma = initial_sigma()
    best: ExpertRollout | None = None
    saved: list[ExpertRollout] = []

    with results_path.open("a", encoding="utf-8") as fh:
        for generation in range(int(generations)):
            params_batch = rng.normal(mean, sigma, size=(int(population), mean.size)).astype(np.float32)
            rollouts = [
                rollout_schedule(params_batch[idx], seed=seed + generation * population + idx, deterministic=deterministic, record=False)
                for idx in range(int(population))
            ]
            fitness = np.asarray([rollout.metrics["fitness"] for rollout in rollouts], dtype=np.float64)
            order = np.argsort(fitness)[::-1]
            elite_params = params_batch[order[: int(elites)]]
            mean = np.mean(elite_params, axis=0).astype(np.float32)
            sigma = np.maximum(np.std(elite_params, axis=0), 0.03).astype(np.float32)
            generation_best = rollouts[int(order[0])]
            if best is None or generation_best.metrics["fitness"] > best.metrics["fitness"]:
                best = rollout_schedule(mean, seed=seed + 100_000 + generation, deterministic=deterministic, record=True)

            for rank, idx in enumerate(order[: min(5, len(order))]):
                row = dict(rollouts[int(idx)].metrics)
                row["generation"] = generation
                row["rank"] = rank
                fh.write(json.dumps(row) + "\n")
            fh.flush()

            for idx in order[: int(elites)]:
                candidate = rollouts[int(idx)]
                if (
                    candidate.metrics["max_altitude_m"] >= save_min_altitude_m
                    and candidate.metrics["q_ok"]
                    and candidate.metrics["g_ok"]
                ):
                    saved.append(rollout_schedule(params_batch[int(idx)], seed=seed + 200_000 + len(saved), deterministic=deterministic, record=True))

            print(
                "generation "
                f"{generation + 1}/{generations}: "
                f"best_fitness={generation_best.metrics['fitness']:.1f} "
                f"success={int(generation_best.metrics['success'])} "
                f"alt={generation_best.metrics['max_altitude_m']:.1f} "
                f"vz={generation_best.metrics['vz_at_burnout_mps']:.1f} "
                f"q={generation_best.metrics['max_q_dyn']:.1f} "
                f"g={generation_best.metrics['max_g_load']:.2f}"
            )

    if not saved and best is not None:
        saved.append(best)
    if not saved:
        raise RuntimeError("search produced no rollouts")

    save_trajectories(saved, out / "trajectories.npz")
    summary = {
        "population": int(population),
        "elites": int(elites),
        "generations": int(generations),
        "seed": int(seed),
        "deterministic": bool(deterministic),
        "saved_trajectories": len(saved),
        "best_metrics": best.metrics if best is not None else None,
        "trajectories_path": str(out / "trajectories.npz"),
        "results_path": str(results_path),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search open-loop expert ascent trajectories with CEM")
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--elites", type=int, default=16)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="artifacts/living/artifacts_expert")
    parser.add_argument("--save-min-altitude-m", type=float, default=98_000.0)
    parser.add_argument("--randomized", action="store_true", help="Enable atmosphere/start/vehicle randomization during search")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cem_search(
        population=args.population,
        elites=args.elites,
        generations=args.generations,
        seed=args.seed,
        out_dir=args.out_dir,
        deterministic=not args.randomized,
        save_min_altitude_m=args.save_min_altitude_m,
    )
