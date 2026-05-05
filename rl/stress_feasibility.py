"""No-RL feasibility probes under named robustness stress cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from rl.feasibility import decode_feedback_params, fixed_action_controller, projected_burnout_apogee, rank_metrics
from rl.stress_eval import StressCase, build_stress_env, select_cases


def _episode_row(info: dict[str, Any], *, seed: int, steps: int) -> dict[str, Any]:
    max_q = float(info.get("max_q_dyn", 0.0))
    max_g = float(info.get("max_g_load", 0.0))
    mission = info.get("mission", {})
    row = {
        "seed": int(seed),
        "steps": int(steps),
        "success": bool(info.get("success", False)),
        "termination_reason": str(info.get("termination_reason", "unknown")),
        "max_altitude_m": float(info.get("max_altitude_m", 0.0)),
        "altitude_at_burnout_m": float(info.get("altitude_at_burnout_m", float("nan"))),
        "velocity_at_burnout_mps": float(info.get("velocity_at_burnout_mps", float("nan"))),
        "vz_at_burnout_mps": float(info.get("vz_at_burnout_mps", float("nan"))),
        "max_q_dyn": max_q,
        "max_g_load": max_g,
        "q_ok": bool(max_q <= float(mission.get("max_q_pa", 70_000.0))),
        "g_ok": bool(max_g <= float(mission.get("max_g_load", 8.5))),
        "fuel_used_fraction": float(info.get("fuel_used_fraction", 0.0)),
    }
    row["projected_burnout_apogee_m"] = projected_burnout_apogee(row)
    return row


def rollout_controller(
    case: StressCase,
    *,
    controller: Callable[[Any, dict[str, Any]], np.ndarray],
    seed: int,
    target_altitude_m: float,
) -> dict[str, Any]:
    env = build_stress_env(case, target_altitude_m=target_altitude_m)
    obs, info = env.reset(seed=int(seed))
    _ = obs
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        action = controller(env, info)
        obs, _, terminated, truncated, info = env.step(action)
        _ = obs
        steps += 1
    row = _episode_row(info, seed=seed, steps=steps)
    env.close()
    return row


def rollout_controller_in_env(
    env: Any,
    *,
    controller: Callable[[Any, dict[str, Any]], np.ndarray],
    seed: int,
) -> dict[str, Any]:
    obs, info = env.reset(seed=int(seed))
    _ = obs
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        action = controller(env, info)
        obs, _, terminated, truncated, info = env.step(action)
        _ = obs
        steps += 1
    return _episode_row(info, seed=seed, steps=steps)


def _controller_rank(summary: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return (
        float(summary["success_rate"]),
        float(summary["q_ok_rate"]) + float(summary["g_ok_rate"]),
        float(summary["avg_max_altitude_m"]),
        float(summary["avg_projected_burnout_apogee_m"]),
        float(summary["avg_burnout_vz_mps"]),
        -float(summary["avg_max_q"]),
    )


def summarise_controller(label: str, rows: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    success = np.array([row["success"] for row in rows], dtype=bool)
    q_ok = np.array([row["q_ok"] for row in rows], dtype=bool)
    g_ok = np.array([row["g_ok"] for row in rows], dtype=bool)
    max_alt = np.array([row["max_altitude_m"] for row in rows], dtype=float)
    max_q = np.array([row["max_q_dyn"] for row in rows], dtype=float)
    max_g = np.array([row["max_g_load"] for row in rows], dtype=float)
    burnout_vz = np.array([row["vz_at_burnout_mps"] for row in rows], dtype=float)
    projected = np.array([row["projected_burnout_apogee_m"] for row in rows], dtype=float)
    return {
        "label": label,
        "metadata": metadata,
        "n_episodes": len(rows),
        "success_rate": float(np.mean(success)) if rows else 0.0,
        "q_ok_rate": float(np.mean(q_ok)) if rows else 0.0,
        "g_ok_rate": float(np.mean(g_ok)) if rows else 0.0,
        "avg_max_altitude_m": float(np.mean(max_alt)) if rows else 0.0,
        "best_max_altitude_m": float(np.max(max_alt)) if rows else 0.0,
        "avg_projected_burnout_apogee_m": float(np.nanmean(projected)) if rows else 0.0,
        "avg_burnout_vz_mps": float(np.nanmean(burnout_vz)) if rows else 0.0,
        "avg_max_q": float(np.mean(max_q)) if rows else 0.0,
        "avg_max_g_load": float(np.mean(max_g)) if rows else 0.0,
        "episodes": rows,
    }


def fixed_controller_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"label": "vertical_full", "kind": "vertical_full", "params": {}}]
    for q_cap in (45_000.0, 50_000.0, 55_000.0, 60_000.0):
        for bucket in (0.4, 0.5, 0.6):
            specs.append(
                {
                    "label": f"qbucket_q{int(q_cap)}_b{bucket:.2f}_u60k",
                    "kind": "q_bucket",
                    "params": {"q_cap_pa": q_cap, "bucket": bucket, "high": 1.0, "until_alt_m": 60_000.0},
                }
            )
    for q_cap in (45_000.0, 50_000.0, 55_000.0, 60_000.0):
        for floor in (0.4, 0.5, 0.6):
            specs.append(
                {
                    "label": f"feedback_q{int(q_cap)}_f{floor:.2f}",
                    "kind": "feedback",
                    "params": {
                        "q_cap_pa": q_cap,
                        "floor": floor,
                        "high": 1.0,
                        "gain": 3.5,
                        "bucket_until_alt_m": 60_000.0,
                        "pitch_start_s": 45.0,
                        "pitch_mid_s": 130.0,
                        "gimbal_norm": 0.15,
                    },
                }
            )
    return specs


def evaluate_controller_spec(
    case: StressCase,
    spec: dict[str, Any],
    *,
    seed_start: int,
    n_episodes: int,
    target_altitude_m: float,
) -> dict[str, Any]:
    controller = fixed_action_controller(spec["kind"], spec["params"])
    env = build_stress_env(case, target_altitude_m=target_altitude_m)
    rows = [
        rollout_controller_in_env(env, controller=controller, seed=seed_start + idx)
        for idx in range(int(n_episodes))
    ]
    env.close()
    return summarise_controller(spec["label"], rows, {"controller": spec["kind"], "params": spec["params"]})


def _candidate_score(summary: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    return _controller_rank(summary)


def cem_search_case(
    case: StressCase,
    *,
    seed: int,
    train_episodes: int,
    population: int,
    generations: int,
    restarts: int,
    target_altitude_m: float,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    means = [
        np.array([0.0, -0.25, 1.0, 0.2, 0.5, -0.35, 0.0, -0.7], dtype=np.float32),
        np.array([-0.5, -0.5, 1.0, 0.5, 0.8, -0.6, -0.2, -1.0], dtype=np.float32),
        np.array([0.4, -0.1, 1.0, 0.4, 0.4, -0.2, 0.1, -0.8], dtype=np.float32),
    ]
    summaries: list[dict[str, Any]] = []
    for restart in range(int(restarts)):
        mean = means[restart % len(means)].copy()
        sigma = np.full(8, 0.55, dtype=np.float32)
        for generation in range(int(generations)):
            samples = np.clip(rng.normal(mean, sigma, size=(int(population), 8)), -1.5, 1.5).astype(np.float32)
            generation_summaries: list[dict[str, Any]] = []
            for idx, raw in enumerate(samples):
                params = decode_feedback_params(raw)
                spec = {
                    "label": f"cem_r{restart}_g{generation}_i{idx}",
                    "kind": "feedback",
                    "params": params,
                }
                summary = evaluate_controller_spec(
                    case,
                    spec,
                    seed_start=seed + restart * 100_000 + generation * 1_000 + idx * 100,
                    n_episodes=train_episodes,
                    target_altitude_m=target_altitude_m,
                )
                summary["metadata"]["raw"] = raw.tolist()
                summary["metadata"]["restart"] = restart
                summary["metadata"]["generation"] = generation
                generation_summaries.append(summary)
            generation_summaries.sort(key=_candidate_score, reverse=True)
            elite_count = max(2, int(population) // 4)
            elite_raw = np.asarray([row["metadata"]["raw"] for row in generation_summaries[:elite_count]], dtype=np.float32)
            mean = np.mean(elite_raw, axis=0)
            sigma = np.maximum(np.std(elite_raw, axis=0), 0.08).astype(np.float32)
            summaries.extend(generation_summaries[:5])
            best = generation_summaries[0]
            print(
                f"{case.name} cem r={restart} g={generation + 1}/{generations} "
                f"success={best['success_rate']:.2f} alt={best['avg_max_altitude_m']:.1f} "
                f"q_ok={best['q_ok_rate']:.2f} q={best['avg_max_q']:.1f}"
            )
    return summaries


def evaluate_case_feasibility(
    case: StressCase,
    *,
    seed_start: int,
    n_episodes: int,
    target_altitude_m: float,
    cem_population: int,
    cem_generations: int,
    cem_restarts: int,
    cem_train_episodes: int,
    skip_cem: bool,
) -> dict[str, Any]:
    rows = []
    for spec in fixed_controller_specs():
        summary = evaluate_controller_spec(
            case,
            spec,
            seed_start=seed_start,
            n_episodes=n_episodes,
            target_altitude_m=target_altitude_m,
        )
        rows.append(summary)
    if not skip_cem:
        rows.extend(
            cem_search_case(
                case,
                seed=seed_start + 50_000,
                train_episodes=cem_train_episodes,
                population=cem_population,
                generations=cem_generations,
                restarts=cem_restarts,
                target_altitude_m=target_altitude_m,
            )
        )
    rows.sort(key=_candidate_score, reverse=True)
    best = rows[0] if rows else None
    return {
        "case": asdict(case),
        "n_controllers": len(rows),
        "best": {key: value for key, value in best.items() if key != "episodes"} if best else None,
        "top_controllers": [{key: value for key, value in row.items() if key != "episodes"} for row in rows[:10]],
        "controllers": rows,
    }


def run_stress_feasibility(
    *,
    out_dir: str | Path = "artifacts/living/artifacts_robustness/stress_feasibility",
    case_names: str = "isp_0.85x,joint_harsh_propulsion,joint_worst_plausible",
    seed_start: int = 24_000,
    n_episodes: int = 25,
    target_altitude_m: float = 100_000.0,
    cem_population: int = 12,
    cem_generations: int = 3,
    cem_restarts: int = 2,
    cem_train_episodes: int = 3,
    skip_cem: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    selected = select_cases(case_names)
    case_results = []
    for idx, case in enumerate(selected):
        print(f"stress feasibility case {case.name}")
        case_results.append(
            evaluate_case_feasibility(
                case,
                seed_start=seed_start + idx * 10_000,
                n_episodes=n_episodes,
                target_altitude_m=target_altitude_m,
                cem_population=cem_population,
                cem_generations=cem_generations,
                cem_restarts=cem_restarts,
                cem_train_episodes=cem_train_episodes,
                skip_cem=skip_cem,
            )
        )
    result = {
        "eval": {
            "case_names": case_names,
            "seed_start": int(seed_start),
            "n_episodes_per_fixed_controller": int(n_episodes),
            "target_altitude_m": float(target_altitude_m),
            "cem_population": int(cem_population),
            "cem_generations": int(cem_generations),
            "cem_restarts": int(cem_restarts),
            "cem_train_episodes": int(cem_train_episodes),
            "skip_cem": bool(skip_cem),
        },
        "cases": case_results,
    }
    out_path = out / "stress_feasibility.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"saved stress feasibility: {out_path}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run no-RL feasibility probes under named robustness stress cases")
    parser.add_argument("--out-dir", type=str, default="artifacts/living/artifacts_robustness/stress_feasibility")
    parser.add_argument("--cases", type=str, default="isp_0.85x,joint_harsh_propulsion,joint_worst_plausible")
    parser.add_argument("--seed-start", type=int, default=24_000)
    parser.add_argument("--n-episodes", type=int, default=25)
    parser.add_argument("--target-altitude-m", type=float, default=100_000.0)
    parser.add_argument("--cem-population", type=int, default=12)
    parser.add_argument("--cem-generations", type=int, default=3)
    parser.add_argument("--cem-restarts", type=int, default=2)
    parser.add_argument("--cem-train-episodes", type=int, default=3)
    parser.add_argument("--skip-cem", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_stress_feasibility(
        out_dir=args.out_dir,
        case_names=args.cases,
        seed_start=args.seed_start,
        n_episodes=args.n_episodes,
        target_altitude_m=args.target_altitude_m,
        cem_population=args.cem_population,
        cem_generations=args.cem_generations,
        cem_restarts=args.cem_restarts,
        cem_train_episodes=args.cem_train_episodes,
        skip_cem=args.skip_cem,
    )
