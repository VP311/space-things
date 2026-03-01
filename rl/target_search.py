"""Search for the highest stable starting target altitude for curriculum."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rl.train import train


def _load_episode_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _learning_score(rows: list[dict], target_altitude_m: float, window: int = 30) -> tuple[float, float, float]:
    if not rows:
        return 0.0, 0.0, 0.0
    recent = rows[-window:]
    success_rate = sum(1.0 for r in recent if bool(r.get("success", False))) / max(len(recent), 1)
    mean_max_alt = sum(float(r.get("max_altitude_m", 0.0)) for r in recent) / max(len(recent), 1)
    altitude_ratio = min(mean_max_alt / max(target_altitude_m, 1.0), 1.5)
    score = 0.55 * success_rate + 0.45 * altitude_ratio
    return score, success_rate, mean_max_alt


def search_best_start_target(
    rounds: int = 4,
    timesteps_per_round: int = 250_000,
    n_envs: int = 4,
    initial_target_altitude_m: float = 2_000.0,
    min_target_altitude_m: float = 800.0,
    max_target_altitude_m: float = 40_000.0,
) -> dict:
    candidate = float(initial_target_altitude_m)
    best_target = candidate
    best_score = -1.0
    trials: list[dict] = []

    for i in range(rounds):
        train(
            total_timesteps=timesteps_per_round,
            n_envs=n_envs,
            initial_target_altitude_m=candidate,
            curriculum_enabled=False,
        )
        rows = _load_episode_rows(Path("artifacts/episode_summaries.jsonl"))
        score, success_rate, mean_max_alt = _learning_score(rows, target_altitude_m=candidate)
        trials.append(
            {
                "round": i + 1,
                "candidate_target_altitude_m": candidate,
                "score": score,
                "success_rate": success_rate,
                "mean_max_altitude_m": mean_max_alt,
            }
        )

        if score > best_score + 0.02:
            best_score = score
            best_target = candidate
            candidate = min(candidate * 1.25, max_target_altitude_m)
        else:
            candidate = max(candidate * 0.8, min_target_altitude_m)

    growth_rate = 1.2 if best_score >= 0.7 else 1.12
    result = {
        "recommended_start_target_altitude_m": best_target,
        "recommended_curriculum_growth_rate": growth_rate,
        "best_score": best_score,
        "trials": trials,
    }

    out = Path("artifacts/target_search.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"saved target search results: {out}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search for best curriculum start altitude")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--timesteps-per-round", type=int, default=250_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--initial-target-altitude-m", type=float, default=2_000.0)
    parser.add_argument("--min-target-altitude-m", type=float, default=800.0)
    parser.add_argument("--max-target-altitude-m", type=float, default=40_000.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    search_best_start_target(
        rounds=args.rounds,
        timesteps_per_round=args.timesteps_per_round,
        n_envs=args.n_envs,
        initial_target_altitude_m=args.initial_target_altitude_m,
        min_target_altitude_m=args.min_target_altitude_m,
        max_target_altitude_m=args.max_target_altitude_m,
    )
