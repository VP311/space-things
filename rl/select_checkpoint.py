"""Select the best checkpoint using fixed/official eval histories."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _compact_row(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {key: value for key, value in row.items() if key != "episodes"}


def _metric(row: dict[str, Any] | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return default
    return float(row.get(key, default))


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def _count_from_rate(row: dict[str, Any], key: str, n: int) -> int:
    return int(round(float(row.get(key, 0.0)) * n))


def wilson_summary(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    n = int(row.get("n_episodes", len(row.get("episodes", [])) or 0))
    success_count = _count_from_rate(row, "success_rate", n)
    q_ok_count = _count_from_rate(row, "q_ok_rate", n)
    g_ok_count = _count_from_rate(row, "g_ok_rate", n)
    success_lcb, success_ucb = wilson_interval(success_count, n)
    q_lcb, q_ucb = wilson_interval(q_ok_count, n)
    g_lcb, g_ucb = wilson_interval(g_ok_count, n)
    score = success_lcb * q_lcb * g_lcb
    return {
        "n_episodes": n,
        "success_count": success_count,
        "q_ok_count": q_ok_count,
        "g_ok_count": g_ok_count,
        "success_rate": float(row.get("success_rate", 0.0)),
        "q_ok_rate": float(row.get("q_ok_rate", 0.0)),
        "g_ok_rate": float(row.get("g_ok_rate", 0.0)),
        "success_ci_95": [success_lcb, success_ucb],
        "q_ok_ci_95": [q_lcb, q_ucb],
        "g_ok_ci_95": [g_lcb, g_ucb],
        "success_lcb": success_lcb,
        "q_ok_lcb": q_lcb,
        "g_ok_lcb": g_lcb,
        "score": score,
    }


def _with_wilson(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate = dict(candidate)
    candidate["fixed_wilson"] = wilson_summary(candidate.get("fixed"))
    candidate["official_wilson"] = wilson_summary(candidate.get("official"))
    primary = candidate["official_wilson"] if candidate["official_wilson"] is not None else candidate["fixed_wilson"]
    candidate["primary_wilson"] = primary
    return candidate


def rank_key(candidate: dict[str, Any]) -> tuple[float, float, float, float, float, float, float, float]:
    candidate = _with_wilson(candidate)
    official = candidate.get("official_wilson")
    fixed = candidate.get("fixed_wilson")
    primary_wilson = official if official is not None else fixed
    primary_metrics = candidate.get("official") if candidate.get("official") is not None else candidate.get("fixed")
    primary_wilson = primary_wilson or {}
    return (
        float(primary_wilson.get("score", 0.0)),
        float((fixed or {}).get("score", 0.0)),
        float(primary_wilson.get("success_lcb", 0.0)),
        float(primary_wilson.get("q_ok_lcb", 0.0)),
        float(primary_wilson.get("g_ok_lcb", 0.0)),
        _metric(primary_metrics, "avg_q_margin_pa"),
        _metric(primary_metrics, "avg_burnout_vz_mps"),
        _metric(primary_metrics, "avg_max_altitude_m"),
    )


def select_checkpoint(artifacts_dir: str | Path) -> dict[str, Any]:
    artifacts = Path(artifacts_dir)
    eval_logs = artifacts / "eval_logs"
    checkpoints = artifacts / "checkpoints"

    fixed_rows = {
        int(row["num_timesteps"]): row
        for row in _read_jsonl(eval_logs / "fixed_final_eval_history.jsonl")
        if "num_timesteps" in row
    }
    official_rows = {
        int(row["num_timesteps"]): row
        for row in _read_jsonl(eval_logs / "official_fixed_final_eval_history.jsonl")
        if "num_timesteps" in row
    }
    candidates: list[dict[str, Any]] = []
    for step in sorted(set(fixed_rows) | set(official_rows)):
        model_path = checkpoints / f"ppo_rocket_{step}_steps.zip"
        vecnorm_path = checkpoints / f"vecnormalize_{step}_steps.pkl"
        if not model_path.exists() or not vecnorm_path.exists():
            continue
        candidates.append(
            _with_wilson(
                {
                    "num_timesteps": step,
                    "model_path": str(model_path),
                    "vecnormalize_path": str(vecnorm_path),
                    "fixed": _compact_row(fixed_rows.get(step)),
                    "official": _compact_row(official_rows.get(step)),
                }
            )
        )

    if not candidates:
        raise FileNotFoundError(f"no checkpoints with matching VecNormalize stats found under {checkpoints}")

    best = max(candidates, key=rank_key)
    result = {
        "selected": best,
        "rank_key": list(rank_key(best)),
        "rank_method": "Wilson 95% LCB score = success_lcb * q_ok_lcb * g_ok_lcb; official eval preferred when present",
        "n_candidates": len(candidates),
        "candidates": sorted(candidates, key=rank_key, reverse=True),
        "artifacts_dir": str(artifacts),
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select best PPO checkpoint from eval histories")
    parser.add_argument("--artifacts-dir", type=str, required=True)
    parser.add_argument("--out-path", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    selection = select_checkpoint(args.artifacts_dir)
    text = json.dumps(selection, indent=2)
    print(text)
    if args.out_path is not None:
        Path(args.out_path).write_text(text + "\n")
