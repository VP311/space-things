"""Export scripted q-bucket expert trajectories for behavior cloning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rl.feasibility import fixed_action_controller, rollout_controller

DEFAULT_Q_CAPS_PA = (45_000.0, 50_000.0, 55_000.0, 60_000.0)
DEFAULT_BUCKETS = (0.4, 0.5, 0.6)
DEFAULT_BUCKET_UNTIL_ALT_M = 60_000.0


def qbucket_specs(
    *,
    q_caps_pa: tuple[float, ...] = DEFAULT_Q_CAPS_PA,
    buckets: tuple[float, ...] = DEFAULT_BUCKETS,
    until_alt_m: float = DEFAULT_BUCKET_UNTIL_ALT_M,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for q_cap in q_caps_pa:
        for bucket in buckets:
            params = {
                "q_cap_pa": float(q_cap),
                "bucket": float(bucket),
                "high": 1.0,
                "until_alt_m": float(until_alt_m),
            }
            specs.append(
                {
                    "label": f"scripted_qbucket_q{int(q_cap)}_b{bucket:.2f}",
                    "kind": "q_bucket",
                    "params": params,
                }
            )
    return specs


def _json_ready(row: dict[str, Any]) -> dict[str, Any]:
    clean = {k: v for k, v in row.items() if k not in {"observations", "actions"}}
    if isinstance(clean.get("rank"), tuple):
        clean["rank"] = list(clean["rank"])
    return clean


def save_expert_rollouts(rollouts: list[dict[str, Any]], path: Path) -> None:
    if not rollouts:
        raise ValueError("no successful expert rollouts to save")
    path.parent.mkdir(parents=True, exist_ok=True)
    observations = np.concatenate([row["observations"] for row in rollouts], axis=0).astype(np.float32)
    actions = np.concatenate([row["actions"] for row in rollouts], axis=0).astype(np.float32)
    episode_ids = np.concatenate([np.full(len(row["actions"]), idx, dtype=np.int32) for idx, row in enumerate(rollouts)])
    episode_starts = np.cumsum([0] + [len(row["actions"]) for row in rollouts[:-1]]).astype(np.int32)
    metrics_json = np.asarray([json.dumps(_json_ready(row), sort_keys=True) for row in rollouts])
    np.savez_compressed(
        path,
        observations=observations,
        actions=actions,
        episode_ids=episode_ids,
        episode_starts=episode_starts,
        metrics_json=metrics_json,
    )


def export_scripted_experts(
    *,
    out_dir: str | Path = "artifacts/living/artifacts_expert_solvable",
    seed: int = 37,
    q_caps_pa: tuple[float, ...] = DEFAULT_Q_CAPS_PA,
    buckets: tuple[float, ...] = DEFAULT_BUCKETS,
    until_alt_m: float = DEFAULT_BUCKET_UNTIL_ALT_M,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []

    for idx, spec in enumerate(qbucket_specs(q_caps_pa=q_caps_pa, buckets=buckets, until_alt_m=until_alt_m)):
        row = rollout_controller(
            label=spec["label"],
            controller=fixed_action_controller(spec["kind"], spec["params"]),
            seed=seed + idx,
            record=True,
            metadata={"controller": spec["kind"], "variant": "current", "params": spec["params"]},
        )
        rows.append(row)
        if row["success"] and row["q_ok"] and row["g_ok"]:
            successful.append(row)
        print(
            f"{row['label']}: success={int(row['success'])} "
            f"alt={row['max_altitude_m']:.1f} q={row['max_q_dyn']:.1f} "
            f"g={row['max_g_load']:.2f} class={row['failure_class']}"
        )

    results_path = out / "scripted_results.jsonl"
    with results_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_json_ready(row)) + "\n")

    trajectories_path = out / "trajectories.npz"
    save_expert_rollouts(successful, trajectories_path)
    n_samples = int(sum(len(row["actions"]) for row in successful))
    summary = {
        "seed": int(seed),
        "q_caps_pa": [float(v) for v in q_caps_pa],
        "buckets": [float(v) for v in buckets],
        "until_alt_m": float(until_alt_m),
        "n_candidates": len(rows),
        "n_successful": len(successful),
        "n_samples": n_samples,
        "results_path": str(results_path),
        "trajectories_path": str(trajectories_path),
        "best_success": _json_ready(max(successful, key=lambda row: row["max_altitude_m"])) if successful else None,
    }
    (out / "scripted_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export scripted q-bucket expert trajectories")
    parser.add_argument("--out-dir", type=str, default="artifacts/living/artifacts_expert_solvable")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--until-alt-m", type=float, default=DEFAULT_BUCKET_UNTIL_ALT_M)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export_scripted_experts(out_dir=args.out_dir, seed=args.seed, until_alt_m=args.until_alt_m)
