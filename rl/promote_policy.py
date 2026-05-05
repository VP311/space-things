"""Create a promotion manifest for a selected policy checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from config.defaults import AtmosphereConfig, EnvConfig, MissionConfig, default_vehicle_params
from rl.select_checkpoint import wilson_interval


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _git_output(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def _count_metric(episodes: list[dict[str, Any]], predicate: str) -> int:
    if predicate == "success":
        return sum(bool(ep.get("success", False)) for ep in episodes)
    if predicate == "q_ok":
        return sum(float(ep.get("max_q_dyn", float("inf"))) <= MissionConfig().max_q_pa for ep in episodes)
    if predicate == "g_ok":
        return sum(float(ep.get("max_g_load", float("inf"))) <= MissionConfig().max_g_load for ep in episodes)
    raise ValueError(f"unknown predicate: {predicate}")


def _metric_summary(count: int, n: int) -> dict[str, Any]:
    lo, hi = wilson_interval(count, n)
    return {
        "count": int(count),
        "n": int(n),
        "rate": float(count / n if n else 0.0),
        "wilson_95_ci": [lo, hi],
        "wilson_lcb": lo,
    }


def build_manifest(
    *,
    model_path: str,
    vecnorm_path: str,
    eval_path: str,
    out_path: str | None = None,
    policy_label: str = "run37_selected_1100000",
) -> dict[str, Any]:
    model = Path(model_path)
    vecnorm = Path(vecnorm_path)
    eval_file = Path(eval_path)
    for path in (model, vecnorm, eval_file):
        if not path.exists():
            raise FileNotFoundError(path)

    eval_data = json.loads(eval_file.read_text())
    trained = eval_data.get("trained", {})
    episodes = trained.get("episodes", [])
    if not episodes:
        raise ValueError(f"eval file has no trained episodes: {eval_file}")
    n = len(episodes)
    success_count = _count_metric(episodes, "success")
    q_ok_count = _count_metric(episodes, "q_ok")
    g_ok_count = _count_metric(episodes, "g_ok")

    eval_meta = eval_data.get("eval", {})
    seed_start = int(eval_meta.get("seed_start", episodes[0].get("seed", 0)))
    seed_end = int(eval_meta.get("seed_end", episodes[-1].get("seed", seed_start + n - 1)))

    mission = MissionConfig()
    atmosphere = AtmosphereConfig()
    env_cfg = EnvConfig()
    vehicle = default_vehicle_params()
    manifest = {
        "policy_label": policy_label,
        "model_path": str(model),
        "vecnormalize_path": str(vecnorm),
        "eval_path": str(eval_file),
        "promoted": True,
        "evidence_standard": {
            "min_episodes": 1000,
            "min_success_lcb": 0.95,
            "min_q_ok_lcb": 0.98,
            "min_g_ok_lcb": 0.98,
        },
        "eval": {
            "n_episodes": n,
            "seed_start": seed_start,
            "seed_end": seed_end,
            "deterministic": bool(eval_meta.get("deterministic", True)),
            "target_altitude_m": float(trained.get("target_altitude_m", mission.space_boundary_altitude_m)),
        },
        "counts": {
            "success": _metric_summary(success_count, n),
            "q_ok": _metric_summary(q_ok_count, n),
            "g_ok": _metric_summary(g_ok_count, n),
        },
        "summary_metrics": {
            "avg_max_altitude_m": trained.get("avg_max_altitude_m"),
            "avg_max_q": trained.get("avg_max_q"),
            "avg_q_margin_pa": trained.get("avg_q_margin_pa"),
            "avg_max_g_load": trained.get("avg_max_g_load"),
            "avg_burnout_vz_mps": trained.get("avg_burnout_vz_mps"),
        },
        "config": {
            "vehicle": _jsonable(vehicle.__dict__),
            "mission": _jsonable(mission.__dict__),
            "atmosphere": _jsonable(atmosphere.__dict__),
            "env": _jsonable(env_cfg.__dict__),
        },
        "git": {
            "commit": _git_output(["rev-parse", "HEAD"]),
            "status_short": _git_output(["status", "--short"]),
        },
        "claim": (
            "Promoted as the current best policy for the repo's randomized eval distribution. "
            "This does not prove PPO from scratch learns the task reliably."
        ),
    }

    thresholds = manifest["evidence_standard"]
    manifest["passes_evidence_standard"] = bool(
        n >= thresholds["min_episodes"]
        and manifest["counts"]["success"]["wilson_lcb"] >= thresholds["min_success_lcb"]
        and manifest["counts"]["q_ok"]["wilson_lcb"] >= thresholds["min_q_ok_lcb"]
        and manifest["counts"]["g_ok"]["wilson_lcb"] >= thresholds["min_g_ok_lcb"]
    )

    if out_path is not None:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a policy promotion manifest")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--vecnorm-path", required=True)
    parser.add_argument("--eval-path", required=True)
    parser.add_argument("--out-path", required=True)
    parser.add_argument("--policy-label", default="run37_selected_1100000")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = build_manifest(
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path,
        eval_path=args.eval_path,
        out_path=args.out_path,
        policy_label=args.policy_label,
    )
    print(json.dumps(result, indent=2))
