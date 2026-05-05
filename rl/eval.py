"""Evaluation harness for baseline and trained rocket policies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from config.defaults import AtmosphereConfig, EnvConfig, MissionConfig
from rl.policy_eval import (
    DEFAULT_BEST_FINAL_MODEL_PATH,
    evaluate_policy,
    load_vecnormalize_for_eval,
    resolve_model_path,
    resolve_vecnorm_path,
)


def run_eval(
    model_path: str = DEFAULT_BEST_FINAL_MODEL_PATH,
    vecnorm_path: str | None = None,
    out_path: str = "artifacts/living/eval_results.json",
    n_episodes: int = 100,
    seed_start: int = 2000,
    target_altitude_m: float = 100_000.0,
    allow_missing_vecnorm: bool = False,
    deterministic: bool = True,
    skip_baseline: bool = False,
) -> dict:
    cfg = EnvConfig()
    mission = MissionConfig()
    atmosphere_cfg = AtmosphereConfig()
    results: dict[str, object] = {
        "eval": {
            "n_episodes": int(n_episodes),
            "seed_start": int(seed_start),
            "seed_end": int(seed_start + n_episodes - 1),
            "deterministic": bool(deterministic),
            "skip_baseline": bool(skip_baseline),
        },
        "config": {
            "dt": cfg.dt,
            "t_final": cfg.t_final,
            "train_total_timesteps": cfg.train_total_timesteps,
            "target_altitude_m": target_altitude_m,
            "mission": mission.__dict__,
            "atmosphere": atmosphere_cfg.__dict__,
        }
    }

    if not skip_baseline:
        baseline = evaluate_policy(
            mode="baseline",
            n_episodes=n_episodes,
            seed_start=seed_start,
            target_altitude_m=target_altitude_m,
            deterministic=deterministic,
        )
        results["baseline"] = baseline

    resolved_model_path = resolve_model_path(model_path)
    model_file = Path(resolved_model_path)
    if model_file.exists():
        resolved_vecnorm = resolve_vecnorm_path(vecnorm_path, allow_missing=allow_missing_vecnorm)
        try:
            model = PPO.load(str(model_file))
            vecnorm = (
                load_vecnormalize_for_eval(resolved_vecnorm, target_altitude_m=target_altitude_m)
                if resolved_vecnorm is not None
                else None
            )
            trained = evaluate_policy(
                mode="trained",
                n_episodes=n_episodes,
                model=model,
                vecnorm=vecnorm,
                seed_start=seed_start,
                target_altitude_m=target_altitude_m,
                deterministic=deterministic,
            )
            results["trained"] = trained
        except Exception as exc:
            results["trained"] = {
                "available": False,
                "reason": f"incompatible or invalid model at {model_file}: {type(exc).__name__}: {exc}",
            }
    else:
        results["trained"] = {"available": False, "reason": f"missing model at {model_file}"}

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))

    print(f"saved eval: {out}")
    for key in ("baseline", "trained"):
        if key not in results:
            continue
        section = results.get(key, {})
        if isinstance(section, dict) and section.get("available") is False:
            print(f"{key}: unavailable ({section.get('reason')})")
            continue
        if isinstance(section, dict):
            print(
                f"{key}: "
                f"success_rate={section['success_rate']:.3f} "
                f"q_ok_rate={section['q_ok_rate']:.3f} "
                f"avg_max_alt={section['avg_max_altitude_m']:.1f} "
                f"avg_max_q={section['avg_max_q']:.1f} "
                f"avg_burnout_vz={section['avg_burnout_vz_mps']:.1f}"
            )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate baseline and trained rocket policies")
    parser.add_argument("--model-path", type=str, default=DEFAULT_BEST_FINAL_MODEL_PATH)
    parser.add_argument("--vecnorm-path", type=str, default=None)
    parser.add_argument("--out-path", type=str, default="artifacts/living/eval_results.json")
    parser.add_argument("--n-episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--target-altitude-m", type=float, default=100_000.0)
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_true",
        default=True,
        help="Use deterministic policy actions",
    )
    parser.add_argument(
        "--stochastic",
        dest="deterministic",
        action="store_false",
        help="Use stochastic policy actions",
    )
    parser.add_argument("--skip-baseline", action="store_true", help="Only evaluate the trained policy")
    parser.add_argument(
        "--allow-missing-vecnorm",
        action="store_true",
        help="Allow trained-policy eval without VecNormalize stats",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_eval(
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path,
        out_path=args.out_path,
        n_episodes=args.n_episodes,
        seed_start=args.seed_start,
        target_altitude_m=args.target_altitude_m,
        allow_missing_vecnorm=args.allow_missing_vecnorm,
        deterministic=args.deterministic,
        skip_baseline=args.skip_baseline,
    )
