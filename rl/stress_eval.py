"""Robustness/OOD stress evaluation for promoted ascent policies."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO

from config.defaults import AtmosphereConfig, CurriculumConfig, EnvConfig, MissionConfig, default_vehicle_params
from rl.feasibility import fixed_action_controller
from rl.policy_eval import (
    build_eval_env,
    evaluate_env,
    load_vecnormalize_for_eval,
    resolve_model_path,
    resolve_vecnorm_path,
    summarise_rows,
)
from rl.select_checkpoint import wilson_summary
from sim.vehicle import VehicleParams


DEFAULT_RUN38_MODEL = "artifacts/living/artifacts_promoted/run38_250k/model.zip"
DEFAULT_RUN38_VECNORM = "artifacts/living/artifacts_promoted/run38_250k/vecnormalize.pkl"
DEFAULT_QBUCKET_PARAMS = {"q_cap_pa": 55_000.0, "bucket": 0.5, "high": 1.0, "until_alt_m": 60_000.0}


@dataclass(frozen=True)
class StressCase:
    name: str
    description: str
    vehicle_scales: dict[str, float]
    atmosphere_scales: dict[str, float]
    observation_noise_std: float = 0.0
    action_lag_steps: int = 0
    domain_randomization: bool = False
    atmosphere_randomization: bool = True


def _scale_vehicle(base: VehicleParams, scales: dict[str, float]) -> VehicleParams:
    return replace(
        base,
        max_thrust=base.max_thrust * float(scales.get("max_thrust", 1.0)),
        isp=base.isp * float(scales.get("isp", 1.0)),
        dry_mass=base.dry_mass * float(scales.get("dry_mass", 1.0)),
        prop_mass=base.prop_mass * float(scales.get("prop_mass", 1.0)),
        area_ref=base.area_ref * float(scales.get("area_ref", 1.0)),
        cd=base.cd * float(scales.get("cd", 1.0)),
    )


def _scale_range(low: float, high: float, scale: float) -> tuple[float, float]:
    return float(low * scale), float(high * scale)


def _build_atmosphere(case: StressCase) -> AtmosphereConfig:
    base = AtmosphereConfig(randomize=case.atmosphere_randomization)
    density_scale = float(case.atmosphere_scales.get("density_scale", 1.0))
    scale_height_scale = float(case.atmosphere_scales.get("scale_height", 1.0))
    wind_scale = float(case.atmosphere_scales.get("wind", 1.0))
    gust_scale = float(case.atmosphere_scales.get("gust", wind_scale))
    density_min, density_max = _scale_range(base.density_scale_min, base.density_scale_max, density_scale)
    scale_height_min, scale_height_max = _scale_range(base.scale_height_min_m, base.scale_height_max_m, scale_height_scale)
    wind_bias_x_min, wind_bias_x_max = _scale_range(base.wind_bias_x_min_mps, base.wind_bias_x_max_mps, wind_scale)
    wind_bias_y_min, wind_bias_y_max = _scale_range(base.wind_bias_y_min_mps, base.wind_bias_y_max_mps, wind_scale)
    wind_shear_x_min, wind_shear_x_max = _scale_range(base.wind_shear_x_min_mps, base.wind_shear_x_max_mps, wind_scale)
    wind_shear_y_min, wind_shear_y_max = _scale_range(base.wind_shear_y_min_mps, base.wind_shear_y_max_mps, wind_scale)
    gust_x_min, gust_x_max = _scale_range(base.gust_amp_x_min_mps, base.gust_amp_x_max_mps, gust_scale)
    gust_y_min, gust_y_max = _scale_range(base.gust_amp_y_min_mps, base.gust_amp_y_max_mps, gust_scale)
    return replace(
        base,
        density_scale_min=density_min,
        density_scale_max=density_max,
        scale_height_min_m=scale_height_min,
        scale_height_max_m=scale_height_max,
        wind_bias_x_min_mps=wind_bias_x_min,
        wind_bias_x_max_mps=wind_bias_x_max,
        wind_bias_y_min_mps=wind_bias_y_min,
        wind_bias_y_max_mps=wind_bias_y_max,
        wind_shear_x_min_mps=wind_shear_x_min,
        wind_shear_x_max_mps=wind_shear_x_max,
        wind_shear_y_min_mps=wind_shear_y_min,
        wind_shear_y_max_mps=wind_shear_y_max,
        gust_amp_x_min_mps=gust_x_min,
        gust_amp_x_max_mps=gust_x_max,
        gust_amp_y_min_mps=gust_y_min,
        gust_amp_y_max_mps=gust_y_max,
    )


def make_stress_cases() -> list[StressCase]:
    cases = [
        StressCase(
            name="nominal_distribution",
            description="Current randomized eval distribution.",
            vehicle_scales={},
            atmosphere_scales={},
            domain_randomization=True,
            atmosphere_randomization=True,
        )
    ]
    for value in (0.8, 0.9, 1.1, 1.2):
        cases.append(StressCase(f"cd_{value:.1f}x", f"CD scaled to {value:.1f}x.", {"cd": value}, {}))
    for value in (0.85, 0.925, 1.075, 1.15):
        cases.append(StressCase(f"isp_{value:.3g}x", f"Isp scaled to {value:.3g}x.", {"isp": value}, {}))
    for value in (0.9, 0.95, 1.05, 1.1):
        cases.append(StressCase(f"dry_mass_{value:.2g}x", f"Dry mass scaled to {value:.2g}x.", {"dry_mass": value}, {}))
    for value in (0.9, 0.95, 1.05, 1.1):
        cases.append(StressCase(f"thrust_{value:.2g}x", f"Max thrust scaled to {value:.2g}x.", {"max_thrust": value}, {}))
    for value in (1.25, 1.5, 2.0):
        cases.append(StressCase(f"wind_{value:.2g}x", f"Wind and gust ranges scaled to {value:.2g}x.", {}, {"wind": value, "gust": value}))
    for value in (0.8, 1.2):
        cases.append(StressCase(f"density_{value:.1f}x", f"Density scale range multiplied by {value:.1f}x.", {}, {"density_scale": value}))
    for value in (0.9, 1.1):
        cases.append(StressCase(f"scale_height_{value:.1f}x", f"Scale-height range multiplied by {value:.1f}x.", {}, {"scale_height": value}))
    for value in (1, 2, 3):
        cases.append(StressCase(f"actuator_lag_{value}_steps", f"Actuator command lag of {value} env steps.", {}, {}, action_lag_steps=value))
    for value in (0.01, 0.03, 0.05):
        cases.append(StressCase(f"obs_noise_{value:.2f}", f"Gaussian observation noise std {value:.2f}.", {}, {}, observation_noise_std=value))
    cases.extend(
        [
            StressCase(
                "joint_mild",
                "Mild combined aero/propulsion/wind/lag stress.",
                {"cd": 1.1, "max_thrust": 0.95, "isp": 0.925, "dry_mass": 1.05},
                {"wind": 1.25, "gust": 1.25, "density_scale": 1.1},
                observation_noise_std=0.01,
                action_lag_steps=1,
            ),
            StressCase(
                "joint_harsh_aero",
                "High drag, denser atmosphere, stronger wind.",
                {"cd": 1.2},
                {"wind": 1.5, "gust": 1.5, "density_scale": 1.2, "scale_height": 1.1},
                observation_noise_std=0.01,
            ),
            StressCase(
                "joint_harsh_propulsion",
                "Lower thrust and Isp with heavier dry mass.",
                {"max_thrust": 0.9, "isp": 0.85, "dry_mass": 1.1},
                {},
                action_lag_steps=1,
            ),
            StressCase(
                "joint_harsh_wind_noise_lag",
                "Strong wind/gusts with sensor noise and actuator lag.",
                {},
                {"wind": 2.0, "gust": 2.0},
                observation_noise_std=0.03,
                action_lag_steps=3,
            ),
            StressCase(
                "joint_worst_plausible",
                "Combined harsh vehicle, atmosphere, noise, and actuator stress.",
                {"cd": 1.2, "max_thrust": 0.9, "isp": 0.85, "dry_mass": 1.1},
                {"wind": 2.0, "gust": 2.0, "density_scale": 1.2, "scale_height": 1.1},
                observation_noise_std=0.05,
                action_lag_steps=3,
            ),
        ]
    )
    return cases


def _case_by_name() -> dict[str, StressCase]:
    return {case.name: case for case in make_stress_cases()}


def vehicle_to_dict(params: VehicleParams) -> dict[str, float]:
    return {
        "max_thrust": float(params.max_thrust),
        "isp": float(params.isp),
        "dry_mass": float(params.dry_mass),
        "prop_mass": float(params.prop_mass),
        "area_ref": float(params.area_ref),
        "cd": float(params.cd),
        "gimbal_arm": float(params.gimbal_arm),
        "max_gimbal": float(params.max_gimbal),
        "aero_damp": float(params.aero_damp),
    }


def select_cases(names: str | None, *, limit: int | None = None) -> list[StressCase]:
    cases_by_name = _case_by_name()
    if names:
        selected = []
        for name in [item.strip() for item in names.split(",") if item.strip()]:
            if name not in cases_by_name:
                raise ValueError(f"unknown stress case {name!r}; choices: {', '.join(sorted(cases_by_name))}")
            selected.append(cases_by_name[name])
    else:
        selected = make_stress_cases()
    if limit is not None:
        selected = selected[: int(limit)]
    return selected


def build_stress_env(case: StressCase, *, target_altitude_m: float = 100_000.0) -> Any:
    vehicle = _scale_vehicle(default_vehicle_params(), case.vehicle_scales)
    atmosphere = _build_atmosphere(case)
    return build_eval_env(
        target_altitude_m,
        params=vehicle,
        atmosphere_cfg=atmosphere,
        domain_randomization=case.domain_randomization,
        observation_noise_std=case.observation_noise_std,
        action_lag_steps=case.action_lag_steps,
    )


def _termination_counts(summary: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in summary.get("episodes", []):
        reason = str(row.get("termination_reason", "unknown"))
        counts[reason] = counts.get(reason, 0) + 1
    return counts


def _attach_confidence(summary: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(summary)
    enriched["wilson"] = wilson_summary(summary)
    enriched["termination_counts"] = _termination_counts(summary)
    return enriched


def run_controller_episode(
    env: Any,
    *,
    seed: int,
    controller: Callable[[Any, dict[str, Any]], np.ndarray],
) -> dict[str, Any]:
    obs, info = env.reset(seed=seed)
    _ = obs
    terminated = False
    truncated = False
    steps = 0
    while not (terminated or truncated):
        action = controller(env, info)
        obs, _, terminated, truncated, info = env.step(action)
        _ = obs
        steps += 1
    return {
        "seed": seed,
        "steps": steps,
        "success": bool(info["success"]),
        "crash": bool(info["crash"]),
        "hard_violation": bool(info["hard_violation"]),
        "termination_reason": str(info["termination_reason"]),
        "target_altitude_m": float(info["mission"]["target_altitude_m"]),
        "max_altitude_m": float(info["max_altitude_m"]),
        "max_q_dyn": float(info["max_q_dyn"]),
        "q_margin_pa": float(info.get("q_margin_pa", 0.0)),
        "q_over_limit_fraction": float(info.get("q_over_limit_fraction", 0.0)),
        "max_g_load": float(info["max_g_load"]),
        "altitude_at_burnout_m": float(info.get("altitude_at_burnout_m", float("nan"))),
        "velocity_at_burnout_mps": float(info.get("velocity_at_burnout_mps", float("nan"))),
        "vz_at_burnout_mps": float(info.get("vz_at_burnout_mps", float("nan"))),
        "terminal_altitude_m": float(info["altitude_m"]),
        "terminal_vx_mps": float(info["vx_mps"]),
        "terminal_vz_mps": float(info["vz_mps"]),
        "terminal_flight_path_angle_deg": float(info["flight_path_angle_deg"]),
        "fuel_used_fraction": float(info["fuel_used_fraction"]),
    }


def evaluate_scripted_qbucket(case: StressCase, *, n_episodes: int, seed_start: int, target_altitude_m: float) -> dict[str, Any]:
    controller = fixed_action_controller("q_bucket", DEFAULT_QBUCKET_PARAMS)
    env = build_stress_env(case, target_altitude_m=target_altitude_m)
    rows = [
        run_controller_episode(env, seed=seed_start + idx, controller=controller)
        for idx in range(int(n_episodes))
    ]
    summary = summarise_rows(
        rows,
        mode="scripted_qbucket",
        mission=env.mission,
        target_altitude_m=target_altitude_m,
    )
    env.close()
    return _attach_confidence(summary)


def evaluate_case(
    case: StressCase,
    *,
    model: PPO,
    vecnorm: Any,
    n_episodes: int,
    seed_start: int,
    target_altitude_m: float,
    deterministic: bool,
    include_baseline: bool,
    include_scripted: bool,
) -> dict[str, Any]:
    env = build_stress_env(case, target_altitude_m=target_altitude_m)
    trained = evaluate_env(
        env,
        mode="trained",
        n_episodes=n_episodes,
        seed_start=seed_start,
        model=model,
        vecnorm=vecnorm,
        deterministic=deterministic,
    )
    result: dict[str, Any] = {
        "case": asdict(case),
        "vehicle": vehicle_to_dict(_scale_vehicle(default_vehicle_params(), case.vehicle_scales)),
        "atmosphere": asdict(_build_atmosphere(case)),
        "trained": _attach_confidence(trained),
    }
    if include_baseline:
        result["baseline"] = _attach_confidence(
            evaluate_env(
                env,
                mode="baseline",
                n_episodes=n_episodes,
                seed_start=seed_start,
                deterministic=deterministic,
            )
        )
    env.close()
    if include_scripted:
        result["scripted_qbucket"] = evaluate_scripted_qbucket(
            case,
            n_episodes=n_episodes,
            seed_start=seed_start,
            target_altitude_m=target_altitude_m,
        )
    return result


def run_stress_eval(
    *,
    model_path: str = DEFAULT_RUN38_MODEL,
    vecnorm_path: str = DEFAULT_RUN38_VECNORM,
    out_dir: str | Path = "artifacts/living/artifacts_robustness",
    n_episodes: int = 200,
    seed_start: int = 20_000,
    target_altitude_m: float = 100_000.0,
    case_names: str | None = None,
    limit_cases: int | None = None,
    deterministic: bool = True,
    include_baseline: bool = False,
    include_scripted: bool = True,
) -> dict[str, Any]:
    resolved_model = resolve_model_path(model_path)
    resolved_vecnorm = resolve_vecnorm_path(vecnorm_path, allow_missing=False)
    model = PPO.load(resolved_model)
    vecnorm = load_vecnormalize_for_eval(resolved_vecnorm, target_altitude_m=target_altitude_m)
    selected = select_cases(case_names, limit=limit_cases)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    case_results = []
    for case in selected:
        print(f"stress case {case.name}: episodes={n_episodes} seed_start={seed_start}")
        case_results.append(
            evaluate_case(
                case,
                model=model,
                vecnorm=vecnorm,
                n_episodes=n_episodes,
                seed_start=seed_start,
                target_altitude_m=target_altitude_m,
                deterministic=deterministic,
                include_baseline=include_baseline,
                include_scripted=include_scripted,
            )
        )

    vecnorm.close()
    result = {
        "eval": {
            "n_episodes": int(n_episodes),
            "seed_start": int(seed_start),
            "seed_end": int(seed_start + n_episodes - 1),
            "target_altitude_m": float(target_altitude_m),
            "deterministic": bool(deterministic),
            "include_baseline": bool(include_baseline),
            "include_scripted": bool(include_scripted),
        },
        "policy": {
            "model_path": resolved_model,
            "vecnormalize_path": resolved_vecnorm,
        },
        "cases": case_results,
    }
    out_path = out / "stress_eval.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"saved stress eval: {out_path}")
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained rocket policy on OOD robustness stress cases")
    parser.add_argument("--model-path", type=str, default=DEFAULT_RUN38_MODEL)
    parser.add_argument("--vecnorm-path", type=str, default=DEFAULT_RUN38_VECNORM)
    parser.add_argument("--out-dir", type=str, default="artifacts/living/artifacts_robustness")
    parser.add_argument("--n-episodes", type=int, default=200)
    parser.add_argument("--seed-start", type=int, default=20_000)
    parser.add_argument("--target-altitude-m", type=float, default=100_000.0)
    parser.add_argument("--cases", type=str, default=None, help="Comma-separated stress case names")
    parser.add_argument("--limit-cases", type=int, default=None, help="Evaluate only the first N selected cases")
    parser.add_argument("--stochastic", dest="deterministic", action="store_false", default=True)
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--skip-scripted", dest="include_scripted", action="store_false", default=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_stress_eval(
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path,
        out_dir=args.out_dir,
        n_episodes=args.n_episodes,
        seed_start=args.seed_start,
        target_altitude_m=args.target_altitude_m,
        case_names=args.cases,
        limit_cases=args.limit_cases,
        deterministic=args.deterministic,
        include_baseline=args.include_baseline,
        include_scripted=args.include_scripted,
    )
