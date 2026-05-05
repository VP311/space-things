"""Deterministic feasibility probes for the 100 km ascent target.

This module deliberately avoids PPO/reward training. It runs hand-built and
searched controllers against the current simulator to answer whether the
vehicle/mission envelope has a q/g-compliant 100 km solution.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

import numpy as np

from config.defaults import AtmosphereConfig, CurriculumConfig, EnvConfig, MissionConfig, default_vehicle_params
from rl.env import RocketAscentEnv
from sim.atmosphere import AtmosphereProfile
from sim.constants import G0
from sim.vehicle import VehicleParams

TARGET_ALTITUDE_M = 100_000.0


class VacuumRocketAscentEnv(RocketAscentEnv):
    def sample_atmosphere_profile(self) -> AtmosphereProfile:
        cfg = self.atmosphere_cfg
        return AtmosphereProfile(
            density_scale=0.0,
            scale_height_m=8_500.0,
            gust_freq_x_hz=cfg.gust_freq_x_hz,
            gust_freq_y_hz=cfg.gust_freq_y_hz,
        )


def build_feasibility_env(
    *,
    vehicle: VehicleParams | None = None,
    mission: MissionConfig | None = None,
    vacuum: bool = False,
    record: bool = False,
) -> RocketAscentEnv:
    env_cfg = EnvConfig()
    curriculum_cfg = CurriculumConfig()
    env_cls = VacuumRocketAscentEnv if vacuum else RocketAscentEnv
    env = env_cls(
        params=vehicle or default_vehicle_params(),
        mission=mission or MissionConfig(),
        atmosphere_cfg=AtmosphereConfig(randomize=False),
        curriculum_milestones_m=curriculum_cfg.target_milestones_m,
        dt=env_cfg.dt,
        t_final=env_cfg.t_final,
        record=record,
        start_state_randomization=False,
        domain_randomization=False,
        action_repeat=env_cfg.action_repeat,
    )
    env.set_curriculum(TARGET_ALTITUDE_M, 0.0)
    return env


def q_feedback_throttle(
    *,
    q_dyn_pa: float,
    q_cap_pa: float,
    floor: float,
    high: float,
    gain: float,
) -> float:
    if q_cap_pa <= 0.0:
        raise ValueError("q_cap_pa must be positive")
    over = max(0.0, float(q_dyn_pa) / float(q_cap_pa) - 1.0)
    return float(np.clip(float(high) - float(gain) * over, float(floor), float(high)))


def decode_feedback_params(raw: np.ndarray) -> dict[str, float]:
    raw = np.asarray(raw, dtype=np.float32).reshape(-1)
    if raw.shape[0] != 8:
        raise ValueError(f"expected 8 feedback parameters, got {raw.shape[0]}")
    return {
        "q_cap_pa": float(np.interp(np.clip(raw[0], -1.0, 1.0), [-1.0, 1.0], [45_000.0, 72_000.0])),
        "floor": float(np.interp(np.clip(raw[1], -1.0, 1.0), [-1.0, 1.0], [0.35, 0.78])),
        "high": float(np.interp(np.clip(raw[2], -1.0, 1.0), [-1.0, 1.0], [0.82, 1.0])),
        "gain": float(np.interp(np.clip(raw[3], -1.0, 1.0), [-1.0, 1.0], [0.5, 6.0])),
        "bucket_until_alt_m": float(np.interp(np.clip(raw[4], -1.0, 1.0), [-1.0, 1.0], [25_000.0, 60_000.0])),
        "pitch_start_s": float(np.interp(np.clip(raw[5], -1.0, 1.0), [-1.0, 1.0], [20.0, 95.0])),
        "pitch_mid_s": float(np.interp(np.clip(raw[6], -1.0, 1.0), [-1.0, 1.0], [70.0, 165.0])),
        "gimbal_norm": float(np.interp(np.clip(raw[7], -1.0, 1.0), [-1.0, 1.0], [0.0, 0.9])),
    }


def feedback_action(env: RocketAscentEnv, info: dict[str, Any], params: dict[str, float]) -> np.ndarray:
    altitude = float(info.get("altitude_m", 0.0))
    q_dyn = float(info.get("q_dyn_pa", 0.0))
    if altitude < float(params["bucket_until_alt_m"]):
        throttle = q_feedback_throttle(
            q_dyn_pa=q_dyn,
            q_cap_pa=float(params["q_cap_pa"]),
            floor=float(params["floor"]),
            high=float(params["high"]),
            gain=float(params["gain"]),
        )
    else:
        throttle = float(params["high"])

    pitch_start = float(params["pitch_start_s"])
    pitch_mid = max(float(params["pitch_mid_s"]), pitch_start + 1.0)
    if env.t <= pitch_start:
        pitch_frac = 0.0
    elif env.t >= pitch_mid:
        pitch_frac = 1.0
    else:
        pitch_frac = (env.t - pitch_start) / (pitch_mid - pitch_start)
    gimbal = -float(params["gimbal_norm"]) * float(env.params.max_gimbal) * pitch_frac
    return np.array([throttle, gimbal], dtype=np.float32)


def fixed_action_controller(kind: str, params: dict[str, float]) -> Callable[[RocketAscentEnv, dict[str, Any]], np.ndarray]:
    def controller(env: RocketAscentEnv, info: dict[str, Any]) -> np.ndarray:
        if kind == "vertical_full":
            return np.array([1.0, 0.0], dtype=np.float32)
        if kind == "q_bucket":
            q_dyn = float(info.get("q_dyn_pa", 0.0))
            altitude = float(info.get("altitude_m", 0.0))
            throttle = float(params["bucket"]) if q_dyn > float(params["q_cap_pa"]) and altitude < float(params["until_alt_m"]) else float(params["high"])
            return np.array([throttle, 0.0], dtype=np.float32)
        if kind == "feedback":
            return feedback_action(env, info, params)
        raise ValueError(f"unknown controller kind: {kind}")

    return controller


def projected_burnout_apogee(metrics: dict[str, Any]) -> float:
    burnout_alt = float(metrics.get("altitude_at_burnout_m", float("nan")))
    burnout_vz = float(metrics.get("vz_at_burnout_mps", float("nan")))
    if not np.isfinite(burnout_alt) or not np.isfinite(burnout_vz):
        return float(metrics.get("max_altitude_m", 0.0))
    return float(burnout_alt + (burnout_vz**2) / (2.0 * G0))


def classify_failure(metrics: dict[str, Any]) -> str:
    if bool(metrics["success"]):
        return "success"
    if float(metrics["max_altitude_m"]) >= TARGET_ALTITUDE_M and not bool(metrics["q_ok"]):
        return "reaches_100km_but_q_violation"
    if float(metrics["max_altitude_m"]) >= TARGET_ALTITUDE_M and not bool(metrics["g_ok"]):
        return "reaches_100km_but_g_violation"
    if not bool(metrics["q_ok"]):
        return "q_limited_shortfall"
    if not bool(metrics["g_ok"]):
        return "g_limited_shortfall"
    return "energy_shortfall"


def rank_metrics(metrics: dict[str, Any]) -> tuple[float, float, float, float, float, float]:
    success = 1.0 if bool(metrics["success"]) else 0.0
    reaches = 1.0 if float(metrics["max_altitude_m"]) >= TARGET_ALTITUDE_M else 0.0
    q_ok = 1.0 if bool(metrics["q_ok"]) else 0.0
    g_ok = 1.0 if bool(metrics["g_ok"]) else 0.0
    return (
        success,
        reaches,
        q_ok + g_ok,
        float(metrics["max_altitude_m"]),
        float(metrics.get("projected_burnout_apogee_m", 0.0)),
        float(metrics.get("vz_at_burnout_mps", 0.0)) if np.isfinite(float(metrics.get("vz_at_burnout_mps", 0.0))) else 0.0,
    )


def rollout_controller(
    *,
    label: str,
    controller: Callable[[RocketAscentEnv, dict[str, Any]], np.ndarray],
    seed: int,
    vehicle: VehicleParams | None = None,
    mission: MissionConfig | None = None,
    vacuum: bool = False,
    record: bool = False,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    env = build_feasibility_env(vehicle=vehicle, mission=mission, vacuum=vacuum, record=record)
    obs, info = env.reset(seed=int(seed))
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = controller(env, info)
        observations.append(np.asarray(obs, dtype=np.float32).copy())
        actions.append(np.asarray(action, dtype=np.float32).copy())
        obs, _, terminated, truncated, info = env.step(action)

    max_q = float(info.get("max_q_dyn", 0.0))
    max_g = float(info.get("max_g_load", 0.0))
    metrics: dict[str, Any] = {
        "label": label,
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
        "fuel_used_fraction": float(info.get("fuel_used_fraction", 0.0)),
        "steps": len(actions),
        "vacuum": bool(vacuum),
        "vehicle": {
            "max_thrust": float(env.params.max_thrust),
            "dry_mass": float(env.params.dry_mass),
            "prop_mass": float(env.params.prop_mass),
            "area_ref": float(env.params.area_ref),
            "cd": float(env.params.cd),
        },
        "mission": {
            "target_altitude_m": float(env.curriculum_target_altitude_m),
            "max_q_pa": float(env.mission.max_q_pa),
            "max_g_load": float(env.mission.max_g_load),
        },
        "metadata": metadata or {},
    }
    metrics["projected_burnout_apogee_m"] = projected_burnout_apogee(metrics)
    metrics["failure_class"] = classify_failure(metrics)
    metrics["rank"] = rank_metrics(metrics)
    if record:
        metrics["observations"] = np.asarray(observations, dtype=np.float32)
        metrics["actions"] = np.asarray(actions, dtype=np.float32)
    env.close()
    return metrics


def baseline_probe_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {"label": "current_vertical_full", "kind": "vertical_full", "params": {}, "variant": "current"},
        {"label": "current_vacuum_vertical_full", "kind": "vertical_full", "params": {}, "variant": "vacuum"},
    ]
    for q_cap in (50_000.0, 60_000.0, 65_000.0, 70_000.0):
        for bucket in (0.45, 0.55, 0.65, 0.75):
            specs.append(
                {
                    "label": f"current_qbucket_q{int(q_cap)}_b{bucket:.2f}",
                    "kind": "q_bucket",
                    "params": {"q_cap_pa": q_cap, "bucket": bucket, "high": 1.0, "until_alt_m": 45_000.0},
                    "variant": "current",
                }
            )
    for q_cap in (55_000.0, 62_500.0, 70_000.0):
        for floor in (0.40, 0.55, 0.70):
            specs.append(
                {
                    "label": f"current_feedback_q{int(q_cap)}_f{floor:.2f}",
                    "kind": "feedback",
                    "params": {
                        "q_cap_pa": q_cap,
                        "floor": floor,
                        "high": 1.0,
                        "gain": 3.0,
                        "bucket_until_alt_m": 50_000.0,
                        "pitch_start_s": 45.0,
                        "pitch_mid_s": 130.0,
                        "gimbal_norm": 0.25,
                    },
                    "variant": "current",
                }
            )
    return specs


def sensitivity_specs() -> list[dict[str, Any]]:
    return [dict({"variant": variant}, **variant_overrides(variant)) for variant in sensitivity_variants()]


def sensitivity_variants() -> tuple[str, ...]:
    return (
        "cd0",
        "area_half",
        "prop_25t",
        "prop_30t",
        "thrust_1p8",
        "q_limit_80k",
        "q_limit_90k",
        "q_limit_110k",
    )


def variant_overrides(variant: str) -> dict[str, Any]:
    base = default_vehicle_params()
    if variant == "current":
        return {"vehicle": None, "mission": None, "vacuum": False}
    if variant == "vacuum":
        return {"vehicle": None, "mission": None, "vacuum": True}
    if variant == "cd0":
        return {"vehicle": replace(base, cd=0.0), "mission": None, "vacuum": False}
    if variant == "area_half":
        return {"vehicle": replace(base, area_ref=0.5 * base.area_ref), "mission": None, "vacuum": False}
    if variant == "prop_25t":
        return {"vehicle": replace(base, prop_mass=25_000.0), "mission": None, "vacuum": False}
    if variant == "prop_30t":
        return {"vehicle": replace(base, prop_mass=30_000.0), "mission": None, "vacuum": False}
    if variant == "thrust_1p8":
        return {"vehicle": replace(base, max_thrust=1.8e6), "mission": None, "vacuum": False}
    if variant == "q_limit_80k":
        return {"vehicle": None, "mission": replace(MissionConfig(), max_q_pa=80_000.0), "vacuum": False}
    if variant == "q_limit_90k":
        return {"vehicle": None, "mission": replace(MissionConfig(), max_q_pa=90_000.0), "vacuum": False}
    if variant == "q_limit_110k":
        return {"vehicle": None, "mission": replace(MissionConfig(), max_q_pa=110_000.0), "vacuum": False}
    raise ValueError(f"unknown feasibility variant: {variant}")


def cem_feedback_search(
    *,
    out_rows: list[dict[str, Any]],
    seed: int,
    population: int,
    generations: int,
    restarts: int,
) -> None:
    rng = np.random.default_rng(seed)
    means = [
        np.array([0.1, 0.0, 1.0, 0.0, 0.2, -0.2, 0.0, -0.5], dtype=np.float32),
        np.array([-0.2, -0.5, 1.0, 0.4, 0.4, -0.5, -0.1, -0.2], dtype=np.float32),
        np.array([0.8, 0.2, 1.0, -0.2, 0.0, 0.4, 0.6, 0.0], dtype=np.float32),
        np.array([0.4, -0.2, 1.0, 0.8, 0.8, 0.0, 0.5, 0.5], dtype=np.float32),
    ]
    for restart in range(int(restarts)):
        mean = means[restart % len(means)].copy()
        sigma = np.full(8, 0.65, dtype=np.float32)
        for generation in range(int(generations)):
            samples = np.clip(rng.normal(mean, sigma, size=(int(population), 8)), -1.5, 1.5).astype(np.float32)
            rows: list[dict[str, Any]] = []
            for idx, raw in enumerate(samples):
                params = decode_feedback_params(raw)
                row = rollout_controller(
                    label=f"cem_feedback_r{restart}_g{generation}_i{idx}",
                    controller=fixed_action_controller("feedback", params),
                    seed=seed + restart * 100_000 + generation * 1_000 + idx,
                    metadata={"controller": "feedback_cem", "restart": restart, "generation": generation, "raw": raw.tolist(), "params": params},
                )
                rows.append(row)
            rows.sort(key=rank_metrics, reverse=True)
            elite_count = max(2, int(population) // 4)
            elite_raw = np.asarray([rows[idx]["metadata"]["raw"] for idx in range(elite_count)], dtype=np.float32)
            mean = np.mean(elite_raw, axis=0)
            sigma = np.maximum(np.std(elite_raw, axis=0), 0.08).astype(np.float32)
            for rank, row in enumerate(rows[:5]):
                row["search_rank"] = rank
                out_rows.append(row)
            best = rows[0]
            print(
                "cem "
                f"restart={restart} generation={generation + 1}/{generations} "
                f"success={int(best['success'])} alt={best['max_altitude_m']:.1f} "
                f"vz={best['vz_at_burnout_mps']:.1f} q={best['max_q_dyn']:.1f} "
                f"g={best['max_g_load']:.2f} class={best['failure_class']}"
            )


def _json_ready(row: dict[str, Any]) -> dict[str, Any]:
    clean = {k: v for k, v in row.items() if k not in {"observations", "actions"}}
    if isinstance(clean.get("rank"), tuple):
        clean["rank"] = list(clean["rank"])
    return clean


def save_recorded_trajectories(rows: list[dict[str, Any]], path: Path, *, top_n: int = 5) -> None:
    recorded = [row for row in rows if "observations" in row and "actions" in row]
    if not recorded:
        return
    recorded.sort(key=rank_metrics, reverse=True)
    selected = recorded[:top_n]
    observations = np.concatenate([row["observations"] for row in selected], axis=0).astype(np.float32)
    actions = np.concatenate([row["actions"] for row in selected], axis=0).astype(np.float32)
    episode_ids = np.concatenate([np.full(len(row["actions"]), idx, dtype=np.int32) for idx, row in enumerate(selected)])
    metrics_json = np.asarray([json.dumps(_json_ready(row), sort_keys=True) for row in selected])
    np.savez_compressed(path, observations=observations, actions=actions, episode_ids=episode_ids, metrics_json=metrics_json)


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows_sorted = sorted(rows, key=rank_metrics, reverse=True)
    current_rows = [r for r in rows if r.get("metadata", {}).get("variant", "current") == "current" and not r.get("vacuum")]
    success_current = [r for r in current_rows if r["success"]]
    reaches_any = [r for r in rows if r["max_altitude_m"] >= TARGET_ALTITUDE_M]
    return {
        "n_rows": len(rows),
        "current_success_count": len(success_current),
        "any_reaches_100km_count": len(reaches_any),
        "best_overall": _json_ready(rows_sorted[0]) if rows_sorted else None,
        "best_current": _json_ready(max(current_rows, key=rank_metrics)) if current_rows else None,
        "best_reaches_100km": _json_ready(max(reaches_any, key=rank_metrics)) if reaches_any else None,
        "failure_class_counts": {key: sum(1 for row in rows if row["failure_class"] == key) for key in sorted({row["failure_class"] for row in rows})},
    }


def run_feasibility(
    *,
    out_dir: str | Path = "artifacts/living/artifacts_feasibility",
    seed: int = 37,
    cem_population: int = 24,
    cem_generations: int = 6,
    cem_restarts: int = 3,
    skip_cem: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for spec in baseline_probe_specs():
        row = rollout_controller(
            label=spec["label"],
            controller=fixed_action_controller(spec["kind"], spec["params"]),
            seed=seed,
            vacuum=spec["variant"] == "vacuum",
            metadata={"controller": spec["kind"], "variant": spec["variant"], "params": spec["params"]},
        )
        rows.append(row)
        print(f"probe {row['label']}: class={row['failure_class']} alt={row['max_altitude_m']:.1f} q={row['max_q_dyn']:.1f}")

    for sens in sensitivity_specs():
        row = rollout_controller(
            label=f"sensitivity_{sens['variant']}_vertical_full",
            controller=fixed_action_controller("vertical_full", {}),
            seed=seed,
            vehicle=sens.get("vehicle"),
            mission=sens.get("mission"),
            vacuum=bool(sens.get("vacuum", False)),
            metadata={"controller": "vertical_full", "variant": sens["variant"]},
        )
        rows.append(row)
        print(f"sensitivity {row['label']}: class={row['failure_class']} alt={row['max_altitude_m']:.1f} q={row['max_q_dyn']:.1f}")

    if not skip_cem:
        cem_feedback_search(
            out_rows=rows,
            seed=seed + 10_000,
            population=cem_population,
            generations=cem_generations,
            restarts=cem_restarts,
        )

    best_rows = sorted(rows, key=rank_metrics, reverse=True)[:5]
    recorded: list[dict[str, Any]] = []
    for idx, best in enumerate(best_rows):
        meta = dict(best.get("metadata", {}))
        if meta.get("controller") == "feedback_cem":
            controller = fixed_action_controller("feedback", meta["params"])
        else:
            controller = fixed_action_controller(str(meta.get("controller", "vertical_full")), meta.get("params", {}))
        variant = str(meta.get("variant", "current"))
        overrides = variant_overrides(variant)
        recorded.append(
            rollout_controller(
                label=f"recorded_{idx}_{best['label']}",
                controller=controller,
                seed=seed + 500_000 + idx,
                vehicle=overrides.get("vehicle"),
                mission=overrides.get("mission"),
                vacuum=bool(overrides.get("vacuum", False)),
                record=True,
                metadata=meta,
            )
        )
    rows.extend(recorded)

    results_path = out / "results.jsonl"
    with results_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(_json_ready(row)) + "\n")
    save_recorded_trajectories(recorded, out / "trajectories.npz")
    summary = summarise(rows)
    summary.update(
        {
            "seed": int(seed),
            "cem_population": int(cem_population),
            "cem_generations": int(cem_generations),
            "cem_restarts": int(cem_restarts),
            "skip_cem": bool(skip_cem),
            "results_path": str(results_path),
            "trajectories_path": str(out / "trajectories.npz"),
        }
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run deterministic rocket feasibility probes")
    parser.add_argument("--out-dir", type=str, default="artifacts/living/artifacts_feasibility")
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--cem-population", type=int, default=24)
    parser.add_argument("--cem-generations", type=int, default=6)
    parser.add_argument("--cem-restarts", type=int, default=3)
    parser.add_argument("--skip-cem", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_feasibility(
        out_dir=args.out_dir,
        seed=args.seed,
        cem_population=args.cem_population,
        cem_generations=args.cem_generations,
        cem_restarts=args.cem_restarts,
        skip_cem=args.skip_cem,
    )
