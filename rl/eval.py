"""Evaluation harness for baseline and trained rocket policies."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from config.defaults import AtmosphereConfig, EnvConfig, MissionConfig, default_vehicle_params
from rl.env import RocketAscentEnv, baseline_action


def _resolve_model_path(preferred_path: str) -> str:
    preferred = Path(preferred_path)
    best = Path("artifacts/best_model/best_model.zip")
    if preferred.exists():
        return str(preferred)
    if best.exists():
        return str(best)
    return str(preferred)


def evaluate_policy(
    mode: str,
    n_episodes: int = 100,
    model: PPO | None = None,
    seed_start: int = 1000,
) -> dict:
    if mode not in {"baseline", "trained"}:
        raise ValueError("mode must be 'baseline' or 'trained'")
    if mode == "trained" and model is None:
        raise ValueError("trained mode requires a loaded model")

    params = default_vehicle_params()
    cfg = EnvConfig()
    mission = MissionConfig()
    atmosphere_cfg = AtmosphereConfig()
    env = RocketAscentEnv(
        params=params,
        mission=mission,
        atmosphere_cfg=atmosphere_cfg,
        dt=cfg.dt,
        t_final=cfg.t_final,
        record=False,
    )

    rows: list[dict] = []
    for i in range(n_episodes):
        seed = seed_start + i
        obs, info = env.reset(seed=seed)
        terminated = False
        truncated = False
        steps = 0
        while not (terminated or truncated):
            if mode == "baseline":
                action = baseline_action(
                    env.t,
                    env.params.max_gimbal,
                    altitude_m=float(info["altitude_m"]),
                    q_dyn_pa=float(info["q_dyn_pa"]),
                    q_limit_pa=env.mission.max_q_pa,
                )
            else:
                action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            _ = reward
            steps += 1

        rows.append(
            {
                "seed": seed,
                "steps": steps,
                "success": bool(info["success"]),
                "crash": bool(info["crash"]),
                "hard_violation": bool(info["hard_violation"]),
                "max_altitude_m": float(info["max_altitude_m"]),
                "max_q_dyn": float(info["max_q_dyn"]),
                "max_g_load": float(info["max_g_load"]),
                "terminal_altitude_m": float(info["altitude_m"]),
                "terminal_vx_mps": float(info["vx_mps"]),
                "terminal_vz_mps": float(info["vz_mps"]),
                "terminal_flight_path_angle_deg": float(info["flight_path_angle_deg"]),
                "fuel_used_fraction": float(info["fuel_used_fraction"]),
            }
        )

    success = np.array([r["success"] for r in rows], dtype=bool)
    crash = np.array([r["crash"] for r in rows], dtype=bool)
    hard_violation = np.array([r["hard_violation"] for r in rows], dtype=bool)
    max_q = np.array([r["max_q_dyn"] for r in rows], dtype=float)
    max_g = np.array([r["max_g_load"] for r in rows], dtype=float)
    terminal_alt = np.array([r["terminal_altitude_m"] for r in rows], dtype=float)
    terminal_vx = np.array([r["terminal_vx_mps"] for r in rows], dtype=float)
    terminal_vz = np.array([r["terminal_vz_mps"] for r in rows], dtype=float)
    terminal_gamma = np.array([r["terminal_flight_path_angle_deg"] for r in rows], dtype=float)
    fuel_used = np.array([r["fuel_used_fraction"] for r in rows], dtype=float)
    q_violation = max_q > env.mission.max_q_pa
    g_violation = max_g > env.mission.max_g_load

    return {
        "mode": mode,
        "n_episodes": n_episodes,
        "success_rate": float(np.mean(success)),
        "crash_rate": float(np.mean(crash)),
        "hard_violation_rate": float(np.mean(hard_violation)),
        "q_violation_rate": float(np.mean(q_violation)),
        "g_violation_rate": float(np.mean(g_violation)),
        "avg_max_q": float(np.mean(max_q)),
        "avg_max_g_load": float(np.mean(max_g)),
        "avg_terminal_altitude_m": float(np.mean(terminal_alt)),
        "avg_terminal_vx_mps": float(np.mean(terminal_vx)),
        "avg_terminal_vz_mps": float(np.mean(terminal_vz)),
        "avg_terminal_flight_path_angle_deg": float(np.mean(terminal_gamma)),
        "avg_fuel_used_fraction": float(np.mean(fuel_used)),
        "episodes": rows,
    }


def run_eval(
    model_path: str = "artifacts/best_model/best_model.zip",
    out_path: str = "artifacts/eval_results.json",
    n_episodes: int = 100,
) -> dict:
    cfg = EnvConfig()
    mission = MissionConfig()
    atmosphere_cfg = AtmosphereConfig()
    results: dict[str, object] = {
        "config": {
            "dt": cfg.dt,
            "t_final": cfg.t_final,
            "train_total_timesteps": cfg.train_total_timesteps,
            "mission": mission.__dict__,
            "atmosphere": atmosphere_cfg.__dict__,
        }
    }

    baseline = evaluate_policy(mode="baseline", n_episodes=n_episodes, model=None, seed_start=1000)
    results["baseline"] = baseline

    resolved_model_path = _resolve_model_path(model_path)
    model_file = Path(resolved_model_path)
    if model_file.exists():
        try:
            model = PPO.load(str(model_file))
            trained = evaluate_policy(mode="trained", n_episodes=n_episodes, model=model, seed_start=2000)
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
        section = results.get(key, {})
        if isinstance(section, dict) and section.get("available") is False:
            print(f"{key}: unavailable ({section.get('reason')})")
            continue
        if isinstance(section, dict):
            print(
                f"{key}: "
                f"success_rate={section['success_rate']:.3f} "
                f"crash_rate={section['crash_rate']:.3f} "
                f"avg_max_q={section['avg_max_q']:.1f} "
                f"avg_max_g={section['avg_max_g_load']:.2f} "
                f"avg_terminal_alt={section['avg_terminal_altitude_m']:.1f}"
            )
    return results


if __name__ == "__main__":
    run_eval()
