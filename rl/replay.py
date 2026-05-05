"""Replay a trained PPO policy and save telemetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from config.defaults import default_vehicle_params
from rl.policy_eval import (
    DEFAULT_BEST_FINAL_MODEL_PATH,
    build_eval_env,
    load_vecnormalize_for_eval,
    normalise_obs,
    resolve_model_path,
    resolve_vecnorm_path,
)


def replay(
    model_path: str = DEFAULT_BEST_FINAL_MODEL_PATH,
    vecnorm_path: str | None = None,
    telemetry_path: str = "artifacts/living/telemetry.npz",
    metrics_path: str = "artifacts/living/metrics.json",
    target_altitude_m: float = 100_000.0,
    allow_missing_vecnorm: bool = False,
) -> tuple[Path, Path]:
    params = default_vehicle_params()
    resolved_model_path = resolve_model_path(model_path)
    resolved_vecnorm = resolve_vecnorm_path(vecnorm_path, allow_missing=allow_missing_vecnorm)

    # Raw env for stepping and telemetry — DummyVecEnv auto-resets on done which
    # clears telemetry before we can read it, so we use the bare env directly.
    env = build_eval_env(target_altitude_m=target_altitude_m, record=True)
    vecnorm = (
        load_vecnormalize_for_eval(resolved_vecnorm, target_altitude_m=target_altitude_m)
        if resolved_vecnorm is not None
        else None
    )
    model = PPO.load(resolved_model_path)

    raw_obs, info = env.reset(seed=0, options={"record": True})
    obs = normalise_obs(raw_obs, vecnorm)

    terminated = False
    truncated = False
    reward_progress: list[float] = []
    reward_q_pen: list[float] = []
    reward_g_pen: list[float] = []
    reward_fuel_pen: list[float] = []
    reward_tilt_pen: list[float] = []
    reward_smooth_pen: list[float] = []
    reward_high_stage_energy_bonus: list[float] = []
    reward_overshoot_pen: list[float] = []

    while not (terminated or truncated):
        try:
            action, _ = model.predict(obs, deterministic=True)
        except Exception as exc:
            raise RuntimeError(
                "Loaded policy is incompatible with current environment. "
                "Retrain with `python3 -m rl.train` before replay."
            ) from exc
        if isinstance(action, np.ndarray) and action.ndim > 1:
            action = action[0]
        raw_obs, _reward, terminated, truncated, info = env.step(action)
        obs = normalise_obs(raw_obs, vecnorm)
        comps = info.get("reward_components_step", {})
        reward_progress.append(float(comps.get("progress", 0.0)))
        reward_q_pen.append(float(comps.get("q_pen", 0.0)))
        reward_g_pen.append(float(comps.get("g_pen", 0.0)))
        reward_fuel_pen.append(float(comps.get("fuel_pen", 0.0)))
        reward_tilt_pen.append(float(comps.get("tilt_pen", 0.0)))
        reward_smooth_pen.append(float(comps.get("smooth_pen", 0.0)))
        reward_high_stage_energy_bonus.append(float(comps.get("high_stage_energy_bonus", 0.0)))
        reward_overshoot_pen.append(float(comps.get("overshoot_pen", 0.0)))

    if env.telemetry is None or not env.telemetry:
        raise RuntimeError("No telemetry collected during replay.")

    samples = env.telemetry
    t = np.array([s.t for s in samples], dtype=np.float32)
    x = np.array([s.pos[0] for s in samples], dtype=np.float32)
    y = np.array([s.pos[1] for s in samples], dtype=np.float32)
    z = np.array([s.altitude for s in samples], dtype=np.float32)
    vx = np.array([s.vel[0] for s in samples], dtype=np.float32)
    vz = np.array([s.vel[2] for s in samples], dtype=np.float32)
    q_dyn = np.array([s.q_dyn for s in samples], dtype=np.float32)
    speed = np.array([s.speed for s in samples], dtype=np.float32)
    mass = np.array([s.mass for s in samples], dtype=np.float32)
    throttle = np.array([s.control.throttle for s in samples], dtype=np.float32)
    gimbal_pitch = np.array([s.control.gimbal_pitch for s in samples], dtype=np.float32)

    telemetry_out = Path(telemetry_path)
    telemetry_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        telemetry_out,
        t=t,
        x=x,
        y=y,
        z=z,
        vx=vx,
        vz=vz,
        q_dyn=q_dyn,
        speed=speed,
        mass=mass,
        throttle=throttle,
        gimbal_pitch=gimbal_pitch,
        reward_progress=np.array(reward_progress, dtype=np.float32),
        reward_q_pen=np.array(reward_q_pen, dtype=np.float32),
        reward_g_pen=np.array(reward_g_pen, dtype=np.float32),
        reward_fuel_pen=np.array(reward_fuel_pen, dtype=np.float32),
        reward_tilt_pen=np.array(reward_tilt_pen, dtype=np.float32),
        reward_smooth_pen=np.array(reward_smooth_pen, dtype=np.float32),
        reward_high_stage_energy_bonus=np.array(reward_high_stage_energy_bonus, dtype=np.float32),
        reward_overshoot_pen=np.array(reward_overshoot_pen, dtype=np.float32),
    )

    max_altitude = float(np.max(z))
    max_q_dyn = float(np.max(q_dyn))
    apogee_idx = int(np.argmax(z))
    apogee_time = float(t[apogee_idx])
    burnout_idx = np.where(mass <= params.dry_mass + 1e-6)[0]
    time_to_burnout = float(t[burnout_idx[0]]) if burnout_idx.size > 0 else float("nan")
    total_downrange = float(np.linalg.norm([x[-1], y[-1]]))

    metrics = {
        "target_altitude_m": float(target_altitude_m),
        "max_altitude_m": max_altitude,
        "max_q_pa": max_q_dyn,
        "q_margin_pa": float(info.get("q_margin_pa", 0.0)),
        "q_over_limit_fraction": float(info.get("q_over_limit_fraction", 0.0)),
        "max_g_load": float(info.get("max_g_load", 0.0)),
        "apogee_time_s": apogee_time,
        "time_to_burnout_s": time_to_burnout,
        "altitude_at_burnout_m": float(info.get("altitude_at_burnout_m", float("nan"))),
        "velocity_at_burnout_mps": float(info.get("velocity_at_burnout_mps", float("nan"))),
        "vz_at_burnout_mps": float(info.get("vz_at_burnout_mps", float("nan"))),
        "total_downrange_m": total_downrange,
        "crash": bool(info.get("crash", False)),
        "hard_violation": bool(info.get("hard_violation", False)),
        "success": bool(info.get("success", False)),
        "terminal_altitude_m": float(info.get("altitude_m", 0.0)),
        "terminal_vx_mps": float(info.get("vx_mps", 0.0)),
        "terminal_vz_mps": float(info.get("vz_mps", 0.0)),
        "terminal_flight_path_angle_deg": float(info.get("flight_path_angle_deg", 0.0)),
        "fuel_used_fraction": float(info.get("fuel_used_fraction", 0.0)),
    }

    metrics_out = Path(metrics_path)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2))

    print(f"saved telemetry: {telemetry_out}")
    print(f"saved metrics: {metrics_out}")
    print(f"max altitude: {max_altitude:.3f} m")
    print(f"max q_dyn: {max_q_dyn:.3f} Pa")
    print(f"max g-load: {metrics['max_g_load']:.3f}")
    print(f"apogee time: {apogee_time:.3f} s")
    if np.isfinite(time_to_burnout):
        print(f"time to burnout: {time_to_burnout:.3f} s")
    else:
        print("time to burnout: not reached")
    print(f"total downrange: {total_downrange:.3f} m")
    print(
        "terminal: "
        f"crash={metrics['crash']} "
        f"hard_violation={metrics['hard_violation']} "
        f"success={metrics['success']}"
    )
    return telemetry_out, metrics_out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a trained rocket policy and save telemetry")
    parser.add_argument("--model-path", type=str, default=DEFAULT_BEST_FINAL_MODEL_PATH)
    parser.add_argument("--vecnorm-path", type=str, default=None)
    parser.add_argument("--telemetry-path", type=str, default="artifacts/living/telemetry.npz")
    parser.add_argument("--metrics-path", type=str, default="artifacts/living/metrics.json")
    parser.add_argument("--target-altitude-m", type=float, default=100_000.0)
    parser.add_argument(
        "--allow-missing-vecnorm",
        action="store_true",
        help="Allow replay without VecNormalize stats",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    replay(
        model_path=args.model_path,
        vecnorm_path=args.vecnorm_path,
        telemetry_path=args.telemetry_path,
        metrics_path=args.metrics_path,
        target_altitude_m=args.target_altitude_m,
        allow_missing_vecnorm=args.allow_missing_vecnorm,
    )
