"""Replay a trained PPO policy and save telemetry."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from config.defaults import EnvConfig, default_vehicle_params
from rl.env import RocketAscentEnv


def replay(
    model_path: str = "artifacts/ppo_rocket.zip",
    telemetry_path: str = "artifacts/telemetry.npz",
    metrics_path: str = "artifacts/metrics.json",
) -> tuple[Path, Path]:
    params = default_vehicle_params()
    cfg = EnvConfig()
    env = RocketAscentEnv(
        params=params,
        dt=cfg.dt,
        t_final=cfg.t_final,
        q_limit=cfg.q_limit,
        target_apogee=cfg.target_apogee,
        record=True,
    )
    model = PPO.load(model_path)

    obs, info = env.reset(seed=0, options={"record": True})
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        _ = reward

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
    )

    max_altitude = float(np.max(z))
    max_q_dyn = float(np.max(q_dyn))
    apogee_idx = int(np.argmax(z))
    apogee_time = float(t[apogee_idx])
    burnout_idx = np.where(mass <= params.dry_mass + 1e-6)[0]
    time_to_burnout = float(t[burnout_idx[0]]) if burnout_idx.size > 0 else float("nan")
    total_downrange = float(np.linalg.norm([x[-1], y[-1]]))

    metrics = {
        "max_altitude_m": max_altitude,
        "max_q_pa": max_q_dyn,
        "apogee_time_s": apogee_time,
        "time_to_burnout_s": time_to_burnout,
        "total_downrange_m": total_downrange,
        "crash": bool(info.get("crash", False)),
        "success": bool(info.get("apogee_reached", False)),
    }

    metrics_out = Path(metrics_path)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.write_text(json.dumps(metrics, indent=2))

    print(f"saved telemetry: {telemetry_out}")
    print(f"saved metrics: {metrics_out}")
    print(f"max altitude: {max_altitude:.3f} m")
    print(f"max q_dyn: {max_q_dyn:.3f} Pa")
    print(f"apogee time: {apogee_time:.3f} s")
    if np.isfinite(time_to_burnout):
        print(f"time to burnout: {time_to_burnout:.3f} s")
    else:
        print("time to burnout: not reached")
    print(f"total downrange: {total_downrange:.3f} m")
    print(f"terminal: crash={metrics['crash']} success={metrics['success']}")
    return telemetry_out, metrics_out


if __name__ == "__main__":
    replay()
