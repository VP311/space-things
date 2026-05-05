"""Shared helpers for deterministic evaluation and replay."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from config.defaults import AtmosphereConfig, CurriculumConfig, EnvConfig, MissionConfig, default_vehicle_params
from rl.env import RocketAscentEnv, baseline_action
from sim.vehicle import VehicleParams

DEFAULT_BEST_FINAL_MODEL_PATH = "artifacts/living/artifacts_train/best_final_model.zip"
DEFAULT_BEST_FINAL_VECNORM_PATH = "artifacts/living/artifacts_train/best_final_vecnormalize.pkl"
DEFAULT_FALLBACK_MODEL_PATH = "artifacts/living/artifacts_train/ppo_rocket.zip"
DEFAULT_FALLBACK_VECNORM_PATH = "artifacts/living/artifacts_train/vecnormalize.pkl"


def resolve_model_path(preferred_path: str) -> str:
    candidates = [
        Path(preferred_path),
        Path(DEFAULT_BEST_FINAL_MODEL_PATH),
        Path("artifacts/living/artifacts_train/best_model/best_model.zip"),
        Path(DEFAULT_FALLBACK_MODEL_PATH),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(Path(preferred_path))


def resolve_vecnorm_path(
    preferred_path: str | None,
    *,
    allow_missing: bool,
) -> str | None:
    candidates: list[Path] = []
    if preferred_path is not None:
        candidates.append(Path(preferred_path))
    else:
        candidates.extend([Path(DEFAULT_BEST_FINAL_VECNORM_PATH), Path(DEFAULT_FALLBACK_VECNORM_PATH)])

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    if allow_missing:
        return None
    searched = ", ".join(str(candidate) for candidate in candidates) if candidates else "<none>"
    raise FileNotFoundError(f"VecNormalize stats not found; looked for: {searched}")


def build_eval_env(
    target_altitude_m: float,
    *,
    record: bool = False,
    start_state_randomization: bool = True,
    mission: MissionConfig | None = None,
    atmosphere_cfg: AtmosphereConfig | None = None,
    env_cfg: EnvConfig | None = None,
    params: VehicleParams | None = None,
    domain_randomization: bool = True,
    observation_noise_std: float = 0.0,
    action_lag_steps: int = 0,
) -> RocketAscentEnv:
    mission = mission or MissionConfig()
    atmosphere_cfg = atmosphere_cfg or AtmosphereConfig()
    env_cfg = env_cfg or EnvConfig()
    curriculum_cfg = CurriculumConfig()
    env = RocketAscentEnv(
        params=params or default_vehicle_params(),
        mission=mission,
        atmosphere_cfg=atmosphere_cfg,
        curriculum_milestones_m=curriculum_cfg.target_milestones_m,
        dt=env_cfg.dt,
        t_final=env_cfg.t_final,
        record=record,
        start_state_randomization=start_state_randomization,
        domain_randomization=domain_randomization,
        action_repeat=env_cfg.action_repeat,
        observation_noise_std=observation_noise_std,
        action_lag_steps=action_lag_steps,
    )
    env.set_curriculum(target_altitude_m=float(target_altitude_m), start_altitude_cap_m=0.0)
    return env


def load_vecnormalize_for_eval(
    vecnorm_path: str,
    *,
    target_altitude_m: float,
) -> VecNormalize:
    dummy = DummyVecEnv([lambda: build_eval_env(target_altitude_m)])
    vecnorm = VecNormalize.load(vecnorm_path, dummy)
    vecnorm.training = False
    vecnorm.norm_reward = False
    return vecnorm


def normalise_obs(obs: np.ndarray, vecnorm: Any | None) -> np.ndarray:
    obs_arr = np.asarray(obs, dtype=np.float32)
    if vecnorm is None:
        return obs_arr
    rms = vecnorm.obs_rms
    normed = (obs_arr - rms.mean) / np.sqrt(rms.var + vecnorm.epsilon)
    normed = np.clip(normed, -vecnorm.clip_obs, vecnorm.clip_obs)
    return np.asarray(normed, dtype=np.float32)


def run_episode(
    env: RocketAscentEnv,
    *,
    mode: str,
    seed: int,
    model: PPO | Any | None = None,
    vecnorm: Any | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    if mode not in {"baseline", "trained"}:
        raise ValueError("mode must be 'baseline' or 'trained'")
    if mode == "trained" and model is None:
        raise ValueError("trained mode requires a model")

    obs, info = env.reset(seed=seed)
    terminated = False
    truncated = False
    steps = 0
    is_recurrent = isinstance(model, RecurrentPPO)
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool) if is_recurrent else None
    while not (terminated or truncated):
        if mode == "baseline":
            action = baseline_action(
                env.t,
                env.params.max_gimbal,
                altitude_m=float(info["altitude_m"]),
                q_dyn_pa=float(info["q_dyn_pa"]),
                q_limit_pa=env.mission.max_q_pa,
            )
        elif is_recurrent:
            model_obs = normalise_obs(obs, vecnorm)
            action, lstm_states = model.predict(
                model_obs, state=lstm_states, episode_start=episode_starts, deterministic=deterministic
            )
            if isinstance(action, np.ndarray) and action.ndim > 1:
                action = action[0]
            episode_starts = np.zeros((1,), dtype=bool)
        else:
            model_obs = normalise_obs(obs, vecnorm)
            action, _ = model.predict(model_obs, deterministic=deterministic)
            if isinstance(action, np.ndarray) and action.ndim > 1:
                action = action[0]
        obs, reward, terminated, truncated, info = env.step(action)
        _ = reward
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


def summarise_rows(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    mission: MissionConfig,
    target_altitude_m: float,
) -> dict[str, Any]:
    success = np.array([row["success"] for row in rows], dtype=bool)
    crash = np.array([row["crash"] for row in rows], dtype=bool)
    hard_violation = np.array([row["hard_violation"] for row in rows], dtype=bool)
    max_altitude = np.array([row["max_altitude_m"] for row in rows], dtype=float)
    max_q = np.array([row["max_q_dyn"] for row in rows], dtype=float)
    q_margin = np.array([row.get("q_margin_pa", 0.0) for row in rows], dtype=float)
    q_over_limit_fraction = np.array([row.get("q_over_limit_fraction", 0.0) for row in rows], dtype=float)
    max_g = np.array([row["max_g_load"] for row in rows], dtype=float)
    terminal_altitude = np.array([row["terminal_altitude_m"] for row in rows], dtype=float)
    terminal_vx = np.array([row["terminal_vx_mps"] for row in rows], dtype=float)
    terminal_vz = np.array([row["terminal_vz_mps"] for row in rows], dtype=float)
    terminal_gamma = np.array([row["terminal_flight_path_angle_deg"] for row in rows], dtype=float)
    fuel_used = np.array([row["fuel_used_fraction"] for row in rows], dtype=float)
    burnout_vz = np.array([row["vz_at_burnout_mps"] for row in rows], dtype=float)
    burnout_altitude = np.array([row["altitude_at_burnout_m"] for row in rows], dtype=float)

    q_ok = max_q <= mission.max_q_pa
    g_ok = max_g <= mission.max_g_load

    return {
        "mode": mode,
        "n_episodes": len(rows),
        "target_altitude_m": float(target_altitude_m),
        "success_rate": float(np.mean(success)),
        "crash_rate": float(np.mean(crash)),
        "hard_violation_rate": float(np.mean(hard_violation)),
        "q_ok_rate": float(np.mean(q_ok)),
        "q_violation_rate": float(np.mean(~q_ok)),
        "g_ok_rate": float(np.mean(g_ok)),
        "g_violation_rate": float(np.mean(~g_ok)),
        "avg_max_altitude_m": float(np.mean(max_altitude)),
        "avg_max_q": float(np.mean(max_q)),
        "avg_q_margin_pa": float(np.mean(q_margin)),
        "avg_q_over_limit_fraction": float(np.mean(q_over_limit_fraction)),
        "avg_max_g_load": float(np.mean(max_g)),
        "avg_burnout_altitude_m": float(np.nanmean(burnout_altitude)),
        "avg_burnout_vz_mps": float(np.nanmean(burnout_vz)),
        "avg_terminal_altitude_m": float(np.mean(terminal_altitude)),
        "avg_terminal_vx_mps": float(np.mean(terminal_vx)),
        "avg_terminal_vz_mps": float(np.mean(terminal_vz)),
        "avg_terminal_flight_path_angle_deg": float(np.mean(terminal_gamma)),
        "avg_fuel_used_fraction": float(np.mean(fuel_used)),
        "episodes": rows,
    }


def evaluate_env(
    env: RocketAscentEnv,
    *,
    mode: str,
    n_episodes: int,
    seed_start: int,
    model: PPO | Any | None = None,
    vecnorm: Any | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    rows = [
        run_episode(
            env,
            mode=mode,
            seed=seed_start + idx,
            model=model,
            vecnorm=vecnorm,
            deterministic=deterministic,
        )
        for idx in range(n_episodes)
    ]
    return summarise_rows(
        rows,
        mode=mode,
        mission=env.mission,
        target_altitude_m=env.curriculum_target_altitude_m,
    )


def evaluate_policy(
    *,
    mode: str,
    n_episodes: int,
    seed_start: int,
    target_altitude_m: float,
    model: PPO | Any | None = None,
    vecnorm: Any | None = None,
    deterministic: bool = True,
) -> dict[str, Any]:
    env = build_eval_env(target_altitude_m)
    return evaluate_env(
        env,
        mode=mode,
        n_episodes=n_episodes,
        seed_start=seed_start,
        model=model,
        vecnorm=vecnorm,
        deterministic=deterministic,
    )
