"""Train PPO on the rocket ascent environment with curriculum and diagnostics."""

from __future__ import annotations

import argparse
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from config.defaults import (
    AtmosphereConfig,
    CurriculumConfig,
    EnvConfig,
    MissionConfig,
    PPOConfig,
    default_vehicle_params,
)
from rl.anchored_ppo import AnchoredPPO
from rl.env import RocketAscentEnv
from rl.policy_eval import build_eval_env, evaluate_env
from rl.select_checkpoint import wilson_interval


@dataclass
class CurriculumManager:
    current_target_m: float
    target_milestones_m: tuple[float, ...]
    promotion_window_episodes: int
    start_altitude_fraction: float
    low_stage_success_threshold: float
    high_stage_threshold_m: float
    high_stage_success_threshold: float
    high_stage_altitude_ratio_threshold: float

    def __post_init__(self) -> None:
        milestones = tuple(sorted({float(m) for m in self.target_milestones_m} | {float(self.current_target_m)}))
        self.target_milestones_m = milestones
        self._current_idx = milestones.index(float(self.current_target_m))
        self.success_history: deque[float] = deque(maxlen=self.promotion_window_episodes)
        self.altitude_ratio_history: deque[float] = deque(maxlen=self.promotion_window_episodes)
        self.q_ok_history: deque[float] = deque(maxlen=self.promotion_window_episodes)
        self.last_decision: dict[str, Any] = {
            "rolling_success": 0.0,
            "rolling_altitude_ratio": 0.0,
            "rolling_q_ok": 0.0,
            "ready": False,
            "reason": "warming_up",
        }

    def record(self, success: bool, altitude_ratio: float, q_ok: bool) -> tuple[bool, float]:
        self.success_history.append(1.0 if success else 0.0)
        self.altitude_ratio_history.append(float(np.clip(altitude_ratio, 0.0, 2.0)))
        self.q_ok_history.append(1.0 if q_ok else 0.0)
        if len(self.success_history) < self.promotion_window_episodes:
            self.last_decision = {
                "rolling_success": sum(self.success_history) / len(self.success_history),
                "rolling_altitude_ratio": sum(self.altitude_ratio_history) / len(self.altitude_ratio_history),
                "rolling_q_ok": sum(self.q_ok_history) / len(self.q_ok_history),
                "ready": False,
                "reason": "warming_up",
            }
            return False, self.current_target_m
        if self._current_idx >= len(self.target_milestones_m) - 1:
            self.last_decision = {
                "rolling_success": sum(self.success_history) / len(self.success_history),
                "rolling_altitude_ratio": sum(self.altitude_ratio_history) / len(self.altitude_ratio_history),
                "rolling_q_ok": sum(self.q_ok_history) / len(self.q_ok_history),
                "ready": False,
                "reason": "final_target_reached",
            }
            return False, self.current_target_m

        rolling_success = sum(self.success_history) / len(self.success_history)
        rolling_alt_ratio = sum(self.altitude_ratio_history) / len(self.altitude_ratio_history)
        rolling_q_ok = sum(self.q_ok_history) / len(self.q_ok_history)
        # High stages (>=65km) require 70% success + altitude ratio check.
        # Low/mid stages use a lower 45% threshold so curriculum progresses naturally.
        threshold = self.high_stage_success_threshold if self.current_target_m >= self.high_stage_threshold_m else self.low_stage_success_threshold
        ready = rolling_success >= threshold
        reason = "success_below_threshold"
        if self.current_target_m >= self.high_stage_threshold_m:
            if ready and rolling_alt_ratio < self.high_stage_altitude_ratio_threshold:
                ready = False
                reason = "altitude_ratio_below_threshold"
            elif ready:
                reason = "promote"
        elif ready:
            reason = "promote"
        self.last_decision = {
            "rolling_success": rolling_success,
            "rolling_altitude_ratio": rolling_alt_ratio,
            "rolling_q_ok": rolling_q_ok,
            "ready": ready,
            "reason": reason,
        }
        if not ready:
            return False, self.current_target_m

        self._current_idx += 1
        self.current_target_m = self.target_milestones_m[self._current_idx]
        self.success_history.clear()
        self.altitude_ratio_history.clear()
        self.q_ok_history.clear()
        return True, self.current_target_m

    def start_altitude_cap(self) -> float:
        return max(0.0, self.current_target_m * self.start_altitude_fraction)

    def final_target_m(self) -> float:
        return float(self.target_milestones_m[-1])


class EpisodeSummaryCallback(BaseCallback):
    def __init__(
        self,
        out_path: Path,
        curriculum: CurriculumManager | None,
        curriculum_sync_envs: list[RocketAscentEnv] | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.out_path = out_path
        self.curriculum = curriculum
        self.curriculum_sync_envs = curriculum_sync_envs or []
        self._fh: Any | None = None
        self._episode_idx = 0

    def _on_training_start(self) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.out_path.open("w", encoding="utf-8")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        if self._fh is None:
            return True

        for done, info in zip(dones, infos):
            if not done:
                continue
            self._episode_idx += 1

            reward_sums = dict(info.get("reward_components_cumulative", {}))
            row = {
                "episode": self._episode_idx,
                "t": float(info.get("t", 0.0)),
                "max_altitude_m": float(info.get("max_altitude_m", 0.0)),
                "altitude_at_burnout_m": float(info.get("altitude_at_burnout_m", float("nan"))),
                "velocity_at_burnout_mps": float(info.get("velocity_at_burnout_mps", float("nan"))),
                "vz_at_burnout_mps": float(info.get("vz_at_burnout_mps", float("nan"))),
                "altitude_at_apogee_m": float(info.get("altitude_at_apogee_m", 0.0)),
                "time_to_apogee_s": float(info.get("time_to_apogee_s", float("nan"))),
                "max_q_dyn": float(info.get("max_q_dyn", 0.0)),
                "q_margin_pa": float(info.get("q_margin_pa", 0.0)),
                "q_over_limit_fraction": float(info.get("q_over_limit_fraction", 0.0)),
                "max_g_load": float(info.get("max_g_load", 0.0)),
                "fuel_used_fraction": float(info.get("fuel_used_fraction", 0.0)),
                "success": bool(info.get("success", False)),
                "termination_reason": str(info.get("termination_reason", "unknown")),
                "mean_throttle": float(info.get("mean_throttle", 0.0)),
                "throttle_variance": float(info.get("throttle_variance", 0.0)),
                "gimbal_variance": float(info.get("gimbal_variance", 0.0)),
                "curriculum_target_altitude_m": float(info.get("mission", {}).get("target_altitude_m", 0.0)),
                "altitude_ratio_to_target": float(info.get("max_altitude_m", 0.0))
                / max(float(info.get("mission", {}).get("target_altitude_m", 1.0)), 1.0),
                "reward_components": reward_sums,
            }

            if self.curriculum is not None:
                target_now = float(info.get("mission", {}).get("target_altitude_m", 1.0))
                max_alt = float(info.get("max_altitude_m", 0.0))
                altitude_ratio = max_alt / max(target_now, 1.0)
                q_ok = float(info.get("max_q_dyn", 0.0)) <= float(info.get("mission", {}).get("max_q_pa", float("inf")))
                promoted, target = self.curriculum.record(bool(info.get("success", False)), altitude_ratio, q_ok=q_ok)
                row["curriculum_window"] = dict(self.curriculum.last_decision)
                if promoted:
                    cap = self.curriculum.start_altitude_cap()
                    self.training_env.env_method(
                        "set_curriculum",
                        target_altitude_m=float(target),
                        start_altitude_cap_m=float(cap),
                    )
                    for env in self.curriculum_sync_envs:
                        env.set_curriculum(target_altitude_m=float(target), start_altitude_cap_m=float(cap))
                    if self.verbose > 0:
                        print(
                            "curriculum promote: "
                            f"target_altitude_m={target:.1f} start_altitude_cap_m={cap:.1f} "
                            f"rolling_success={self.curriculum.last_decision['rolling_success']:.3f} "
                            f"rolling_altitude_ratio={self.curriculum.last_decision['rolling_altitude_ratio']:.3f} "
                            f"rolling_q_ok={self.curriculum.last_decision['rolling_q_ok']:.3f} "
                            f"reason={self.curriculum.last_decision['reason']}"
                        )
                elif self.verbose > 0 and self._episode_idx % max(self.curriculum.promotion_window_episodes, 1) == 0:
                    print(
                        "curriculum hold: "
                        f"target_altitude_m={target_now:.1f} "
                        f"rolling_success={self.curriculum.last_decision['rolling_success']:.3f} "
                        f"rolling_altitude_ratio={self.curriculum.last_decision['rolling_altitude_ratio']:.3f} "
                        f"rolling_q_ok={self.curriculum.last_decision['rolling_q_ok']:.3f} "
                        f"reason={self.curriculum.last_decision['reason']}"
                    )

            self._fh.write(json.dumps(row) + "\n")

        self._fh.flush()
        return True

    def _on_training_end(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


class FixedFinalEvalCallback(BaseCallback):
    def __init__(
        self,
        *,
        progress_env: RocketAscentEnv,
        fixed_final_env: RocketAscentEnv,
        eval_freq_steps: int,
        eval_episodes: int,
        final_eval_seed_start: int,
        official_eval_freq_steps: int,
        official_eval_episodes: int,
        official_eval_seed_start: int,
        official_success_threshold: float,
        official_success_streak: int,
        best_model_path: Path,
        best_vecnorm_path: Path,
        progress_latest_path: Path,
        fixed_final_latest_path: Path,
        official_latest_path: Path,
        progress_history_path: Path,
        fixed_final_history_path: Path,
        official_history_path: Path,
        guardrail_min_success_rate: float | None = None,
        guardrail_min_q_ok_rate: float | None = None,
        guardrail_min_g_ok_rate: float | None = None,
        guardrail_min_success_lcb: float | None = None,
        guardrail_min_q_ok_lcb: float | None = None,
        guardrail_min_g_ok_lcb: float | None = None,
        guardrail_min_timesteps: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.progress_env = progress_env
        self.fixed_final_env = fixed_final_env
        self.eval_freq_steps = int(eval_freq_steps)
        self.eval_episodes = int(eval_episodes)
        self.final_eval_seed_start = int(final_eval_seed_start)
        self.official_eval_freq_steps = int(official_eval_freq_steps)
        self.official_eval_episodes = int(official_eval_episodes)
        self.official_eval_seed_start = int(official_eval_seed_start)
        self.official_success_threshold = float(official_success_threshold)
        self.official_success_streak = int(official_success_streak)
        self.guardrail_min_success_rate = guardrail_min_success_rate
        self.guardrail_min_q_ok_rate = guardrail_min_q_ok_rate
        self.guardrail_min_g_ok_rate = guardrail_min_g_ok_rate
        self.guardrail_min_success_lcb = guardrail_min_success_lcb
        self.guardrail_min_q_ok_lcb = guardrail_min_q_ok_lcb
        self.guardrail_min_g_ok_lcb = guardrail_min_g_ok_lcb
        self.guardrail_min_timesteps = int(max(0, guardrail_min_timesteps))
        self.best_model_path = best_model_path
        self.best_model_base = best_model_path.with_suffix("")
        self.best_vecnorm_path = best_vecnorm_path
        self.progress_latest_path = progress_latest_path
        self.fixed_final_latest_path = fixed_final_latest_path
        self.official_latest_path = official_latest_path
        self.progress_history_path = progress_history_path
        self.fixed_final_history_path = fixed_final_history_path
        self.official_history_path = official_history_path
        self._best_success_rate = -1.0
        self._best_mean_max_altitude = -1.0
        self._official_streak = 0

    def _on_training_start(self) -> None:
        for path in (
            self.progress_latest_path,
            self.fixed_final_latest_path,
            self.official_latest_path,
            self.progress_history_path,
            self.fixed_final_history_path,
            self.official_history_path,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                path.write_text("{}")
            elif path.suffix == ".jsonl":
                path.write_text("")

    def _evaluate_raw_env(self, env: RocketAscentEnv, *, n_episodes: int, seed_start: int) -> dict[str, Any]:
        vecnorm = self.training_env if isinstance(self.training_env, VecNormalize) else None
        return evaluate_env(
            env,
            mode="trained",
            n_episodes=n_episodes,
            seed_start=seed_start,
            model=self.model,
            vecnorm=vecnorm,
            deterministic=True,
        )

    def _write_result(self, latest_path: Path, history_path: Path, result: dict[str, Any]) -> None:
        latest_path.write_text(json.dumps(result, indent=2))
        with history_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(result) + "\n")

    def _record_metrics(self, prefix: str, result: dict[str, Any]) -> None:
        self.logger.record(f"{prefix}/success_rate", result["success_rate"])
        self.logger.record(f"{prefix}/q_ok_rate", result["q_ok_rate"])
        self.logger.record(f"{prefix}/avg_max_altitude_m", result["avg_max_altitude_m"])
        self.logger.record(f"{prefix}/avg_burnout_vz_mps", result["avg_burnout_vz_mps"])

    @staticmethod
    def _count_from_rate(result: dict[str, Any], key: str) -> int:
        n_episodes = int(result.get("n_episodes", len(result.get("episodes", [])) or 0))
        return int(round(float(result.get(key, 0.0)) * n_episodes))

    def _guardrail_decision(self, label: str, result: dict[str, Any]) -> tuple[bool, dict[str, Any]]:
        if self.num_timesteps < self.guardrail_min_timesteps:
            return True, {"active": False, "reason": "before_min_timesteps"}

        n_episodes = int(result.get("n_episodes", len(result.get("episodes", [])) or 0))
        success_count = self._count_from_rate(result, "success_rate")
        q_ok_count = self._count_from_rate(result, "q_ok_rate")
        g_ok_count = self._count_from_rate(result, "g_ok_rate")
        success_lcb, success_ucb = wilson_interval(success_count, n_episodes)
        q_lcb, q_ucb = wilson_interval(q_ok_count, n_episodes)
        g_lcb, g_ucb = wilson_interval(g_ok_count, n_episodes)
        checks = {
            "active": True,
            "label": label,
            "n_episodes": n_episodes,
            "success_count": success_count,
            "q_ok_count": q_ok_count,
            "g_ok_count": g_ok_count,
            "success_lcb": success_lcb,
            "success_ucb": success_ucb,
            "q_ok_lcb": q_lcb,
            "q_ok_ucb": q_ucb,
            "g_ok_lcb": g_lcb,
            "g_ok_ucb": g_ucb,
            "breaches": [],
        }

        thresholds = (
            ("success_rate", self.guardrail_min_success_rate, float(result.get("success_rate", 0.0))),
            ("q_ok_rate", self.guardrail_min_q_ok_rate, float(result.get("q_ok_rate", 0.0))),
            ("g_ok_rate", self.guardrail_min_g_ok_rate, float(result.get("g_ok_rate", 0.0))),
            ("success_lcb", self.guardrail_min_success_lcb, success_lcb),
            ("q_ok_lcb", self.guardrail_min_q_ok_lcb, q_lcb),
            ("g_ok_lcb", self.guardrail_min_g_ok_lcb, g_lcb),
        )
        for name, minimum, observed in thresholds:
            if minimum is not None and observed < float(minimum):
                checks["breaches"].append({"metric": name, "observed": observed, "minimum": float(minimum)})

        return len(checks["breaches"]) == 0, checks

    def _save_best_if_needed(self, result: dict[str, Any]) -> None:
        is_better = result["success_rate"] > self._best_success_rate + 1e-12
        is_tie_break = (
            abs(result["success_rate"] - self._best_success_rate) <= 1e-12
            and result["avg_max_altitude_m"] > self._best_mean_max_altitude + 1e-6
        )
        if not (is_better or is_tie_break):
            return

        self._best_success_rate = float(result["success_rate"])
        self._best_mean_max_altitude = float(result["avg_max_altitude_m"])
        self.model.save(str(self.best_model_base))
        if hasattr(self.training_env, "save"):
            self.training_env.save(str(self.best_vecnorm_path))
        if self.verbose > 0:
            print(
                "fixed-final best: "
                f"success_rate={result['success_rate']:.3f} "
                f"avg_max_altitude_m={result['avg_max_altitude_m']:.1f}"
            )

    def _on_step(self) -> bool:
        if self.eval_freq_steps > 0 and self.num_timesteps % self.eval_freq_steps == 0:
            progress_result = self._evaluate_raw_env(
                self.progress_env,
                n_episodes=self.eval_episodes,
                seed_start=1_000,
            )
            progress_result["num_timesteps"] = self.num_timesteps
            fixed_final_result = self._evaluate_raw_env(
                self.fixed_final_env,
                n_episodes=self.eval_episodes,
                seed_start=self.final_eval_seed_start,
            )
            fixed_final_result["num_timesteps"] = self.num_timesteps
            guardrail_ok, guardrail = self._guardrail_decision("fixed_final_eval", fixed_final_result)
            fixed_final_result["guardrail"] = guardrail
            self._write_result(self.progress_latest_path, self.progress_history_path, progress_result)
            self._write_result(self.fixed_final_latest_path, self.fixed_final_history_path, fixed_final_result)
            self._record_metrics("progress_eval", progress_result)
            self._record_metrics("fixed_final_eval", fixed_final_result)
            if not guardrail_ok:
                if self.verbose > 0:
                    print(f"early stop: guardrail breach on fixed_final_eval: {guardrail['breaches']}")
                return False
            self._save_best_if_needed(fixed_final_result)

        reached_final_target = (
            self.progress_env.curriculum_target_altitude_m >= self.fixed_final_env.curriculum_target_altitude_m - 1e-6
        )
        if (
            reached_final_target
            and self.official_eval_freq_steps > 0
            and self.num_timesteps % self.official_eval_freq_steps == 0
        ):
            official_result = self._evaluate_raw_env(
                self.fixed_final_env,
                n_episodes=self.official_eval_episodes,
                seed_start=self.official_eval_seed_start,
            )
            official_result["num_timesteps"] = self.num_timesteps
            guardrail_ok, guardrail = self._guardrail_decision("official_fixed_final_eval", official_result)
            official_result["guardrail"] = guardrail
            self._write_result(self.official_latest_path, self.official_history_path, official_result)
            self._record_metrics("official_fixed_final_eval", official_result)
            if official_result["success_rate"] >= self.official_success_threshold:
                self._official_streak += 1
            else:
                self._official_streak = 0
            if self._official_streak >= self.official_success_streak:
                if self.verbose > 0:
                    print(
                        "early stop: "
                        f"official fixed-final success_rate={official_result['success_rate']:.3f} "
                        f"for {self._official_streak} consecutive evaluations"
                )
                return False
            if not guardrail_ok:
                if self.verbose > 0:
                    print(f"early stop: guardrail breach on official_fixed_final_eval: {guardrail['breaches']}")
                return False

        return True


class VecNormalizeCheckpointCallback(BaseCallback):
    """Save model checkpoints and matching VecNormalize stats at the same timestep."""

    def __init__(self, *, save_freq: int, save_path: Path, name_prefix: str = "ppo_rocket", verbose: int = 0) -> None:
        super().__init__(verbose)
        self.save_freq = int(max(1, save_freq))
        self.save_path = Path(save_path)
        self.name_prefix = str(name_prefix)

    def _on_training_start(self) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        model_base = self.save_path / f"{self.name_prefix}_{self.num_timesteps}_steps"
        vecnorm_path = self.save_path / f"vecnormalize_{self.num_timesteps}_steps.pkl"
        self.model.save(str(model_base))
        if hasattr(self.training_env, "save"):
            self.training_env.save(str(vecnorm_path))
        if self.verbose > 0:
            print(f"checkpoint saved: {model_base.with_suffix('.zip')} + {vecnorm_path}")
        return True


class EntCoefScheduleCallback(BaseCallback):
    """Linearly decay entropy coefficient from start to end over training."""

    def __init__(self, start: float, end: float, total_timesteps: int) -> None:
        super().__init__()
        self._start = float(start)
        self._end = float(end)
        self._total = max(1, int(total_timesteps))

    def _on_rollout_start(self) -> None:
        frac = min(self.num_timesteps / self._total, 1.0)
        self.model.ent_coef = self._start + (self._end - self._start) * frac

    def _on_step(self) -> bool:
        return True


def _build_env(
    env_cfg: EnvConfig,
    mission: MissionConfig,
    atmosphere_cfg: AtmosphereConfig,
    *,
    initial_target_altitude_m: float,
    curriculum_cfg: CurriculumConfig,
) -> RocketAscentEnv:
    env = RocketAscentEnv(
        params=default_vehicle_params(),
        mission=mission,
        atmosphere_cfg=atmosphere_cfg,
        curriculum_milestones_m=curriculum_cfg.target_milestones_m,
        dt=env_cfg.dt,
        t_final=env_cfg.t_final,
        record=False,
        max_start_altitude_m=initial_target_altitude_m * curriculum_cfg.start_altitude_fraction,
        action_repeat=env_cfg.action_repeat,
    )
    env.set_curriculum(
        target_altitude_m=initial_target_altitude_m,
        start_altitude_cap_m=initial_target_altitude_m * curriculum_cfg.start_altitude_fraction,
    )
    return env


def train(
    total_timesteps: int | None = None,
    n_envs: int | None = None,
    initial_target_altitude_m: float | None = None,
    curriculum_enabled: bool | None = None,
    ent_coef: float | None = None,
    ent_coef_end: float | None = None,
    lr_start: float | None = None,
    lr_end: float | None = None,
    clip_range: float | None = None,
    target_kl: float | None = None,
    seed: int | None = None,
    eval_freq_steps: int | None = None,
    eval_episodes: int | None = None,
    official_eval_freq_steps: int | None = None,
    official_eval_episodes: int | None = None,
    checkpoint_freq_steps: int | None = None,
    guardrail_min_success_rate: float | None = None,
    guardrail_min_q_ok_rate: float | None = None,
    guardrail_min_g_ok_rate: float | None = None,
    guardrail_min_success_lcb: float | None = None,
    guardrail_min_q_ok_lcb: float | None = None,
    guardrail_min_g_ok_lcb: float | None = None,
    guardrail_min_timesteps: int = 0,
    load_model_path: str | None = None,
    load_vecnorm_path: str | None = None,
    anchor_ref_model_path: str | None = None,
    anchor_coef: float = 0.0,
    artifacts_dir: str | None = None,
    use_lstm: bool | None = None,
) -> Path:
    env_cfg = EnvConfig()
    mission = MissionConfig()
    atmosphere_cfg = AtmosphereConfig()
    ppo_cfg = PPOConfig()
    curriculum_cfg = CurriculumConfig()

    run_steps = int(total_timesteps if total_timesteps is not None else ppo_cfg.total_timesteps)
    num_envs = int(n_envs if n_envs is not None else ppo_cfg.n_envs)
    start_target = float(
        initial_target_altitude_m if initial_target_altitude_m is not None else curriculum_cfg.initial_target_altitude_m
    )
    use_curriculum = curriculum_cfg.enabled if curriculum_enabled is None else bool(curriculum_enabled)
    run_ent_coef_start = float(ent_coef if ent_coef is not None else ppo_cfg.ent_coef_start)
    run_ent_coef_end = float(ent_coef_end if ent_coef_end is not None else ppo_cfg.ent_coef_end)
    run_lr_start = float(lr_start if lr_start is not None else ppo_cfg.lr_start)
    run_lr_end = float(lr_end if lr_end is not None else ppo_cfg.lr_end)
    run_clip_range = float(clip_range if clip_range is not None else ppo_cfg.clip_range)
    run_target_kl = float(target_kl if target_kl is not None else ppo_cfg.target_kl)
    run_seed = int(seed if seed is not None else ppo_cfg.seed)
    run_eval_freq_steps = int(eval_freq_steps if eval_freq_steps is not None else ppo_cfg.eval_freq_steps)
    run_eval_episodes = int(eval_episodes if eval_episodes is not None else ppo_cfg.eval_episodes)
    run_official_eval_freq_steps = int(
        official_eval_freq_steps if official_eval_freq_steps is not None else ppo_cfg.official_eval_freq_steps
    )
    run_official_eval_episodes = int(
        official_eval_episodes if official_eval_episodes is not None else ppo_cfg.official_eval_episodes
    )
    run_checkpoint_freq_steps = int(
        checkpoint_freq_steps if checkpoint_freq_steps is not None else ppo_cfg.checkpoint_freq_steps
    )
    run_use_lstm = bool(use_lstm if use_lstm is not None else ppo_cfg.use_lstm)
    run_anchor_coef = float(max(0.0, anchor_coef))
    use_anchor = bool(anchor_ref_model_path is not None and run_anchor_coef > 0.0)
    if use_anchor and run_use_lstm:
        raise ValueError("anchored PPO fine-tuning currently supports plain PPO only; pass --no-lstm")
    fixed_final_target = float(max(curriculum_cfg.target_milestones_m))

    raw_train_env = make_vec_env(
        lambda: _build_env(
            env_cfg,
            mission,
            atmosphere_cfg,
            initial_target_altitude_m=start_target,
            curriculum_cfg=curriculum_cfg,
        ),
        n_envs=num_envs,
        seed=run_seed,
    )
    if load_vecnorm_path is not None:
        train_env = VecNormalize.load(load_vecnorm_path, raw_train_env)
        train_env.training = True
        train_env.norm_reward = True
    else:
        train_env = VecNormalize(raw_train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    progress_eval_env = build_eval_env(start_target)
    fixed_final_eval_env = build_eval_env(fixed_final_target)

    artifacts_dir = Path(artifacts_dir) if artifacts_dir is not None else Path("artifacts/living/artifacts_train")
    checkpoints_dir = artifacts_dir / "checkpoints"
    eval_logs_dir = artifacts_dir / "eval_logs"
    tensorboard_dir = artifacts_dir / "tb"
    summaries_path = artifacts_dir / "episode_summaries.jsonl"
    best_final_model_path = artifacts_dir / "best_final_model.zip"
    best_final_vecnorm_path = artifacts_dir / "best_final_vecnormalize.pkl"
    progress_latest_path = eval_logs_dir / "progress_eval_latest.json"
    fixed_final_latest_path = eval_logs_dir / "fixed_final_eval_latest.json"
    official_latest_path = eval_logs_dir / "official_fixed_final_eval_latest.json"
    progress_history_path = eval_logs_dir / "progress_eval_history.jsonl"
    fixed_final_history_path = eval_logs_dir / "fixed_final_eval_history.jsonl"
    official_history_path = eval_logs_dir / "official_fixed_final_eval_history.jsonl"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    eval_logs_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)

    tensorboard_log: str | None = str(tensorboard_dir)
    try:
        import tensorboard  # type: ignore  # noqa: F401
    except Exception:
        tensorboard_log = None
        print("tensorboard not installed; training will continue without tensorboard logs.")

    checkpoint_freq = max(run_checkpoint_freq_steps // num_envs, 1)

    curriculum_mgr: CurriculumManager | None = None
    if use_curriculum:
        curriculum_mgr = CurriculumManager(
            current_target_m=start_target,
            target_milestones_m=curriculum_cfg.target_milestones_m,
            promotion_window_episodes=curriculum_cfg.promotion_window_episodes,
            start_altitude_fraction=curriculum_cfg.start_altitude_fraction,
            low_stage_success_threshold=curriculum_cfg.low_stage_success_threshold,
            high_stage_threshold_m=curriculum_cfg.high_stage_threshold_m,
            high_stage_success_threshold=curriculum_cfg.high_stage_success_threshold,
            high_stage_altitude_ratio_threshold=curriculum_cfg.high_stage_altitude_ratio_threshold,
        )

    callbacks = CallbackList(
        [
            EntCoefScheduleCallback(
                start=run_ent_coef_start,
                end=run_ent_coef_end,
                total_timesteps=run_steps,
            ),
            EpisodeSummaryCallback(
                summaries_path,
                curriculum=curriculum_mgr,
                curriculum_sync_envs=[progress_eval_env],
                verbose=1,
            ),
            VecNormalizeCheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=checkpoints_dir,
                name_prefix="ppo_rocket",
                verbose=1,
            ),
            FixedFinalEvalCallback(
                progress_env=progress_eval_env,
                fixed_final_env=fixed_final_eval_env,
                eval_freq_steps=run_eval_freq_steps,
                eval_episodes=run_eval_episodes,
                final_eval_seed_start=ppo_cfg.final_eval_seed_start,
                official_eval_freq_steps=run_official_eval_freq_steps,
                official_eval_episodes=run_official_eval_episodes,
                official_eval_seed_start=ppo_cfg.official_eval_seed_start,
                official_success_threshold=ppo_cfg.official_success_threshold,
                official_success_streak=ppo_cfg.official_success_streak,
                guardrail_min_success_rate=guardrail_min_success_rate,
                guardrail_min_q_ok_rate=guardrail_min_q_ok_rate,
                guardrail_min_g_ok_rate=guardrail_min_g_ok_rate,
                guardrail_min_success_lcb=guardrail_min_success_lcb,
                guardrail_min_q_ok_lcb=guardrail_min_q_ok_lcb,
                guardrail_min_g_ok_lcb=guardrail_min_g_ok_lcb,
                guardrail_min_timesteps=guardrail_min_timesteps,
                best_model_path=best_final_model_path,
                best_vecnorm_path=best_final_vecnorm_path,
                progress_latest_path=progress_latest_path,
                fixed_final_latest_path=fixed_final_latest_path,
                official_latest_path=official_latest_path,
                progress_history_path=progress_history_path,
                fixed_final_history_path=fixed_final_history_path,
                official_history_path=official_history_path,
                verbose=1,
            ),
        ]
    )

    def _lr_schedule(progress_remaining: float) -> float:
        return run_lr_end + (run_lr_start - run_lr_end) * progress_remaining

    if load_model_path is not None:
        loader = RecurrentPPO if run_use_lstm else (AnchoredPPO if use_anchor else PPO)
        model = loader.load(
            load_model_path,
            env=train_env,
            tensorboard_log=tensorboard_log,
            verbose=1,
        )
        model.learning_rate = _lr_schedule
        model.lr_schedule = _lr_schedule
        model.clip_range = lambda _: run_clip_range
        model.target_kl = run_target_kl
        model.ent_coef = run_ent_coef_start
        if use_anchor:
            assert isinstance(model, AnchoredPPO)
            reference = PPO.load(str(anchor_ref_model_path), env=train_env)
            model.set_reference_policy(reference.policy, anchor_coef=run_anchor_coef)
        print(f"Warm-started from {load_model_path}")
    elif run_use_lstm:
        model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=train_env,
            n_steps=ppo_cfg.n_steps,
            batch_size=ppo_cfg.batch_size,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            learning_rate=_lr_schedule,
            ent_coef=run_ent_coef_start,
            clip_range=run_clip_range,
            n_epochs=ppo_cfg.n_epochs,
            vf_coef=ppo_cfg.vf_coef,
            policy_kwargs={
                "net_arch": list(ppo_cfg.net_arch),
                "lstm_hidden_size": ppo_cfg.lstm_hidden_size,
                "n_lstm_layers": 1,
                "shared_lstm": False,
                "enable_critic_lstm": True,
            },
            target_kl=run_target_kl,
            tensorboard_log=tensorboard_log,
            seed=run_seed,
            verbose=1,
        )
    else:
        model_cls = AnchoredPPO if use_anchor else PPO
        model = model_cls(
            policy="MlpPolicy",
            env=train_env,
            n_steps=ppo_cfg.n_steps,
            batch_size=ppo_cfg.batch_size,
            gamma=ppo_cfg.gamma,
            gae_lambda=ppo_cfg.gae_lambda,
            learning_rate=_lr_schedule,
            ent_coef=run_ent_coef_start,
            clip_range=run_clip_range,
            n_epochs=ppo_cfg.n_epochs,
            vf_coef=ppo_cfg.vf_coef,
            policy_kwargs={"net_arch": list(ppo_cfg.net_arch)},
            target_kl=run_target_kl,
            tensorboard_log=tensorboard_log,
            seed=run_seed,
            verbose=1,
        )
        if use_anchor:
            assert isinstance(model, AnchoredPPO)
            reference = PPO.load(str(anchor_ref_model_path), env=train_env)
            model.set_reference_policy(reference.policy, anchor_coef=run_anchor_coef)

    run_config = {
        "total_timesteps": run_steps,
        "n_envs": num_envs,
        "initial_target_altitude_m": start_target,
        "fixed_final_target_altitude_m": fixed_final_target,
        "curriculum_enabled": use_curriculum,
        "ppo": {
            "n_steps": ppo_cfg.n_steps,
            "batch_size": ppo_cfg.batch_size,
            "gamma": ppo_cfg.gamma,
            "gae_lambda": ppo_cfg.gae_lambda,
            "lr_start": run_lr_start,
            "lr_end": run_lr_end,
            "ent_coef_start": run_ent_coef_start,
            "ent_coef_end": run_ent_coef_end,
            "clip_range": run_clip_range,
            "n_epochs": ppo_cfg.n_epochs,
            "vf_coef": ppo_cfg.vf_coef,
            "target_kl": run_target_kl,
            "net_arch": list(ppo_cfg.net_arch),
            "use_lstm": run_use_lstm,
            "lstm_hidden_size": ppo_cfg.lstm_hidden_size if run_use_lstm else None,
            "seed": run_seed,
            "action_repeat": env_cfg.action_repeat,
            "checkpoint_freq_steps": run_checkpoint_freq_steps,
            "eval_freq_steps": run_eval_freq_steps,
            "eval_episodes": run_eval_episodes,
            "official_eval_freq_steps": run_official_eval_freq_steps,
            "official_eval_episodes": run_official_eval_episodes,
            "final_eval_seed_start": ppo_cfg.final_eval_seed_start,
            "official_eval_seed_start": ppo_cfg.official_eval_seed_start,
            "official_success_threshold": ppo_cfg.official_success_threshold,
            "official_success_streak": ppo_cfg.official_success_streak,
            "guardrail_min_success_rate": guardrail_min_success_rate,
            "guardrail_min_q_ok_rate": guardrail_min_q_ok_rate,
            "guardrail_min_g_ok_rate": guardrail_min_g_ok_rate,
            "guardrail_min_success_lcb": guardrail_min_success_lcb,
            "guardrail_min_q_ok_lcb": guardrail_min_q_ok_lcb,
            "guardrail_min_g_ok_lcb": guardrail_min_g_ok_lcb,
            "guardrail_min_timesteps": guardrail_min_timesteps,
            "anchor_ref_model_path": anchor_ref_model_path,
            "anchor_coef": run_anchor_coef,
        },
        "curriculum": curriculum_cfg.__dict__,
        "artifacts": {
            "best_final_model_path": str(best_final_model_path),
            "best_final_vecnorm_path": str(best_final_vecnorm_path),
            "progress_eval_latest_path": str(progress_latest_path),
            "fixed_final_eval_latest_path": str(fixed_final_latest_path),
            "official_fixed_final_eval_latest_path": str(official_latest_path),
        },
        "env": {
            "dt": env_cfg.dt,
            "t_final": env_cfg.t_final,
            "mission": mission.__dict__,
            "atmosphere": atmosphere_cfg.__dict__,
        },
    }
    (artifacts_dir / "train_run_config.json").write_text(json.dumps(run_config, indent=2))

    model.learn(total_timesteps=run_steps, callback=callbacks)

    save_base = artifacts_dir / "ppo_rocket"
    model.save(str(save_base))
    train_env.save(str(artifacts_dir / "vecnormalize.pkl"))
    model_path = save_base.with_suffix(".zip")
    print(f"saved model: {model_path}")
    print(f"best final model: {best_final_model_path}")
    print(f"episode summaries: {summaries_path}")

    train_env.close()
    return model_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO rocket guidance policy")
    parser.add_argument("--total-timesteps", type=int, default=None, help="Override total training timesteps")
    parser.add_argument("--n-envs", type=int, default=None, help="Override number of vectorized environments")
    parser.add_argument(
        "--initial-target-altitude-m",
        type=float,
        default=None,
        help="Override initial curriculum target altitude",
    )
    parser.add_argument(
        "--disable-curriculum",
        action="store_true",
        help="Disable curriculum promotion and keep fixed target altitude",
    )
    parser.add_argument("--ent-coef", type=float, default=None, help="Override PPO entropy coefficient")
    parser.add_argument("--ent-coef-end", type=float, default=None, help="Override PPO entropy coefficient schedule end")
    parser.add_argument("--lr-start", type=float, default=None, help="Override PPO learning-rate schedule start")
    parser.add_argument("--lr-end", type=float, default=None, help="Override PPO learning-rate schedule end")
    parser.add_argument("--clip-range", type=float, default=None, help="Override PPO clip range")
    parser.add_argument("--target-kl", type=float, default=None, help="Override PPO target KL")
    parser.add_argument("--seed", type=int, default=None, help="Override PPO random seed")
    parser.add_argument("--eval-freq-steps", type=int, default=None, help="Override benchmark frequency in timesteps")
    parser.add_argument("--eval-episodes", type=int, default=None, help="Override standard benchmark episode count")
    parser.add_argument(
        "--official-eval-freq-steps",
        type=int,
        default=None,
        help="Override official benchmark frequency in timesteps",
    )
    parser.add_argument(
        "--official-eval-episodes",
        type=int,
        default=None,
        help="Override official benchmark episode count",
    )
    parser.add_argument(
        "--checkpoint-freq-steps",
        type=int,
        default=None,
        help="Override checkpoint save frequency in timesteps",
    )
    parser.add_argument(
        "--guardrail-min-success-rate",
        type=float,
        default=None,
        help="Stop training if fixed/official eval raw success rate drops below this value",
    )
    parser.add_argument(
        "--guardrail-min-q-ok-rate",
        type=float,
        default=None,
        help="Stop training if fixed/official eval raw q_ok rate drops below this value",
    )
    parser.add_argument(
        "--guardrail-min-g-ok-rate",
        type=float,
        default=None,
        help="Stop training if fixed/official eval raw g_ok rate drops below this value",
    )
    parser.add_argument(
        "--guardrail-min-success-lcb",
        type=float,
        default=None,
        help="Stop training if fixed/official eval Wilson 95%% success lower bound drops below this value",
    )
    parser.add_argument(
        "--guardrail-min-q-ok-lcb",
        type=float,
        default=None,
        help="Stop training if fixed/official eval Wilson 95%% q_ok lower bound drops below this value",
    )
    parser.add_argument(
        "--guardrail-min-g-ok-lcb",
        type=float,
        default=None,
        help="Stop training if fixed/official eval Wilson 95%% g_ok lower bound drops below this value",
    )
    parser.add_argument(
        "--guardrail-min-timesteps",
        type=int,
        default=0,
        help="Do not enforce guardrails before this many training timesteps",
    )
    parser.add_argument("--load-model", type=str, default=None, help="Path to model zip for warm-start")
    parser.add_argument("--load-vecnorm", type=str, default=None, help="Path to vecnormalize pkl for warm-start")
    parser.add_argument(
        "--anchor-ref-model",
        type=str,
        default=None,
        help="Frozen plain-PPO reference model for action-MSE anchored fine-tuning",
    )
    parser.add_argument(
        "--anchor-coef",
        type=float,
        default=0.0,
        help="Coefficient for action-MSE anchor to --anchor-ref-model",
    )
    parser.add_argument("--artifacts-dir", type=str, default=None, help="Override artifacts output directory")
    lstm_group = parser.add_mutually_exclusive_group()
    lstm_group.add_argument("--use-lstm", dest="use_lstm", action="store_true", default=None, help="Force RecurrentPPO (LSTM)")
    lstm_group.add_argument("--no-lstm", dest="use_lstm", action="store_false", default=None, help="Force plain PPO (no LSTM)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        initial_target_altitude_m=args.initial_target_altitude_m,
        curriculum_enabled=False if args.disable_curriculum else None,
        ent_coef=args.ent_coef,
        ent_coef_end=args.ent_coef_end,
        lr_start=args.lr_start,
        lr_end=args.lr_end,
        clip_range=args.clip_range,
        target_kl=args.target_kl,
        seed=args.seed,
        eval_freq_steps=args.eval_freq_steps,
        eval_episodes=args.eval_episodes,
        official_eval_freq_steps=args.official_eval_freq_steps,
        official_eval_episodes=args.official_eval_episodes,
        checkpoint_freq_steps=args.checkpoint_freq_steps,
        guardrail_min_success_rate=args.guardrail_min_success_rate,
        guardrail_min_q_ok_rate=args.guardrail_min_q_ok_rate,
        guardrail_min_g_ok_rate=args.guardrail_min_g_ok_rate,
        guardrail_min_success_lcb=args.guardrail_min_success_lcb,
        guardrail_min_q_ok_lcb=args.guardrail_min_q_ok_lcb,
        guardrail_min_g_ok_lcb=args.guardrail_min_g_ok_lcb,
        guardrail_min_timesteps=args.guardrail_min_timesteps,
        load_model_path=args.load_model,
        load_vecnorm_path=args.load_vecnorm,
        anchor_ref_model_path=args.anchor_ref_model,
        anchor_coef=args.anchor_coef,
        artifacts_dir=args.artifacts_dir,
        use_lstm=args.use_lstm,
    )
