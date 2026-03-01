"""Plot rollout diagnostics from telemetry produced by rl.replay."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_diagnostics(
    telemetry_path: str = "artifacts/telemetry.npz",
    out_path: str = "artifacts/diagnostics.png",
) -> Path:
    data = np.load(telemetry_path)
    t = data["t"]
    z = data["z"]
    vx = data["vx"]
    vz = data["vz"]
    q_dyn = data["q_dyn"]
    throttle = data["throttle"]
    r_progress = data.get("reward_progress", np.zeros_like(t))
    r_q = data.get("reward_q_pen", np.zeros_like(t))
    r_g = data.get("reward_g_pen", np.zeros_like(t))
    r_fuel = data.get("reward_fuel_pen", np.zeros_like(t))

    n = min(len(t), len(r_progress))
    tt = t[:n]

    fig, axes = plt.subplots(5, 1, figsize=(11, 14), sharex=True)

    axes[0].plot(t, z, color="#1f77b4", linewidth=1.8)
    axes[0].set_ylabel("altitude [m]")
    axes[0].grid(alpha=0.25)

    axes[1].plot(t, vx, label="vx", color="#2ca02c")
    axes[1].plot(t, vz, label="vz", color="#ff7f0e")
    axes[1].set_ylabel("velocity [m/s]")
    axes[1].legend(loc="best")
    axes[1].grid(alpha=0.25)

    axes[2].plot(t, q_dyn, color="#d62728")
    axes[2].set_ylabel("q_dyn [Pa]")
    axes[2].grid(alpha=0.25)

    axes[3].plot(t, throttle, color="#9467bd")
    axes[3].set_ylabel("throttle")
    axes[3].set_ylim(-0.05, 1.05)
    axes[3].grid(alpha=0.25)

    axes[4].plot(tt, r_progress[:n], label="progress", color="#1f77b4")
    axes[4].plot(tt, r_q[:n], label="q_pen", color="#d62728")
    axes[4].plot(tt, r_g[:n], label="g_pen", color="#ff7f0e")
    axes[4].plot(tt, r_fuel[:n], label="fuel_pen", color="#2ca02c")
    axes[4].set_xlabel("time [s]")
    axes[4].set_ylabel("reward components")
    axes[4].legend(loc="best")
    axes[4].grid(alpha=0.25)

    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved diagnostics plot: {out}")
    return out


if __name__ == "__main__":
    plot_diagnostics()
