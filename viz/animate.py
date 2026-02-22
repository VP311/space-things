"""Create a simple trajectory animation from replay telemetry."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter


def animate_telemetry(
    telemetry_path: str = "artifacts/telemetry.npz",
    output_path: str = "artifacts/trajectory.gif",
) -> Path:
    data = np.load(telemetry_path)
    t = data["t"]
    x = data["x"]
    z = data["z"]
    q_dyn = data["q_dyn"]
    speed = data["speed"]

    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1.0], height_ratios=[1.0, 1.0])
    ax_traj = fig.add_subplot(gs[:, 0])
    ax_q = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[1, 1])

    ax_traj.plot(x, z, color="#c7c7c7", linewidth=1.0, alpha=0.8)
    traj_line, = ax_traj.plot([], [], color="#1f77b4", linewidth=2.0)
    rocket_dot, = ax_traj.plot([], [], "o", color="#d62728", markersize=6)
    text = ax_traj.text(0.02, 0.98, "", transform=ax_traj.transAxes, va="top")

    ax_q.plot(t, q_dyn, color="#c7c7c7", linewidth=1.0, alpha=0.8)
    q_line, = ax_q.plot([], [], color="#2ca02c", linewidth=2.0)

    ax_v.plot(t, speed, color="#c7c7c7", linewidth=1.0, alpha=0.8)
    v_line, = ax_v.plot([], [], color="#9467bd", linewidth=2.0)

    ax_traj.set_title("Trajectory (x-z)")
    ax_traj.set_xlabel("x [m]")
    ax_traj.set_ylabel("z [m]")
    ax_traj.grid(True, alpha=0.3)

    ax_q.set_title("Dynamic Pressure")
    ax_q.set_xlabel("t [s]")
    ax_q.set_ylabel("q_dyn [Pa]")
    ax_q.grid(True, alpha=0.3)

    ax_v.set_title("Speed")
    ax_v.set_xlabel("t [s]")
    ax_v.set_ylabel("speed [m/s]")
    ax_v.grid(True, alpha=0.3)

    x_pad = max(1.0, 0.05 * (float(np.max(x)) - float(np.min(x)) + 1e-6))
    z_pad = max(1.0, 0.05 * (float(np.max(z)) - float(np.min(z)) + 1e-6))
    ax_traj.set_xlim(float(np.min(x) - x_pad), float(np.max(x) + x_pad))
    ax_traj.set_ylim(float(np.min(z) - z_pad), float(np.max(z) + z_pad))
    ax_q.set_xlim(float(t[0]), float(t[-1]))
    ax_v.set_xlim(float(t[0]), float(t[-1]))
    ax_q.set_ylim(0.0, float(np.max(q_dyn) * 1.1 + 1.0))
    ax_v.set_ylim(0.0, float(np.max(speed) * 1.1 + 1.0))

    stride = max(1, len(t) // 300)
    frame_idx = np.arange(0, len(t), stride, dtype=int)
    if frame_idx[-1] != len(t) - 1:
        frame_idx = np.append(frame_idx, len(t) - 1)

    def update(frame_i: int):
        i = int(frame_idx[frame_i])
        traj_line.set_data(x[: i + 1], z[: i + 1])
        rocket_dot.set_data([x[i]], [z[i]])
        q_line.set_data(t[: i + 1], q_dyn[: i + 1])
        v_line.set_data(t[: i + 1], speed[: i + 1])
        text.set_text(f"t={t[i]:.2f}s\nz={z[i]:.1f}m\nv={speed[i]:.1f}m/s\nq={q_dyn[i]:.1f}Pa")
        return traj_line, rocket_dot, q_line, v_line, text

    anim = FuncAnimation(fig, update, frames=len(frame_idx), interval=50, blit=False)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(out, writer=PillowWriter(fps=20))
    plt.close(fig)
    print(f"saved animation: {out}")
    return out


if __name__ == "__main__":
    animate_telemetry()
