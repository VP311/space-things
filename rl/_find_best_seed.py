"""Find best replay seed at the final curriculum target and save telemetry for it."""
import sys
import numpy as np
from stable_baselines3 import PPO
from config.defaults import AtmosphereConfig, EnvConfig, MissionConfig, default_vehicle_params
from rl.env import RocketAscentEnv
from viz.animate import animate_telemetry
from pathlib import Path

CURRICULUM_TARGET = 14_000.0
params = default_vehicle_params()
cfg = EnvConfig()
mission = MissionConfig()
atm = AtmosphereConfig()
model = PPO.load("artifacts/ppo_rocket.zip")

best_apogee, best_seed = 0, 0
results = []

for seed in range(50):
    env = RocketAscentEnv(params, mission, atm, dt=cfg.dt, t_final=cfg.t_final, record=False)
    env.set_curriculum(target_altitude_m=CURRICULUM_TARGET)
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    apogee = info["max_altitude_m"]
    results.append({"seed": seed, "apogee": apogee, "success": info["success"], "reason": info["termination_reason"]})
    if apogee > best_apogee:
        best_apogee = apogee
        best_seed = seed

results.sort(key=lambda r: r["apogee"], reverse=True)
print("Top 10 seeds:")
for r in results[:10]:
    print(f"  seed={r['seed']:3d}  apogee={r['apogee']/1000:.2f}km  success={r['success']}  reason={r['reason']}")
print(f"Best seed={best_seed}  apogee={best_apogee/1000:.2f}km")
print(f"Success rate: {sum(r['success'] for r in results)/len(results):.2f}")
print(f"Avg apogee:   {sum(r['apogee'] for r in results)/len(results)/1000:.2f}km")

# Now record telemetry for the best seed
print(f"\nRecording telemetry for seed {best_seed}...")
env = RocketAscentEnv(params, mission, atm, dt=cfg.dt, t_final=cfg.t_final, record=True)
env.set_curriculum(target_altitude_m=CURRICULUM_TARGET)
obs, info = env.reset(seed=best_seed, options={"record": True})
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, _, terminated, truncated, info = env.step(action)
    done = terminated or truncated

samples = env.telemetry
t  = np.array([s.t          for s in samples], dtype=np.float32)
x  = np.array([s.pos[0]     for s in samples], dtype=np.float32)
y  = np.array([s.pos[1]     for s in samples], dtype=np.float32)
z  = np.array([s.altitude   for s in samples], dtype=np.float32)
vx = np.array([s.vel[0]     for s in samples], dtype=np.float32)
vz = np.array([s.vel[2]     for s in samples], dtype=np.float32)
q  = np.array([s.q_dyn      for s in samples], dtype=np.float32)
sp = np.array([s.speed      for s in samples], dtype=np.float32)
ms = np.array([s.mass       for s in samples], dtype=np.float32)
th = np.array([s.control.throttle     for s in samples], dtype=np.float32)
gp = np.array([s.control.gimbal_pitch for s in samples], dtype=np.float32)

zeros = np.zeros(len(t), dtype=np.float32)
np.savez("artifacts/telemetry.npz",
    t=t, x=x, y=y, z=z, vx=vx, vz=vz,
    q_dyn=q, speed=sp, mass=ms, throttle=th, gimbal_pitch=gp,
    reward_progress=zeros, reward_q_pen=zeros, reward_g_pen=zeros,
    reward_fuel_pen=zeros, reward_tilt_pen=zeros, reward_smooth_pen=zeros,
)
print(f"Max altitude in recording: {float(np.max(z))/1000:.2f} km")
animate_telemetry()
