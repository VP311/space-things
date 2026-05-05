# Space Things

Feasibility-first reinforcement learning and imitation learning for a constrained 2D rocket-ascent simulator.

The goal is simple to state and hard to satisfy: reach 100 km while keeping dynamic pressure under 70 kPa and acceleration under 8.5 g. The project started as a PPO training problem and turned into a more honest control-engineering workflow: prove the vehicle can do the mission, generate a safe expert, imitate it, fine-tune carefully, then stress-test the result.

## What Happened

The first instinct was to make PPO learn the ascent from scratch. That did not work reliably. Across many reward/curriculum runs, policies repeatedly found 80-95 km local solutions, often with clean q/g margins but not enough vertical energy to cross 100 km.

The important discovery was not a magic hyperparameter. A deterministic feasibility harness showed the original 20t propellant vehicle was practically energy-limited under the 70 kPa q envelope. Non-RL controllers hit the same roughly 85 km ceiling. After increasing propellant to 25t, scripted q-bucket controllers could reach about 130 km while staying q/g compliant. That changed the problem from "why won't PPO solve this?" to "how do we teach a policy a known safe ascent structure and make it robust?"

Plain behavior cloning from 12 scripted expert flights failed because the policy drifted off the expert trajectory distribution. DAgger fixed that: roll out the failing policy, collect the states it actually visits, relabel them with the scripted expert, and retrain. That produced a strong neural controller. PPO was then useful only as a guarded fine-tuning stage; open-ended PPO tended to trade safety margin for more energy.

## Current Result

Promoted nominal policy:

- `artifacts/living/artifacts_promoted/run38_250k/model.zip`
- `artifacts/living/artifacts_promoted/run38_250k/vecnormalize.pkl`
- Held-out eval: 990/1000 success, 994/1000 q_ok, 997/1000 g_ok

Robust policy:

- `artifacts/living/artifacts_bc_robust/bc_model.zip`
- `artifacts/living/artifacts_bc_robust/vecnormalize.pkl`
- Stress suite: 35 cases x 200 episodes
- Mean stress success improved from 84.1% to 87.9%
- q_ok improved from 98.8% to 100.0%
- g_ok improved from 98.1% to 100.0%

The remaining zero-success stress cases are not being treated as "train harder" targets. A no-RL exact-stress feasibility diagnostic found that compounded propulsion-loss cases top out far below 100 km even under q-safe feedback/CEM controllers. Those are documented as out-of-envelope unless the vehicle or mission definition changes.

## Architecture

- `sim/`: dynamics, atmosphere, vehicle dataclasses, telemetry helpers.
- `rl/env.py`: Gymnasium rocket-ascent environment, reward, terminal success checks, optional stress noise/lag.
- `rl/train.py`: PPO training, warm starts, checkpoint+VecNormalize saving, eval guardrails.
- `rl/behavior_clone.py`: supervised behavior cloning into the PPO MlpPolicy architecture.
- `rl/dagger.py`: DAgger relabeling from policy rollouts and stress-failure episodes.
- `rl/stress_eval.py`: named robustness/OOD stress suite.
- `rl/stress_feasibility.py`: no-RL controller/CEM feasibility probes under exact stress cases.
- `rl/select_checkpoint.py` and `rl/promote_policy.py`: Wilson-score checkpoint selection and promotion manifests.

## Reproduce Key Checks

Run tests:

```bash
python3 -m unittest tests.test_training_pipeline tests.test_env_mission tests.test_dynamics
```

Run nominal promoted eval:

```bash
python3 -m rl.eval \
  --model-path artifacts/living/artifacts_promoted/run38_250k/model.zip \
  --vecnorm-path artifacts/living/artifacts_promoted/run38_250k/vecnormalize.pkl \
  --out-path artifacts/living/artifacts_promoted/run38_250k/eval_check.json \
  --n-episodes 1000 \
  --seed-start 10000 \
  --target-altitude-m 100000 \
  --skip-baseline \
  --deterministic
```

Run the robustness suite:

```bash
python3 -m rl.stress_eval \
  --model-path artifacts/living/artifacts_bc_robust/bc_model.zip \
  --vecnorm-path artifacts/living/artifacts_bc_robust/vecnormalize.pkl \
  --out-dir artifacts/living/artifacts_robustness/bc_robust \
  --n-episodes 200 \
  --skip-scripted
```

Run exact-stress feasibility for remaining hard cases:

```bash
python3 -m rl.stress_feasibility \
  --out-dir artifacts/living/artifacts_robustness/stress_feasibility_suspects \
  --cases isp_0.85x,joint_harsh_propulsion,joint_worst_plausible \
  --n-episodes 3 \
  --seed-start 25000 \
  --cem-population 4 \
  --cem-generations 2 \
  --cem-restarts 1 \
  --cem-train-episodes 2
```

## Artifact Policy

Generated training/eval artifacts are intentionally ignored by git. The canonical evidence currently lives in:

- `artifacts/living/` for active evidence used by the current project story
- `artifacts/living/artifacts_promoted/run38_250k/`
- `artifacts/living/artifacts_dagger_robust/`
- `artifacts/living/artifacts_bc_robust/`
- `artifacts/living/artifacts_robustness/run38_full/`
- `artifacts/living/artifacts_robustness/bc_robust/`
- `artifacts/living/artifacts_robustness/stress_feasibility_suspects/`
- `artifacts/archived/old_ppo_runs/` for the failed historical PPO runs
- `artifacts/archived/legacy_artifacts/` for the old tracked artifact folder

Older run folders are historical evidence, not active source. They should be archived before any deletion.

## Honest Claim

This is not real flight software and it does not prove robustness to real 6-DOF dynamics, unmodeled engine behavior, real sensors, or a real atmosphere. The credible claim is narrower and stronger:

> Built a feasibility-first constrained-flight learning pipeline that diagnosed an infeasible vehicle setup, generated q-safe expert trajectories, used DAgger/BC to learn a neural ascent controller, guarded PPO fine-tuning against safety drift, and validated robustness under structured OOD stress cases.
