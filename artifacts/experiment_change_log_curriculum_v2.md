# Curriculum V2 Change Log

## Objective
Push ascent beyond the ~3 km local optimum toward scalable 100 km behavior.

## Changes Applied
1. Curriculum promotion now uses either rolling success rate OR rolling max-altitude ratio to target.
2. Added `altitude_ratio_threshold=0.85` to curriculum config.
3. Mission success/error criteria are stage-dependent by target altitude:
   - low (<20 km): altitude + q/g dominated, loose terminal kinematics.
   - mid (20-60 km): moderate kinematic/downrange constraints.
   - high (>=60 km): full constraints.
4. Reduced early powered-flight side-objective pressure in potential shaping (downrange/vy/fpa gated by altitude).
5. Lowered early fuel penalty below 40 km to discourage premature throttle suppression.
6. Added anti-stall terminal penalty if no burnout and max altitude remains <65% of target.
7. Added small burnout-progress bonus during non-terminal coast transitions.
8. Extended horizons for long-term credit assignment:
   - `t_final`: 650 s
   - PPO `n_steps`: 4096
   - PPO `gamma`: 0.999

## Files Touched
- `config/defaults.py`
- `rl/train.py`
- `rl/env.py`
