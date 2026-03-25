# Run 12 Analysis

**Generated:** 2026-03-25 from `artifacts/episode_summaries.jsonl` (2510 episodes, 5M steps)
**Goal:** 100km altitude, max_q_dyn ≤ 70 kPa, g_load ≤ 8.5g
**Outcome:** 4.14% overall success (104/2510), avg alt 85–88km, no improving trend

---

## 1. q Compliance by Altitude Band

q compliance = fraction of episodes whose `max_q_dyn ≤ 70,000 Pa` (the mission limit),
grouped by the episode's `max_altitude_m`.

| Altitude Band | n episodes | q OK % | Mean max-q (Pa) | Mean throttle |
|--------------|-----------|--------|----------------|---------------|
| 0–40 km      | 7         | 57.1%  | 65,206         | 0.257         |
| 40–60 km     | 3         | 66.7%  | 63,747         | 0.278         |
| 60–80 km     | 602       | 43.0%  | 69,885         | 0.257         |
| **80–100 km**| **1,732** | **58.3%** | **67,007**  | **0.245**     |
| 100 km+      | 166       | 62.7%  | 65,734         | 0.235         |

**Key observations:**
- The 60–80 km band has the worst compliance (43%), even though max-q typically occurs in the lower atmosphere (<20 km). This band likely contains the minority of episodes that throttled too hard and q-violated, then failed to reach 80 km.
- The majority band (80–100 km, 69% of all episodes) shows 58.3% compliance — meaning 41.7% of episodes that reached ~87 km altitude still recorded a max-q violation somewhere earlier in flight.
- q compliance does not improve over training time (see rolling window below).
- Successfully flying to 100 km requires clearing q AND altitude — only 4.14% of episodes achieve both.

---

## 2. Reward vs Altitude — Monotonicity Check

| Altitude Bucket | n   | Avg Total Reward | terminal_penalty | progress | survival |
|----------------|-----|-----------------|-----------------|----------|----------|
| < 80 km        | 612 | **−43.1**        | −61.5           | +3.0     | +15.1    |
| 80–90 km       | 1018| **−38.5**        | −60.8           | +3.4     | +15.9    |
| 90–100 km      | 714 | **−36.1**        | −60.6           | +3.8     | +16.5    |
| > 100 km       | 166 | **+66.4**        | −22.6           | +4.2     | +17.2    |

**Reward IS monotonically increasing** but the gradient is severely distorted:

- Below 100 km: reward improves by only **≈0.35/km** (from −43 at 75 km avg to −36 at 95 km avg over 20 km).
- Crossing 100 km: reward jumps by **≈+103** (from −36 to +66) — driven almost entirely by the terminal_penalty dropping from −60.6 to −22.6, plus the +100 success bonus.
- This creates a **near-flat gradient below target with a cliff at the goal**. The policy sees almost the same reward signal at 86 km and 95 km, giving it no gradient to improve in the 80–100 km band.

**Gradient comparison:**
- Progress reward at 86 km vs 95 km episode: ~4*(0.86−0.86) ≈ 0 per step (potential based on current alt, not episode avg)
- Terminal penalty at 86 km vs 95 km: ~−60.8 vs −60.6 = **0.2 difference** — negligible
- At 100 km+: terminal penalty = −22.6 — a **+38 improvement** that only appears after crossing threshold

---

## 3. Reward Component Averages (All 2510 Episodes)

| Component     | Avg per episode |
|--------------|----------------|
| survival      | +15.95         |
| progress      | +3.45          |
| tilt_pen      | −3.65          |
| coasting_pen  | −0.61          |
| q_pen         | −0.63          |
| smooth_pen    | −0.36          |
| omega_pen     | −0.02          |
| g_pen         | 0.00           |
| fuel_pen      | −0.01          |
| **terminal_bonus** | +12.26    |
| **terminal_penalty** | **−58.4** |
| **TOTAL**     | **−32.0**      |

**Non-terminal sum:** +14.1
**terminal_penalty alone:** −58.4 (accounts for **80% of the total negative signal**)

---

## 4. Success Rate Over Training (No Improving Trend)

| Episode Window | Success Rate | Avg Altitude |
|---------------|-------------|-------------|
| ep 1–200      | 3.0%        | 87.8 km     |
| ep 201–400    | 3.0%        | 86.5 km     |
| ep 401–600    | 5.5%        | 87.1 km     |
| ep 601–800    | 5.5%        | 86.6 km     |
| ep 801–1000   | 4.0%        | 86.6 km     |
| ep 1001–1200  | 4.5%        | 85.7 km     |
| ep 1201–1400  | 3.0%        | 86.7 km     |
| ep 1401–1600  | 5.0%        | 87.4 km     |
| ep 1601–1800  | 3.5%        | 84.9 km     |
| ep 1801–2000  | 4.5%        | 85.8 km     |
| ep 2001–2200  | 6.5%        | 86.3 km     |
| ep 2201–2400  | 1.0%        | 85.0 km     |
| ep 2401–2510  | 5.5%        | 86.1 km     |

Policy entered training already at ~87 km average (from warm-start) and stayed there for **all 2510 episodes**. Curriculum target was **fixed at 100 km throughout** (never advanced — no promotion triggered).

---

## 5. Burnout Characteristics

| Metric | Value |
|--------|-------|
| Mean burnout altitude | 24.1 km (± 1.5 km) |
| Mean burnout altitude range | 20.5 – 28.2 km |
| Mean episode throttle | 0.249 |
| Termination: apogee | 99.7% |
| Termination: crash | 0.3% |

**Physics implication:** Burnout at 24 km → to coast to 100 km requires vz_burnout ≥ sqrt(2×9.81×76,000) ≈ 1,220 m/s. The policy reaches ~87 km average, implying vz_burnout ≈ 1,100 m/s. This is 90% of the needed ΔV. The 14 km gap is the marginal shortfall.

**Throttle analysis:** Mean throttle 0.249 looks critically low, but this includes the coast phase (throttle=0). The powered-phase mean is higher (ascent guard forces ≥ 72% below 8 km). However, the low overall mean suggests the policy spends significant time at very low throttle above 8 km — likely throttle-suppression from q-avoidance.

---

## 6. Ranked Hypotheses

### H1: Terminal penalty dominates gradient, preventing learning in the 80–100 km band
**Confidence: 9/10**

**Evidence:**
- terminal_penalty avg = −58.4 vs non-terminal sum = +14.1 (penalty is 4× larger in magnitude)
- Below 100 km, terminal penalty barely changes with altitude: −61.5 at <80 km vs −60.6 at 90–100 km (only 0.9 difference over 10 km)
- The policy sees nearly identical reward at 86 km and 95 km → no gradient to improve
- 2510 episodes with no average altitude improvement confirms hard local optimum

**Proposed experiment (S3):** Reduce terminal failure penalty from `−(50 + 25×err)` to `−(25 + 10×err)`.
- At 86 km miss (err = 0.14): new penalty = −(25 + 10×0.14) = −26.4 vs old −53.5. Terminal_penalty drops from 67% to ~48% of total signal, amplifying shaping reward influence.

**Falsification:** If reducing the penalty causes avg altitude to **drop** (policy no longer motivated to reach 100 km at all), the hypothesis is wrong. Expected to see more variance in altitude outcomes as the policy explores more.

---

### H2: Progress potential too flat in 80–100 km band to pull policy off plateau
**Confidence: 7/10**

**Evidence:**
- phi = clip(alt/target, 0, 2) is linear → progress reward at 86 km = progress reward at 60 km (same per-km value)
- Per-episode progress reward: +3.0 at <80 km, +3.4 at 80–90 km, +3.8 at 90–100 km — only +0.8 improvement over 20 km band
- The final 14 km gap to 100 km produces only ≈0.56 extra progress over the full episode

**Proposed experiment (S4):** Add a bonus multiplier in the 80–100 km band: `phi = clip(alt/target, 0, 1) + 0.5 × clip((alt − 0.8×target) / (0.2×target), 0, 1)`. This makes the 80–100 km band worth 3.5× more than the current linear formula.

**Falsification:** If avg altitude increases but success rate does not (q compliance degrades), the steeper gradient is causing the policy to overshoot with q violations. Success AND altitude must both improve.

---

### H3: Throttle suppression from q-avoidance causes ΔV deficit at burnout
**Confidence: 8/10**

**Evidence:**
- Mean throttle = 0.249 (25% of max thrust)
- Policy is 90% of the way to needed burnout velocity but systematically short
- 60–80 km band has 43% q compliance (worst band) — suggests episodes that pushed through this altitude aggressively faced q violations
- S2 will add `vz_at_burnout_mps` to confirm: if vz_burnout < 1,100 m/s in failed episodes, confirms ΔV deficit

**Proposed experiment (S2 — observability):** Instrument `vz_at_burnout_mps` and `velocity_at_burnout_mps` in telemetry to quantify the ΔV gap per episode.

**Falsification:** If vz_at_burnout is already >1,100 m/s in failed episodes that reach 85–90 km, then ΔV isn't the bottleneck — atmospheric drag during coast phase or trajectory angle is.

---

## 7. Summary of Interventions for Run 13

Based on the above:

1. **S2 (observability):** Add `velocity_at_burnout_mps` and `vz_at_burnout_mps` to telemetry — zero risk, pure measurement.
2. **S3 (reward-change):** Reduce terminal penalty coefficient: `−(50+25×err)` → `−(25+10×err)` to amplify shaping signals.
3. **S4 (reward-change):** Steepen progress potential in 80–100 km band — only implement AFTER S3 is validated, or batch together for Run 13.

Do NOT implement S3 and S4 in the same training run without first validating each independently — they act in the same direction and the combined effect may overshoot.
