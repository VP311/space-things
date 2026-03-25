# Ralph SCRATCHPAD — S2: Measure and log velocity at burnout per episode

## S2 Pre-implementation Reasoning

**Story:** Add `velocity_at_burnout_mps` and `vz_at_burnout_mps` to episode_summaries.jsonl.

**Hypothesis:** Burnout at 24.1km with mean throttle 0.249 → policy is either (a) velocity-limited at burnout (~1,100 m/s actual vs ~1,220 needed for 100km), or (b) trajectory-angle limited. S2 measures this directly. Key formula: to coast from altitude h_0 to h_max, need vz = sqrt(2g(h_max - h_0)).

**Code path:**
1. `env.py`: `self.burnout_scalars` is set when burnout is detected (line 386-387). It's an `EpisodeScalars` with `.vz` and `.speed` fields.
2. `env.py info_dict()` (line 629): already extracts `burnout_altitude` from `self.burnout_scalars.altitude`. Add `.speed` and `.vz` the same way.
3. `train.py _on_step()` (line 100-119): builds the `row` dict from `info`. Add two new fields.

**Files to touch:**
- `rl/env.py`: `info_dict()` — add 2 new fields to return dict
- `rl/train.py`: `_on_step()` — add 2 new fields to `row` dict

**Risk:** Zero — pure observability. No reward logic touched.

**Confidence:**
- Implementation correct: 9/10
- Hypothesis valid: 10/10 (measuring is always valid)

---

# Previous SCRATCHPAD — S1: Diagnose Run 12 and write ANALYSIS.md

## 1. Current State Summary

**Branch:** ralph/run12-analysis-and-fixes (just created from main)
**Progress.txt:** Only contains the start timestamp; no stories completed yet.
**ANALYSIS.md:** Does not exist yet. This is what S1 creates.

**Run 12 at a glance (from episode_summaries.jsonl, 2510 episodes):**
- Curriculum target: 100km throughout ALL 2510 episodes (never changed — started at 100km)
- Average altitude: 85–88km across all windows (no improving trend)
- Success rate: 3–6.5% per 200-episode window, no trend (total: 104/2510 = 4.14%)
- Mean throttle: 0.249 overall (by altitude: <80km 0.257, 80-90 0.248, 90-100 0.241, >100 0.235)
- Burnout altitude: mean 24.1km ± 1.5km
- Termination: 99.7% apogee, 0.3% crash
- Terminal penalty avg: -58.4 (std 12.3) vs non-terminal rewards sum: +14.1
- Total avg reward: -31.99

**q compliance by altitude band:**
- 0-40km: 57.1% ok (mean q=65.2 kPa, n=7)
- 40-60km: 66.7% ok (mean q=63.7 kPa, n=3)
- 60-80km: 43.0% ok (mean q=69.9 kPa, n=602)
- 80-100km: 58.3% ok (mean q=67.0 kPa, n=1732)
- 100km+: 62.7% ok (mean q=65.7 kPa, n=166)

**Reward by altitude bucket:**
- <80km: avg total=-43.1 (terminal_pen=-61.5, progress=3.0, survival=15.1)
- 80-90km: avg total=-38.5 (terminal_pen=-60.8, progress=3.4, survival=15.9)
- 90-100km: avg total=-36.1 (terminal_pen=-60.6, progress=3.8, survival=16.5)
- >100km: avg total=+66.4 (terminal_pen=-22.6, terminal_bonus ~+100, progress=4.2)

**Reward component averages (all episodes):**
- survival: +15.9, progress: +3.5, tilt_pen: -3.6, q_pen: -0.6, coasting_pen: -0.6
- smooth_pen: -0.4, omega_pen: -0.02, g_pen: 0.0, fuel_pen: -0.01
- terminal_bonus: +12.3 (includes success +100 and burnout micro-bonuses)
- terminal_penalty: -58.4
- NON-terminal sum: +14.1
- TOTAL: -32.0

---

## 2. Story Understanding

**Picking S1** (priority 1, passes: false). All others depend on this analysis.

**What S1 asks:** Parse episode_summaries.jsonl and produce ANALYSIS.md with:
1. q compliance by altitude band
2. Reward-vs-altitude table showing monotonicity
3. At least 3 ranked hypotheses with evidence, confidence, proposed experiments, falsification
4. Commit the file

**Underlying hypothesis:** We don't have a precise picture of what failed in Run 12. S1 is pure analysis — no code changes. The output of S1 is the evidence base for S2-S5.

---

## 3. Code Archaeology

Files to touch:
- **Write:** `/Users/vp311/space things/space-things/ANALYSIS.md` (new file)
- **Update:** `scripts/ralph/prd.json` (set S1 passes: true)
- **Append:** `scripts/ralph/progress.txt`

No source files modified.

---

## 4. Risk Assessment

Zero risk — this is a pure documentation/analysis story. Writing ANALYSIS.md cannot break training.

The only risk is writing incorrect analysis, which would mislead S2-S5. Mitigation: compute everything from the raw JSONL data directly.

---

## 5. Expected Outcome

ANALYSIS.md will be written and committed. Subsequent stories (S2-S5) will reference it for evidence.

The analysis will confirm:
1. Terminal penalty domination (confirmed: -58.4 avg vs +14.1 non-terminal)
2. q compliance is worst at 60-80km band (43.0%), improving slightly at higher altitude
3. Reward is monotonically increasing with altitude but very flat below 100km (gradient ≈ 0.5/km)
4. Policy is throttle-suppressing: 24.9% mean throttle → burnout at only 24km → 86km apogee from ballistic coast
5. No improving trend in success rate over training — hard local optimum

---

## 6. Confidence Score

- Hypothesis valid (analysis will be useful for S2-S5): 10/10
- Implementation correct (data accurately parsed): 9/10 (slight uncertainty: q_compliance is inferred from max_q_dyn, not a per-step ratio)

**Note on q compliance:** The episode data only contains `max_q_dyn` per episode (the worst point). A "compliant" episode means max_q_dyn ≤ 70,000 Pa. The 43% at 60-80km means 57% of episodes that peaked in the 60-80km altitude band exceeded 70k Pa at SOME point during flight.

---

## Key Derived Insights for ANALYSIS.md

**Physics sanity check for throttle:**
- mean_throttle = 0.249
- Effective thrust at 24.9%: 1.5e6 × 0.249 = 373,500 N
- Rocket mass at liftoff: 40,000 kg
- Effective TWR = 373,500 / (40,000 × 9.81) = 0.952g
- This is BARELY above gravity — rocket is hovering at liftoff, not accelerating
- At mid-flight (half fuel, 30t mass): TWR = 373,500 / (30,000 × 9.81) = 1.27g — still weak

**Burnout altitude vs velocity physics:**
- Burnout at ~24km with 24.9% throttle
- Isp ≈ 282s (Falcon 9 Merlin SL), but let's use simulated value
- At 24.9% throttle, burn time ≈ 68s (full throttle) / 0.249 × some adjustment... actually:
  - prop_mass = 20t, mdot = max_thrust × throttle / (Isp × g0)
  - Max mdot = 1.5e6 / (282 × 9.81) ≈ 542 kg/s
  - At 24.9%: mdot = 0.249 × 542 ≈ 135 kg/s
  - Time to burn 20t: 20000 / 135 ≈ 148s of burn time
  - But only reaches 24km... should have been climbing for 148s

**Velocity at burnout (inferred):**
- After 148s of climbing from rest to 24km... with low TWR ~1g:
  - Average net accel ≈ 0.3g (TWR 1.3g minus drag and gravity)
  - v_burnout ≈ 0.3 × 9.81 × 148 ≈ 435 m/s vertical
- But to coast to 100km from 24km: need v²/(2g) = (100k-24k) → v = sqrt(2×9.81×76000) ≈ 1,220 m/s
- 435 m/s << 1,220 m/s needed → policy IS velocity-limited
- The 86km apogee: v = sqrt(2×9.81×62000) ≈ 1,103 m/s → burnout vz ≈ 1,100 m/s??
  Wait — that's inconsistent with 24km burnout and 86km apogee meaning vz at burnout must be ~1100 m/s
  Let me reconsider: at 24.9% throttle for 148s, starting from ~0 m/s...
  Actually, with drag and gravity, the effective ΔV at low throttle is worse.
  The 86km apogee with burnout at 24km implies vz_burnout ≈ sqrt(2×9.81×(86000-24000)) ≈ 1,103 m/s
  So somehow vz at burnout IS ~1,100 m/s despite only 24.9% throttle.
  This means: the ascent guard forces throttle ≥ 72% below 8km → only 24.9% average elsewhere
  The 24.9% mean across the FULL episode includes coast phase (throttle=0 after burnout).
  The powered phase throttle must be higher than 24.9%.

The S2 story will add velocity_at_burnout_mps to make this concrete.

**The core problem:** reward gradient below 100km is ≈ 0.5/km (from -43 at 75km to -36 at 95km over 20km = 0.35/km). The terminal penalty varies only slightly (-61.5 to -60.6) across the 75-95km range. Only above 100km does the terminal_penalty drop dramatically (to -22.6). This means the policy sees essentially the same reward signal at 86km vs 95km — there's no pull to do better below 100km.

---

## Ranked Hypotheses for ANALYSIS.md

### H1: Terminal penalty dominates gradient, suppressing shaping signals (HIGH CONFIDENCE: 9/10)
- Evidence: terminal_penalty avg -58.4 (80% of total negative signal), non-terminal sum only +14.1
- Below 100km, penalty barely changes: -61.5 at <80km vs -60.6 at 90-100km (0.9 difference over 10km range)
- Policy sees nearly identical total reward at 86km vs 95km → no gradient to improve
- Fix: Reduce penalty from -(50+25*err) to -(25+10*err) [S3]
- Falsification: If reducing penalty worsens success rate (policy stops caring about altitude)

### H2: Throttle suppression from q-avoidance leaves ΔV on the table (HIGH CONFIDENCE: 8/10)
- Evidence: mean_throttle=0.249 → TWR barely above 1g; burnout alt only 24km (should be higher with more throttle)
- Policy learned low throttle to avoid q violations (60-80km band is worst: 43% compliance)
- At 24.9% mean throttle, the burn is long and at low efficiency → loses ΔV to gravity losses
- If policy throttled hard through max-Q then backed off, would have better ΔV at burnout
- S2 will measure vz_at_burnout to confirm this
- Fix: Steeper gradient in 80-100km band (S4) to incentivize the extra push, or reduce q penalty relative to progress

### H3: Progress potential gradient too flat below target (MEDIUM CONFIDENCE: 7/10)
- phi = clip(alt/target, 0, 2) is linear → same 0.01 improvement per km at 5km as at 95km
- Progress reward contribution at 86km vs 100km: 4*(100/100 - 86/100) = 0.56 over MANY steps
- Actually phi is a per-STEP potential, not episode total. The episode total progress = 4*(phi_final - phi_start)
- Episode total progress averages +3.5, increasing only weakly with altitude (3.0 at <80km vs 4.2 at >100km)
- Fix: Steeper gradient in 80-100km band [S4]

### H4: Warm-start from earlier run may have introduced sub-optimal policy priors (LOWER CONFIDENCE: 5/10)
- Run 12 was warm-started from an earlier model
- Success rate never exceeds 6.5% in any window despite 2510 episodes
- Hard to distinguish from terminal penalty dominance hypothesis
- The policy CAN reach 100km (166 episodes > 100km) but does so randomly
- Evidence: first 200 episodes already show 3% success → policy arrived pre-trained at ~86km optimum
